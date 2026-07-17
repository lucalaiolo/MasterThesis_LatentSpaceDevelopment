"""State expression, dynamics, and clinical statistics ([guideline §5-§7]).

Decodes per-walk state sequences, builds occupancy / dwell / transition
metrics (the gait-specific additions the guideline leads with), fits the
clinical mixed-effects models (FoG primary, medication paired, UPDRS), and
draws the Figure 2/3 analogues.

The clinical signal for Parkinsonian gait is expected in **dwell and
transition structure** (freezing = a breakdown in state progression), not
just occupancy, so all three metric families are tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import palette as pal
from .carepd_adapter import H36M_REGIONS


# ---- Decode + metrics ([guideline §5]) ------------------------------------

def decode_all(model, series: list) -> list:
    """Viterbi state path for every walk ([guideline §5.1])."""
    return [model.decode(x) for x in series]


def occupancy(states: np.ndarray, K: int) -> np.ndarray:
    """Fraction of frames in each state (K,)."""
    return np.bincount(states, minlength=K) / max(len(states), 1)


def dwell_and_transitions(states: np.ndarray, K: int, fps: float) -> dict:
    """Dwell + transition metrics for one walk ([guideline §5.3])."""
    # Run lengths per state (dwell times in seconds).
    runs = {k: [] for k in range(K)}
    if len(states):
        cur, length = states[0], 1
        for s in states[1:]:
            if s == cur:
                length += 1
            else:
                runs[int(cur)].append(length / fps)
                cur, length = s, 1
        runs[int(cur)].append(length / fps)
    median_dwell = np.array([np.median(runs[k]) if runs[k] else 0.0
                             for k in range(K)])
    max_dwell = np.array([np.max(runs[k]) if runs[k] else 0.0
                          for k in range(K)])
    dwell_var = np.array([np.std(runs[k]) if len(runs[k]) > 1 else 0.0
                          for k in range(K)])
    # Transition matrix + summaries.
    T = np.zeros((K, K))
    for a, b in zip(states[:-1], states[1:]):
        T[a, b] += 1
    row = T.sum(1, keepdims=True)
    Tn = T / np.clip(row, 1, None)
    n_trans = int((np.diff(states) != 0).sum())
    trans_rate = n_trans / max(len(states) / fps, 1e-9)      # transitions / s
    p = Tn[Tn > 0]
    trans_entropy = float(-(p * np.log(p)).sum() / max(K, 1))
    return {"median_dwell": median_dwell, "max_dwell": max_dwell,
            "dwell_var": dwell_var, "transition": Tn,
            "trans_rate": trans_rate, "trans_entropy": trans_entropy}


def build_metric_table(data, states_list: list, K: int) -> pd.DataFrame:
    """One row per walk: occupancy/dwell/transition metrics + labels ([§5])."""
    fps = data.fps
    rows = []
    for i, st in enumerate(states_list):
        occ = occupancy(st, K)
        dw = dwell_and_transitions(st, K, fps)
        r = dict(data.info.iloc[i])
        for k in range(K):
            r[f"occ_{k}"] = occ[k]
            r[f"dwell_{k}"] = dw["median_dwell"][k]
            r[f"maxdwell_{k}"] = dw["max_dwell"][k]
        r["trans_rate"] = dw["trans_rate"]
        r["trans_entropy"] = dw["trans_entropy"]
        rows.append(r)
    return pd.DataFrame(rows)


def aggregate_to_subject(table: pd.DataFrame) -> pd.DataFrame:
    """Subject-level means, for the short-walk cohorts ([guideline §6]).

    Metric columns (occupancy / dwell / transition) are averaged over a
    subject's walks; the clinical / nuisance labels are subject-constant and
    taken as ``first`` (a single groupby.agg, so numeric labels like ``fog``
    do not collide between a metric mean and a label join).
    """
    metric_cols = [c for c in table.columns
                   if c.startswith(("occ_", "dwell_", "maxdwell_"))
                   or c in ("trans_rate", "trans_entropy")]
    label_cols = [c for c in ("cohort", "fog", "medication", "updrs_gait",
                              "sex") if c in table.columns]
    agg = {c: "mean" for c in metric_cols}
    agg.update({c: "first" for c in label_cols})
    return table.groupby("subject_id").agg(agg).reset_index()


# ---- Clinical mixed-effects models ([guideline §6]) -----------------------

def fit_lme(table: pd.DataFrame, metric_cols: list, clinical: str,
            random_slope: bool = False) -> pd.DataFrame:
    """MixedLM ``metric ~ clinical + cohort + sex + (1|subject)`` ([§6]).

    Bonferroni-corrects the clinical-term p-value over the ``metric_cols``
    (the K states). Rows with missing ``clinical`` are dropped. Returns a
    tidy results table (coef, p, p_bonferroni, sig).
    """
    import statsmodels.formula.api as smf
    df = table.copy()
    df = df[~df[clinical].isna()]
    if df[clinical].dtype == object:
        df[clinical] = df[clinical].astype("category").cat.codes
    n_tests = len(metric_cols)
    covars = [c for c in ("cohort", "sex") if df[c].nunique() > 1]
    fixed = " + ".join([clinical] + covars) if covars else clinical
    rows = []
    for m in metric_cols:
        sub = df[[m, clinical, "subject_id"] + covars].dropna()
        if sub[clinical].nunique() < 2 or sub["subject_id"].nunique() < 3:
            rows.append({"metric": m, "coef": np.nan, "p": np.nan,
                         "p_bonf": np.nan, "sig": False})
            continue
        try:
            re = f"~{clinical}" if random_slope else None
            model = smf.mixedlm(f"{m} ~ {fixed}", sub, groups=sub["subject_id"],
                                re_formula=re)
            res = model.fit(reml=False, method="lbfgs", disp=False)
            coef, p = res.params.get(clinical, np.nan), res.pvalues.get(clinical, np.nan)
        except Exception:
            coef, p = np.nan, np.nan
        pb = min(p * n_tests, 1.0) if p == p else np.nan
        rows.append({"metric": m, "coef": coef, "p": p, "p_bonf": pb,
                     "sig": bool(pb < 0.05) if pb == pb else False})
    return pd.DataFrame(rows)


def clinical_analysis(table: pd.DataFrame, K: int) -> dict:
    """Run the FoG / medication / UPDRS models over all metric families ([§6])."""
    subj = aggregate_to_subject(table)
    families = {"occupancy": [f"occ_{k}" for k in range(K)],
                "dwell": [f"dwell_{k}" for k in range(K)],
                "maxdwell": [f"maxdwell_{k}" for k in range(K)]}
    scalar = ["trans_rate", "trans_entropy"]
    out = {}
    for fam, cols in families.items():
        out[fam] = {
            "fog": fit_lme(subj, cols, "fog").to_dict("records"),
            "medication": fit_lme(subj, cols, "medication",
                                  random_slope=False).to_dict("records"),
            "updrs": fit_lme(subj[subj.cohort == "BMCLab"], cols,
                             "updrs_gait").to_dict("records"),
        }
    out["transition_scalars"] = {
        "fog": fit_lme(subj, scalar, "fog").to_dict("records")}
    return out


# ---- State characterisation ([guideline §7]) ------------------------------

def state_joint_velocity(data, states_list: list, K: int) -> np.ndarray:
    """Mean per-joint velocity magnitude per state (Fig 3a) ([guideline §7.1]).

    Uses the Set-A joint block of each walk's features. Returns (K, J).
    """
    J = 17
    num = np.zeros((K, J))
    den = np.zeros(K)
    for x, st in zip(data.features, states_list):
        joints = x[:, :J * 3].reshape(len(x), J, 3)
        vel = np.zeros_like(joints)
        vel[1:] = np.diff(joints, axis=0)
        mag = np.linalg.norm(vel, axis=-1)                  # (T, J)
        for k in range(K):
            m = st == k
            if m.any():
                num[k] += mag[m].sum(0)
                den[k] += m.sum()
    return num / np.clip(den[:, None], 1, None)


def state_region_breakdown(state_joint_vel: np.ndarray) -> dict:
    """High-velocity body-region breakdown per state (Fig 3b) ([guideline §7.2])."""
    out = {}
    for region, joints in H36M_REGIONS.items():
        out[region] = state_joint_vel[:, joints].mean(axis=1)   # (K,)
    return out


# ---- Figures ([guideline §5, §7]) -----------------------------------------

def plot_state_sequences(states_list, info, K, out_dir, n=8,
                         name="fig2a_sequences.png"):
    plt = pal.import_matplotlib()
    picks = list(range(min(n, len(states_list))))
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(picks) + 1))
    for row, i in enumerate(picks):
        st = states_list[i]
        ax.scatter(np.arange(len(st)), np.full(len(st), row), c=[pal.state_color(s) for s in st], s=4, marker="s")
    ax.set_yticks(range(len(picks)))
    ax.set_yticklabels([info.iloc[i]["recording"].split(":")[-1] for i in picks], fontsize=7)
    ax.set_xlabel("frame")
    ax.set_title(f"Per-walk state sequences (Fig 2a), K={K}")
    fig.tight_layout()
    return pal.save_fig(fig, out_dir, name)


def plot_group_occupancy(table, K, group_col, out_dir,
                         name="fig2b_occupancy.png"):
    plt = pal.import_matplotlib()
    df = table[~table[group_col].isna()]
    groups = sorted(df[group_col].unique().tolist())
    fig, ax = plt.subplots(figsize=(1.1 * K + 2, 4))
    w = 0.8 / max(len(groups), 1)
    for gi, g in enumerate(groups):
        occ = df[df[group_col] == g][[f"occ_{k}" for k in range(K)]].mean()
        ax.bar(np.arange(K) + gi * w - 0.4 + w / 2, occ.values, w,
               label=f"{group_col}={g}")
    ax.set_xlabel("state")
    ax.set_ylabel("mean occupancy")
    ax.set_title(f"Group-average occupancy by {group_col} (Fig 2b/2c)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return pal.save_fig(fig, out_dir, name)


def plot_state_characterisation(region_breakdown, K, out_dir,
                                name="fig3b_regions.png"):
    plt = pal.import_matplotlib()
    regions = list(region_breakdown.keys())
    M = np.stack([region_breakdown[r] for r in regions], axis=1)   # (K, R)
    fig, ax = plt.subplots(figsize=(1.0 * len(regions) + 2, 0.5 * K + 2))
    im = ax.imshow(M, cmap=pal.HEATMAP_CMAP, aspect="auto")
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(regions)
    ax.set_yticks(range(K))
    ax.set_ylabel("state")
    ax.set_title("Per-state body-region velocity (Fig 3b)")
    fig.colorbar(im, ax=ax, label="mean velocity magnitude")
    fig.tight_layout()
    return pal.save_fig(fig, out_dir, name)
