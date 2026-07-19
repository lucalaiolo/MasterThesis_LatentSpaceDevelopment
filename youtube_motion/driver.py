"""Sweep the two VAE backbones over the three recipes and the mask policies.

This is the entry point for the YouTube 2D-keypoint experiments. It reuses the
whole ``architectures`` stack unchanged — the models, the three training
recipes ([MVAE §3-5]), the six masking policies ([MVAE §2]), and the held-out
MPJPE evaluators ([MVAE §7]) — and only sets ``n_dims=2`` / ``n_joints=18`` so
everything runs on image-plane keypoints instead of 3D mocap.

``run_sweep`` trains one model per valid

    architecture (conv, transformer) x recipe (1, 2, 3) x mask policy

combination, scores each on the *same* time-based validation split, and writes
a single ranked performance table (``results.json`` + ``results.md``) plus a
printed ranking — the 2D analogue of what ``architectures.train.model_selection``
does for one backbone, widened to sweep both backbones and report every run.

CLI:

    python -m youtube_motion.driver --csv path/to/keypoints.csv \
        --out checkpoints/youtube_motion --epochs 100 --device cuda

Import:

    from youtube_motion.data import load_youtube_csv
    from youtube_motion.driver import run_sweep
    bundle = load_youtube_csv("keypoints.csv")
    result = run_sweep(bundle, out_dir="checkpoints/youtube_motion",
                       n_epochs=100, device="cuda")
    print(result["ranked"][0])          # the best (arch, recipe, policy)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
import traceback
from pathlib import Path

import numpy as np

from architectures import TrainingConfig
from architectures.data import build_clips, train_val_split
from architectures.evaluate import evaluate
from architectures.mask_policies import build_policy
from architectures.train import ALL_MASK_POLICIES, ALL_RECIPES, train

from .data import YoutubeMotionBundle
from .skeleton import N_DIMS, N_JOINTS

ALL_ARCHITECTURES: tuple[str, ...] = ("conv", "transformer")


def build_base_config(bundle: YoutubeMotionBundle | None = None,
                      *,
                      architecture: str = "conv",
                      clip_length: int = 32,
                      latent_dim: int = 32,
                      n_epochs: int = 100,
                      batch_size: int = 64,
                      device: str = "cuda",
                      **overrides) -> TrainingConfig:
    """A :class:`TrainingConfig` wired for the 2D YouTube keypoints.

    Fixes the two dataset-specific axes — ``n_dims=2`` (image plane) and
    ``n_joints`` (18 for COCO-18, or whatever the bundle carries) — and leaves
    every training knob at the ``architectures`` default so runs are comparable
    to the 3D pipeline. Any field can be overridden by keyword.

    ``clip_length`` must be divisible by the convolutional downsample factor
    (product of ``conv_strides``, 4 by default); 32 satisfies that.
    """
    n_joints = bundle.n_joints if bundle is not None else N_JOINTS
    fps = 0
    if bundle is not None and bundle.fps:
        med = float(np.median([f for f in bundle.fps if f] or [0]))
        fps = int(round(med)) or 0
    cfg = TrainingConfig(
        architecture=architecture,
        clip_length=clip_length,
        n_joints=n_joints,
        n_dims=N_DIMS,
        latent_dim=latent_dim,
        fps=fps or 30,
        n_epochs=n_epochs,
        batch_size=batch_size,
        device=device,
        **overrides,
    )
    return cfg


def _valid_combo(recipe: int, policy: str, has_limbs: bool) -> str | None:
    """Return a skip-reason string for an invalid combo, else None."""
    if recipe in (2, 3) and policy == "none":
        return "recipes 2 and 3 need a masking policy"
    if policy == "limb" and not has_limbs:
        return "no limbs map on the bundle"
    return None


def run_sweep(bundle: YoutubeMotionBundle,
              base_config: TrainingConfig | None = None,
              *,
              architectures: tuple[str, ...] = ALL_ARCHITECTURES,
              recipes: tuple[int, ...] = ALL_RECIPES,
              mask_policies: tuple[str, ...] = ALL_MASK_POLICIES,
              out_dir: str = "checkpoints/youtube_motion",
              metric: str = "mpjpe_all",
              stride: int | None = None,
              eval_seed: int = 0,
              n_layers_grid: tuple[int, ...] | None = None,
              **config_overrides) -> dict:
    """Train + score every valid (architecture, recipe, mask policy) combo.

    Shared knobs (clip length, latent width, epochs, beta schedule, seed) come
    from ``base_config``; ``architecture``, ``recipe``, ``mask_policy`` and
    ``out_dir`` are set per run. Each run is trained with ``architectures.train``
    and then scored on the time-based validation split with
    ``architectures.evaluate`` — the exact split ``train`` held out internally,
    so no run is scored on data it trained on.

    A run that raises (e.g. CUDA OOM) is recorded with ``status="error"`` and the
    sweep continues, so one bad combo never sinks the whole grid.

    Args:
        bundle: the loaded dataset (:func:`youtube_motion.data.load_youtube_csv`).
        base_config: template config; if None, :func:`build_base_config` is used
            with ``config_overrides`` (e.g. ``n_epochs=``, ``device=``).
        architectures: backbones to sweep. Default ``("conv", "transformer")``.
        recipes: recipes to sweep. Default ``(1, 2, 3)``.
        mask_policies: policies to sweep. Default all six ([MVAE §2]); invalid
            combos are skipped with a printed reason.
        out_dir: root output directory; each run goes to
            ``<out_dir>/<arch>/recipe{N}_{policy}``.
        metric: ranking key — ``"mpjpe_all"`` (reconstruction, default) or
            ``"mpjpe_inpainted"`` (hidden-joint inpainting). Lower is better.
        stride: clip hop; defaults to ``clip_length // 2``.
        eval_seed: seeds the evaluation mask draws.
        n_layers_grid: optional transformer depths to sweep as an extra
            axis (e.g. ``(3, 6, 12)``). Each depth overrides ``n_layers``
            (both encoder and decoder) and its runs land under
            ``<out_dir>/<arch>/L{depth}/recipe{N}_{policy}``. ``None``
            (default) keeps ``base_config``'s depth. The conv backbone
            ignores depth beyond storing it on the record.
        **config_overrides: forwarded to :func:`build_base_config` when
            ``base_config`` is None.
    Returns:
        Dict with ``records`` (one dict per run), ``table`` (records keyed by
        ``(arch, recipe, policy)``), ``ranked`` (records sorted best-first by
        ``metric``, errors last), ``best`` (the top record), and the paths of
        the written ``results.json`` / ``results.md``.
    """
    if base_config is None:
        base_config = build_base_config(bundle, **config_overrides)
    elif config_overrides:
        base_config = dataclasses.replace(base_config, **config_overrides)

    # Force the dataset-specific axes even if a hand-built config forgot them.
    base_config = dataclasses.replace(
        base_config, n_dims=N_DIMS, n_joints=bundle.n_joints)

    if stride is None:
        stride = base_config.clip_length // 2

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    # Rebuild the exact validation split train() uses, once, for scoring.
    clips, video_id, _ = build_clips(bundle.videos, base_config.clip_length,
                                     stride)
    _, val_mask = train_val_split(clips, video_id)
    val_clips = clips[val_mask]
    print(f"[sweep] {len(clips)} clips ({int(val_mask.sum())} val), "
          f"J={base_config.n_joints}, D={base_config.n_dims}, "
          f"clip_length={base_config.clip_length}, stride={stride}")

    records: list[dict] = []
    table: dict[tuple, dict] = {}
    has_limbs = bool(bundle.limbs)
    sweep_depth = n_layers_grid is not None
    depths = list(n_layers_grid) if sweep_depth else [None]

    for arch in architectures:
        for depth in depths:
            for recipe in recipes:
                for policy in mask_policies:
                    skip = _valid_combo(recipe, policy, has_limbs)
                    if skip:
                        dtag = f"L{depth}/" if sweep_depth else ""
                        print(f"[sweep] skip {arch}/{dtag}recipe{recipe}/"
                              f"{policy}: {skip}")
                        continue

                    overrides = dict(
                        architecture=arch, recipe=recipe, mask_policy=policy,
                        mask_limb_names=(list(bundle.limbs) if policy == "limb"
                                         else list(base_config.mask_limb_names)),
                    )
                    if sweep_depth:
                        overrides["n_layers"] = depth
                        sub = root / arch / f"L{depth}" / f"recipe{recipe}_{policy}"
                        key: tuple = (arch, depth, recipe, policy)
                        dtag = f"L{depth} | "
                    else:
                        sub = root / arch / f"recipe{recipe}_{policy}"
                        key = (arch, recipe, policy)
                        dtag = ""
                    cfg = dataclasses.replace(base_config, out_dir=str(sub),
                                              **overrides)
                    print(f"\n[sweep] === {arch} | {dtag}recipe {recipe} | "
                          f"{policy} -> {sub} ===")
                    rec = _run_one(cfg, bundle, val_clips, stride, metric,
                                   eval_seed)
                    records.append(rec)
                    table[key] = rec

    ranked = _rank(records, metric)
    best = ranked[0] if ranked else None

    json_path = root / "results.json"
    md_path = root / "results.md"
    payload = {
        "metric": metric,
        "n_dims": base_config.n_dims,
        "n_joints": base_config.n_joints,
        "clip_length": base_config.clip_length,
        "n_epochs": base_config.n_epochs,
        "dataset": bundle.summary(),
        "records": records,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    md_path.write_text(_render_markdown(payload, ranked))

    _print_ranking(ranked, metric)
    print(f"\n[sweep] wrote {json_path}")
    print(f"[sweep] wrote {md_path}")

    return {"records": records, "table": table, "ranked": ranked,
            "best": best, "results_json": str(json_path),
            "results_md": str(md_path), "base_config": base_config}


def _run_one(cfg: TrainingConfig, bundle: YoutubeMotionBundle,
             val_clips: np.ndarray, stride: int, metric: str,
             eval_seed: int) -> dict:
    """Train one config and score it, returning a flat record dict."""
    base = {"architecture": cfg.architecture, "recipe": cfg.recipe,
            "mask_policy": cfg.mask_policy, "n_layers": cfg.n_layers,
            "n_enc_layers": cfg.encoder_layers(),
            "n_dec_layers": cfg.decoder_layers(), "out_dir": cfg.out_dir}
    t0 = time.time()
    try:
        cfg.validate()
        out = train(cfg, bundle.videos, limbs=bundle.limbs, stride=stride)
        model = out["model"]
        n_params = int(sum(p.numel() for p in model.parameters()))

        pol = build_policy(cfg, limbs=bundle.limbs)
        device = next(model.parameters()).device
        scores = evaluate(model, val_clips, pol,
                          batch_size=cfg.batch_size, device=str(device),
                          seed=eval_seed, recipe=cfg.recipe)

        hist = out["history"]
        train_last = hist["train"][-1] if hist["train"] else {}
        val_last = hist["val"][-1] if hist["val"] else {}
        rec = {
            **base,
            "status": "ok",
            "params": n_params,
            "best_epoch": out.get("best_epoch", -1),
            "checkpoint": (str(out["checkpoint"]) if out.get("checkpoint")
                           else None),
            "train_loss": _get(train_last, "loss"),
            "val_loss": _get(val_last, "loss"),
            "val_rec_full": _get(val_last, "rec_full"),
            "mpjpe_all": float(scores["mpjpe_all"]),
            "mpjpe_visible": float(scores["mpjpe_visible"]),
            "mpjpe_inpainted": float(scores["mpjpe_inpainted"]),
            "seconds": round(time.time() - t0, 1),
        }
        print(f"[run] {cfg.architecture}/recipe{cfg.recipe}/{cfg.mask_policy}: "
              f"{metric}={rec[metric]:.4f}  "
              f"mpjpe_all={rec['mpjpe_all']:.4f}  "
              f"mpjpe_inpainted={rec['mpjpe_inpainted']:.4f}  "
              f"({rec['seconds']:.0f}s)")
        return rec
    except Exception as e:                          # noqa: BLE001 - report, continue
        print(f"[run] ERROR {cfg.architecture}/recipe{cfg.recipe}/"
              f"{cfg.mask_policy}: {e}")
        traceback.print_exc()
        return {**base, "status": "error", "error": str(e),
                "seconds": round(time.time() - t0, 1)}


def _get(d: dict, key: str):
    v = d.get(key)
    return None if v is None else float(v)


def _rank(records: list[dict], metric: str) -> list[dict]:
    """Sort records best-first by ``metric``; errored runs sink to the end."""
    def key(r):
        if r.get("status") != "ok" or r.get(metric) is None:
            return (1, float("inf"))
        return (0, r[metric])
    return sorted(records, key=key)


def _print_ranking(ranked: list[dict], metric: str) -> None:
    print(f"\n[sweep] ranking by {metric} (lower is better):")
    print(f"    {'arch':11s} {'recipe':6s} {'policy':15s} "
          f"{'params':>9s} {'mpjpe_all':>10s} {'mpjpe_inp':>10s}")
    for i, r in enumerate(ranked):
        if r.get("status") != "ok":
            print(f"    {r['architecture']:11s} {r['recipe']:<6d} "
                  f"{r['mask_policy']:15s}  ERROR: {r.get('error','')[:40]}")
            continue
        star = "  <- best" if i == 0 else ""
        print(f"    {r['architecture']:11s} {r['recipe']:<6d} "
              f"{r['mask_policy']:15s} {r['params']:>9d} "
              f"{r['mpjpe_all']:>10.4f} {r['mpjpe_inpainted']:>10.4f}{star}")


def _render_markdown(payload: dict, ranked: list[dict]) -> str:
    """A self-contained markdown report of the sweep."""
    metric = payload["metric"]
    lines = [
        "# YouTube 2D-keypoint VAE sweep",
        "",
        f"- Dataset: {payload['dataset']}",
        f"- Keypoints: J = {payload['n_joints']}, D = {payload['n_dims']} "
        f"(image plane)",
        f"- Clip length: {payload['clip_length']} frames, "
        f"epochs: {payload['n_epochs']}",
        f"- Ranked by **{metric}** (mean per-joint position error, lower is "
        f"better).",
        "",
        "| # | architecture | recipe | mask policy | params | val rec | "
        "mpjpe_all | mpjpe_visible | mpjpe_inpainted | status |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(ranked, 1):
        if r.get("status") != "ok":
            lines.append(
                f"| {i} | {r['architecture']} | {r['recipe']} | "
                f"{r['mask_policy']} | - | - | - | - | - | "
                f"error: {r.get('error','')[:40]} |")
            continue
        lines.append(
            f"| {i} | {r['architecture']} | {r['recipe']} | "
            f"{r['mask_policy']} | {r['params']:,} | "
            f"{_fmt(r.get('val_rec_full'))} | {r['mpjpe_all']:.4f} | "
            f"{r['mpjpe_visible']:.4f} | {r['mpjpe_inpainted']:.4f} | ok |")
    ok = [r for r in ranked if r.get("status") == "ok"]
    if ok:
        best = ok[0]
        lines += [
            "",
            f"**Best:** {best['architecture']} / recipe {best['recipe']} / "
            f"{best['mask_policy']} — {metric} = {best[metric]:.4f}.",
        ]
    return "\n".join(lines) + "\n"


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.4f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Sweep conv + transformer VAEs over recipes and mask "
                    "policies on the YouTube 2D-keypoint dataset.")
    ap.add_argument("--csv", required=True, help="path to the keypoint CSV")
    ap.add_argument("--out", default="checkpoints/youtube_motion",
                    help="output root directory")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--clip-length", type=int, default=32)
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--metric", default="mpjpe_all",
                    choices=["mpjpe_all", "mpjpe_inpainted"])
    ap.add_argument("--preprocess", default="none",
                    choices=["none", "center", "center_scale"])
    ap.add_argument("--max-videos", type=int, default=None,
                    help="cap the number of videos loaded (quick runs)")
    ap.add_argument("--architectures", nargs="+", default=list(ALL_ARCHITECTURES))
    ap.add_argument("--recipes", nargs="+", type=int, default=list(ALL_RECIPES))
    ap.add_argument("--mask-policies", nargs="+", default=list(ALL_MASK_POLICIES))
    args = ap.parse_args(argv)

    from .data import load_youtube_csv
    bundle = load_youtube_csv(args.csv, preprocess=args.preprocess,
                              max_videos=args.max_videos)
    print(f"[data] {bundle.summary()}")

    run_sweep(
        bundle,
        out_dir=args.out,
        architectures=tuple(args.architectures),
        recipes=tuple(args.recipes),
        mask_policies=tuple(args.mask_policies),
        metric=args.metric,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        clip_length=args.clip_length,
        latent_dim=args.latent_dim,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
