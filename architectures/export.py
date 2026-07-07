"""Package a `train_sweep` output directory for download.

`train_sweep` writes each run into `<base>/recipe{N}_{policy}/` with a
generic `best.pt` and `history.json`. Downloading them individually is
tedious and loses the naming — every run's weights file is just
`best.pt`. `package_sweep_outputs` walks that base directory, renames
each run's checkpoint and history after the subfolder, copies the
`plots/` directory alongside, and bundles everything into one zip
archive.

The archive layout is:

    <sweep_dir>_export/
        recipe1_uniform/
            recipe1_uniform.pt
            recipe1_uniform_history.json
            plots/
                <all png files from the run>
        recipe2_uniform/
            ...
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def package_sweep_outputs(sweep_dir: str | Path,
                          archive_path: str | Path | None = None,
                          include_plots: bool = True) -> Path:
    """Bundle a sweep directory into a single zip archive.

    Args:
        sweep_dir: the base directory a `train_sweep` call wrote into.
            Its immediate subdirectories are the individual runs.
        archive_path: destination for the zip. Defaults to
            `<sweep_dir>.zip` next to the sweep folder.
        include_plots: copy the `plots/` directory too. Set False for
            weights-and-history only.
    Returns:
        Path to the written zip archive.
    """
    sweep_dir = Path(sweep_dir).resolve()
    if not sweep_dir.is_dir():
        raise FileNotFoundError(f"sweep_dir not found: {sweep_dir}")

    if archive_path is None:
        archive_path = sweep_dir.with_suffix(".zip")
    archive_path = Path(archive_path).resolve()

    staging = sweep_dir.parent / f"{sweep_dir.name}_export"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    n_runs = 0
    for run_dir in sorted(sweep_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        name = run_dir.name                         # e.g. recipe1_uniform
        dst = staging / name
        dst.mkdir()

        ckpt = run_dir / "best.pt"
        if ckpt.exists():
            shutil.copy2(ckpt, dst / f"{name}.pt")

        history = run_dir / "history.json"
        if history.exists():
            shutil.copy2(history, dst / f"{name}_history.json")

        plots = run_dir / "plots"
        if include_plots and plots.is_dir():
            shutil.copytree(plots, dst / "plots")

        n_runs += 1

    # `make_archive` wants the archive path without the .zip suffix.
    base = archive_path.with_suffix("")
    shutil.make_archive(str(base), "zip",
                        root_dir=str(staging.parent),
                        base_dir=staging.name)
    shutil.rmtree(staging)

    print(f"[export] wrote {n_runs} run(s) to {archive_path}")
    return archive_path


def download_sweep_outputs(sweep_dir: str | Path,
                           archive_path: str | Path | None = None,
                           include_plots: bool = True) -> Path:
    """Package a sweep directory and trigger a Colab browser download.

    A thin wrapper over `package_sweep_outputs` that also calls
    `google.colab.files.download`. Outside Colab it just returns the
    archive path.
    """
    archive = package_sweep_outputs(sweep_dir, archive_path,
                                    include_plots=include_plots)
    try:
        from google.colab import files
        files.download(str(archive))
    except ImportError:
        print(f"[export] not on Colab; archive is at {archive}")
    return archive


def summarize_sweep(sweep_dir: str | Path,
                    out_path: str | Path | None = None,
                    rank_by: str = "val_rec_full_min") -> dict:
    """Read every run's `history.json` and write a unified comparison.

    For each `recipe{N}_{policy}/` subdirectory under `sweep_dir` we
    read `history.json` (the train/val stats dumped by `train`) and
    reduce it to the numbers that matter for picking a winner:

        val_loss_min, val_loss_min_epoch
        val_rec_full_min, val_rec_full_min_epoch     <- default rank
        val_loss_final, val_rec_full_final
        train_rec_full_final
        n_epochs

    `rec_full` is the mean reconstruction MSE on the validation set,
    averaged over batches — the metric we use to compare models.

    Runs are sorted ascending by `rank_by` (lower is better). The
    summary is written to `out_path` (defaults to
    `<sweep_dir>/summary.json`) and returned.

    Args:
        sweep_dir: base folder from `train_sweep`.
        out_path: destination for the unified JSON. Defaults to
            `<sweep_dir>/summary.json`.
        rank_by: which field to sort runs by. Any of the metric keys
            above.
    Returns:
        Dict with `ranked` (list of per-run summaries) and `best` (the
        top entry). Written to disk in the same shape.
    """
    sweep_dir = Path(sweep_dir).resolve()
    if not sweep_dir.is_dir():
        raise FileNotFoundError(f"sweep_dir not found: {sweep_dir}")
    if out_path is None:
        out_path = sweep_dir / "summary.json"
    out_path = Path(out_path)

    runs: list[dict] = []
    for run_dir in sorted(sweep_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        history_path = run_dir / "history.json"
        if not history_path.exists():
            continue
        with open(history_path) as f:
            history = json.load(f)
        train_hist = history.get("train", [])
        val_hist = history.get("val", [])
        if not val_hist:
            continue

        val_losses = [e["loss"] for e in val_hist]
        val_recs = [e["rec_full"] for e in val_hist]
        i_loss = int(min(range(len(val_losses)), key=val_losses.__getitem__))
        i_rec = int(min(range(len(val_recs)), key=val_recs.__getitem__))

        runs.append({
            "run": run_dir.name,
            "n_epochs": len(val_hist),
            "val_loss_min": val_losses[i_loss],
            "val_loss_min_epoch": i_loss,
            "val_rec_full_min": val_recs[i_rec],
            "val_rec_full_min_epoch": i_rec,
            "val_loss_final": val_losses[-1],
            "val_rec_full_final": val_recs[-1],
            "train_rec_full_final": train_hist[-1]["rec_full"] if train_hist else None,
        })

    if not runs:
        raise RuntimeError(f"no runs with history.json found under {sweep_dir}")
    if rank_by not in runs[0]:
        raise ValueError(f"rank_by={rank_by!r} not one of {list(runs[0])[2:]}")

    runs.sort(key=lambda r: r[rank_by])
    summary = {"rank_by": rank_by, "best": runs[0], "ranked": runs}

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    header = (f"{'run':32s}  {'val_loss_min':>13s}  "
              f"{'val_rec_min':>12s}  @ep  {'val_rec_final':>13s}")
    print(f"[summary] {len(runs)} run(s), ranked by {rank_by}, "
          f"written to {out_path}")
    print(header)
    print("-" * len(header))
    for r in runs:
        print(f"{r['run']:32s}  "
              f"{r['val_loss_min']:13.6f}  "
              f"{r['val_rec_full_min']:12.6f}  "
              f"{r['val_rec_full_min_epoch']:>3d}  "
              f"{r['val_rec_full_final']:13.6f}")
    return summary
