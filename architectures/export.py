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
