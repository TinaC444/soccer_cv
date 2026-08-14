import os
from pathlib import Path

import matplotlib

if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
    try:
        import tkinter  # noqa: F401

        matplotlib.use("TkAgg")
    except ImportError:
        pass

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs"
RUN_PATTERN = "test_70_ball*/results.csv"
PLOT_NAME = "training_curves.png"
NON_INTERACTIVE_BACKENDS = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}


def find_latest_results() -> Path:
    result_files = list(RUNS_DIR.glob(RUN_PATTERN))
    if not result_files:
        raise FileNotFoundError(
            f"No test_70 training results were found in {RUNS_DIR}"
        )

    return max(result_files, key=lambda path: path.stat().st_mtime)


def main() -> None:
    results_path = find_latest_results()
    results = pd.read_csv(results_path)
    results.columns = results.columns.str.strip()

    best_row = results.loc[results["metrics/mAP50-95(B)"].idxmax()]
    print(f"Results: {results_path}")
    print(f"Best epoch: {int(best_row['epoch'])}")
    print(f"precision: {best_row['metrics/precision(B)']:.4f}")
    print(f"recall: {best_row['metrics/recall(B)']:.4f}")
    print(f"mAP50: {best_row['metrics/mAP50(B)']:.4f}")
    print(f"mAP50-95: {best_row['metrics/mAP50-95(B)']:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(results["epoch"], results["train/box_loss"], label="Train")
    axes[0, 0].plot(results["epoch"], results["val/box_loss"], label="Validation")
    axes[0, 0].set_title("Box loss")
    axes[0, 0].legend()

    axes[0, 1].plot(results["epoch"], results["train/cls_loss"], label="Train")
    axes[0, 1].plot(results["epoch"], results["val/cls_loss"], label="Validation")
    axes[0, 1].set_title("Classification loss")
    axes[0, 1].set_yscale("symlog", linthresh=1)
    axes[0, 1].legend()

    axes[1, 0].plot(results["epoch"], results["train/dfl_loss"], label="Train")
    axes[1, 0].plot(results["epoch"], results["val/dfl_loss"], label="Validation")
    axes[1, 0].set_title("Distribution focal loss")
    axes[1, 0].legend()

    axes[1, 1].plot(
        results["epoch"],
        results["metrics/mAP50(B)"],
        label="mAP50",
    )
    axes[1, 1].plot(
        results["epoch"],
        results["metrics/mAP50-95(B)"],
        label="mAP50-95",
    )
    axes[1, 1].axvline(
        best_row["epoch"],
        color="black",
        linestyle="--",
        alpha=0.5,
        label=f"Best epoch ({int(best_row['epoch'])})",
    )
    axes[1, 1].set_title("Validation accuracy")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)

    fig.suptitle(results_path.parent.name)
    fig.tight_layout()

    plot_path = results_path.parent / PLOT_NAME
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {plot_path}")

    backend = matplotlib.get_backend().lower()
    if backend in NON_INTERACTIVE_BACKENDS:
        print(
            f"Matplotlib is using the non-interactive {backend!r} backend, "
            "so a plot window cannot open in this terminal. Open the saved PNG instead."
        )
    else:
        plt.show(block=True)


if __name__ == "__main__":
    main()
