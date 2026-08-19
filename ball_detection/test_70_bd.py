import os
import shutil
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset" / "soccer_70"
DATA_YAML = DATASET_DIR / "data.yaml"
PRETRAINED_MODEL = PROJECT_ROOT / "models" / "yolo11n.pt"
TRAINING_RUNS_DIR = PROJECT_ROOT / "runs"
BEST_MODEL_PATH = PROJECT_ROOT / "models" / "test_70_best.pt"
EPOCHS = 100
PATIENCE = 30
BATCH = 8


def main() -> None:
    # data.yaml uses "path: .", so resolve it from the dataset directory.
    os.chdir(DATASET_DIR)

    model = YOLO(str(PRETRAINED_MODEL))
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        patience=PATIENCE,
        batch=BATCH,
        imgsz=1280,
        val=True,
        plots=True,
        project=str(TRAINING_RUNS_DIR),
        name="test_70_ball",
    )

    best_checkpoint = Path(model.trainer.best).resolve()
    best_model = YOLO(str(best_checkpoint))
    metrics = best_model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=1280,
        plots=True,
        project=str(TRAINING_RUNS_DIR),
        name="test_70_ball_val",
    )

    print(f"precision: {metrics.box.mp:.4f}")
    print(f"recall: {metrics.box.mr:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_checkpoint, BEST_MODEL_PATH)
    print(f"Best model: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
