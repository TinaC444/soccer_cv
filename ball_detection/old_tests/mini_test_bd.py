import os
import shutil
from pathlib import Path

from ultralytics import YOLO

#fine tunes yolo on a very small dataset in mini_test,
#copies best checkpoint to models/mini_test_ball_best.pt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset" / "mini_test"
DATA_YAML = DATASET_DIR / "data.yaml"
PRETRAINED_MODEL = PROJECT_ROOT / "models" / "yolo11n.pt"
TRAINING_RUNS_DIR = PROJECT_ROOT / "runs"
BEST_MODEL_PATH = PROJECT_ROOT / "models" / "mini_test_ball_best.pt"


def main() -> None:
    # data.yaml uses "path: .", so resolve it from the dataset directory.
    os.chdir(DATASET_DIR)

    model = YOLO(str(PRETRAINED_MODEL))
    model.train(
        data=str(DATA_YAML),
        epochs=30,
        imgsz=1280,
        val=True,
        plots=False,
        project=str(TRAINING_RUNS_DIR),
        name="mini_test_ball",
    )

    best_checkpoint = Path(model.trainer.best).resolve()
    best_model = YOLO(str(best_checkpoint))
    metrics = best_model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=1280,
        plots=False,
        project=str(TRAINING_RUNS_DIR),
        name="mini_test_ball_validation",
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
