import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO


# Paths and model settings
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "test_70_best.pt"
VALIDATION_IMAGES_DIR = PROJECT_ROOT / "dataset" / "soccer_70" / "images" / "val"
VALIDATION_LABELS_DIR = PROJECT_ROOT / "dataset" / "soccer_70" / "labels" / "val"
OUTPUT_CSV = PROJECT_ROOT / "output" / "compare_sahi_results.csv"
MODEL_TYPE = "ultralytics"
MODEL_IMAGE_SIZE = 640
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BALL_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.10

# Evaluation settings
IOU_THRESHOLD = 0.50
MAX_IMAGES = None
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# SAHI sliced-inference settings
SLICE_HEIGHT = 640
SLICE_WIDTH = 640
OVERLAP_HEIGHT_RATIO = 0.20
OVERLAP_WIDTH_RATIO = 0.20
PERFORM_STANDARD_PREDICTION = True
BATCH_SIZE = 4
POSTPROCESS_TYPE = "GREEDYNMM"
POSTPROCESS_MATCH_METRIC = "IOS"
POSTPROCESS_MATCH_THRESHOLD = 0.50
POSTPROCESS_CLASS_AGNOSTIC = False
FORCE_POSTPROCESS_TYPE = False
SAHI_VERBOSE = 0


@dataclass
class DetectionCounts:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    def add(self, other: "DetectionCounts") -> None:
        self.true_positives += other.true_positives
        self.false_positives += other.false_positives
        self.false_negatives += other.false_negatives

    def scores(self) -> dict[str, float]:
        precision = safe_divide(
            self.true_positives,
            self.true_positives + self.false_positives,
        )
        recall = safe_divide(
            self.true_positives,
            self.true_positives + self.false_negatives,
        )
        f1 = safe_divide(2 * precision * recall, precision + recall)

        # Object detection has no countable set of true-negative background
        # boxes, so this uses TP / (TP + FP + FN), also known as the Jaccard
        # score or critical success index.
        accuracy = safe_divide(
            self.true_positives,
            self.true_positives + self.false_positives + self.false_negatives,
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def load_ground_truth_boxes(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> list[list[float]]:
    boxes = []
    if not label_path.is_file():
        return boxes

    for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(
                f"Expected 5 fields in {label_path}:{line_number}, got {len(fields)}"
            )

        class_id = int(fields[0])
        if class_id != BALL_CLASS_ID:
            continue

        center_x, center_y, box_width, box_height = map(float, fields[1:])
        x1 = (center_x - box_width / 2) * image_width
        y1 = (center_y - box_height / 2) * image_height
        x2 = (center_x + box_width / 2) * image_width
        y2 = (center_y + box_height / 2) * image_height
        boxes.append([x1, y1, x2, y2])

    return boxes


def calculate_iou(box_a: list[float], box_b: list[float]) -> float:
    intersection_x1 = max(box_a[0], box_b[0])
    intersection_y1 = max(box_a[1], box_b[1])
    intersection_x2 = min(box_a[2], box_b[2])
    intersection_y2 = min(box_a[3], box_b[3])

    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    box_a_area = max(0.0, box_a[2] - box_a[0]) * max(
        0.0, box_a[3] - box_a[1]
    )
    box_b_area = max(0.0, box_b[2] - box_b[0]) * max(
        0.0, box_b[3] - box_b[1]
    )
    union_area = box_a_area + box_b_area - intersection_area

    return safe_divide(intersection_area, union_area)


def match_detections(
    predictions: list[tuple[list[float], float]],
    ground_truth_boxes: list[list[float]],
) -> DetectionCounts:
    matched_ground_truth = set()
    true_positives = 0

    # Match high-confidence predictions first, with each ground-truth box used
    # at most once. Unmatched predictions are false positives.
    for prediction_box, _ in sorted(
        predictions,
        key=lambda prediction: prediction[1],
        reverse=True,
    ):
        best_iou = 0.0
        best_ground_truth_index = None

        for ground_truth_index, ground_truth_box in enumerate(ground_truth_boxes):
            if ground_truth_index in matched_ground_truth:
                continue

            iou = calculate_iou(prediction_box, ground_truth_box)
            if iou > best_iou:
                best_iou = iou
                best_ground_truth_index = ground_truth_index

        if best_iou >= IOU_THRESHOLD and best_ground_truth_index is not None:
            matched_ground_truth.add(best_ground_truth_index)
            true_positives += 1

    return DetectionCounts(
        true_positives=true_positives,
        false_positives=len(predictions) - true_positives,
        false_negatives=len(ground_truth_boxes) - true_positives,
    )


def predict_without_sahi(model: YOLO, frame) -> list[tuple[list[float], float]]:
    result = model.predict(
        frame,
        classes=[BALL_CLASS_ID],
        conf=CONFIDENCE_THRESHOLD,
        imgsz=MODEL_IMAGE_SIZE,
        device=DEVICE,
        verbose=False,
    )[0]

    return [
        (
            [float(coordinate) for coordinate in box.xyxy[0].tolist()],
            float(box.conf[0].item()),
        )
        for box in result.boxes
    ]


def predict_with_sahi(
    detection_model,
    frame,
) -> list[tuple[list[float], float]]:
    # OpenCV images are BGR; SAHI expects RGB numpy images.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = get_sliced_prediction(
        rgb_frame,
        detection_model,
        slice_height=SLICE_HEIGHT,
        slice_width=SLICE_WIDTH,
        overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
        overlap_width_ratio=OVERLAP_WIDTH_RATIO,
        perform_standard_pred=PERFORM_STANDARD_PREDICTION,
        postprocess_type=POSTPROCESS_TYPE,
        postprocess_match_metric=POSTPROCESS_MATCH_METRIC,
        postprocess_match_threshold=POSTPROCESS_MATCH_THRESHOLD,
        postprocess_class_agnostic=POSTPROCESS_CLASS_AGNOSTIC,
        batch_size=BATCH_SIZE,
        force_postprocess_type=FORCE_POSTPROCESS_TYPE,
        verbose=SAHI_VERBOSE,
    )

    return [
        (
            [float(coordinate) for coordinate in prediction.bbox.to_xyxy()],
            float(prediction.score.value),
        )
        for prediction in result.object_prediction_list
        if prediction.category.id == BALL_CLASS_ID
    ]


def print_results(
    image_count: int,
    without_sahi: DetectionCounts,
    with_sahi: DetectionCounts,
) -> None:
    without_scores = without_sahi.scores()
    with_scores = with_sahi.scores()

    print(f"\nEvaluated {image_count} images at IoU >= {IOU_THRESHOLD:.2f}")
    print(
        f"{'Method':<16} {'TP':>5} {'FP':>5} {'FN':>5} "
        f"{'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10}"
    )
    print("-" * 89)

    for method, counts, scores in (
        ("Without SAHI", without_sahi, without_scores),
        ("With SAHI", with_sahi, with_scores),
    ):
        print(
            f"{method:<16} "
            f"{counts.true_positives:>5} "
            f"{counts.false_positives:>5} "
            f"{counts.false_negatives:>5} "
            f"{scores['precision']:>10.4f} "
            f"{scores['recall']:>10.4f} "
            f"{scores['f1']:>10.4f} "
            f"{scores['accuracy']:>10.4f}"
        )


def write_results(
    image_count: int,
    without_sahi: DetectionCounts,
    with_sahi: DetectionCounts,
) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="") as output_file:
        fieldnames = [
            "method",
            "images",
            "iou_threshold",
            "confidence_threshold",
            "true_positives",
            "false_positives",
            "false_negatives",
            "precision",
            "recall",
            "f1",
            "accuracy",
        ]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for method, counts in (
            ("without_sahi", without_sahi),
            ("with_sahi", with_sahi),
        ):
            writer.writerow(
                {
                    "method": method,
                    "images": image_count,
                    "iou_threshold": IOU_THRESHOLD,
                    "confidence_threshold": CONFIDENCE_THRESHOLD,
                    "true_positives": counts.true_positives,
                    "false_positives": counts.false_positives,
                    "false_negatives": counts.false_negatives,
                    **counts.scores(),
                }
            )


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Could not find model: {MODEL_PATH}")
    if not VALIDATION_IMAGES_DIR.is_dir():
        raise FileNotFoundError(
            f"Could not find validation images: {VALIDATION_IMAGES_DIR}"
        )

    image_paths = sorted(
        path
        for path in VALIDATION_IMAGES_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if MAX_IMAGES is not None:
        image_paths = image_paths[:MAX_IMAGES]
    if not image_paths:
        raise RuntimeError(f"No validation images found in {VALIDATION_IMAGES_DIR}")

    yolo_model = YOLO(str(MODEL_PATH))
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type=MODEL_TYPE,
        model=yolo_model,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        device=DEVICE,
        image_size=MODEL_IMAGE_SIZE,
    )

    without_sahi_counts = DetectionCounts()
    with_sahi_counts = DetectionCounts()

    print(f"Comparing inference methods on {len(image_paths)} images using {DEVICE}")

    for image_number, image_path in enumerate(image_paths, start=1):
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Could not read validation image: {image_path}")

        image_height, image_width = frame.shape[:2]
        label_path = VALIDATION_LABELS_DIR / f"{image_path.stem}.txt"
        ground_truth_boxes = load_ground_truth_boxes(
            label_path,
            image_width,
            image_height,
        )

        without_sahi_predictions = predict_without_sahi(yolo_model, frame)
        with_sahi_predictions = predict_with_sahi(sahi_model, frame)

        without_sahi_counts.add(
            match_detections(without_sahi_predictions, ground_truth_boxes)
        )
        with_sahi_counts.add(
            match_detections(with_sahi_predictions, ground_truth_boxes)
        )

        print(
            f"\rProcessed images: {image_number}/{len(image_paths)}",
            end="",
            flush=True,
        )

    print_results(len(image_paths), without_sahi_counts, with_sahi_counts)
    write_results(len(image_paths), without_sahi_counts, with_sahi_counts)
    print(f"Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
