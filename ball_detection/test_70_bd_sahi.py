from pathlib import Path

import cv2
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


# Paths and model settings
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_VIDEO = PROJECT_ROOT / "videos" / "test_15s.mp4"
OUTPUT_VIDEO = PROJECT_ROOT / "output" / "output_test_15s_ball_sahi.mp4"
MODEL_PATH = PROJECT_ROOT / "models" / "test_70_best.pt"
MODEL_TYPE = "ultralytics"
MODEL_IMAGE_SIZE = 640
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BALL_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.10

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

# Video and annotation settings
VIDEO_CODEC = "mp4v"
POINT_RADIUS = 6
POINT_COLOR = (0, 255, 255)
FONT_SCALE = 0.7
FONT_COLOR = (255, 255, 255)
FONT_THICKNESS = 2


def draw_detections(frame, predictions):
    annotated_frame = frame.copy()

    for prediction in predictions:
        x1, y1, x2, y2 = map(int, prediction.bbox.to_xyxy())
        confidence = float(prediction.score.value)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        cv2.circle(
            annotated_frame,
            (center_x, center_y),
            POINT_RADIUS,
            POINT_COLOR,
            -1,
        )
        cv2.putText(
            annotated_frame,
            f"ball {confidence:.2f}",
            (center_x + 10, center_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            FONT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return annotated_frame


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Could not find model: {MODEL_PATH}")
    if not INPUT_VIDEO.is_file():
        raise FileNotFoundError(f"Could not find input video: {INPUT_VIDEO}")

    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    detection_model = AutoDetectionModel.from_pretrained(
        model_type=MODEL_TYPE,
        model_path=str(MODEL_PATH),
        confidence_threshold=CONFIDENCE_THRESHOLD,
        device=DEVICE,
        image_size=MODEL_IMAGE_SIZE,
    )

    capture = cv2.VideoCapture(str(INPUT_VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {INPUT_VIDEO}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Could not read the input video's FPS or resolution.")

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*VIDEO_CODEC),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {OUTPUT_VIDEO}")

    print(
        f"Processing {INPUT_VIDEO.name} "
        f"({width}x{height} at {fps:.2f} FPS) on {DEVICE}"
    )

    processed_frames = 0
    frames_with_ball = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            # OpenCV reads BGR, while SAHI expects an RGB numpy image.
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
                verbose=0,
            )

            ball_predictions = [
                prediction
                for prediction in result.object_prediction_list
                if prediction.category.id == BALL_CLASS_ID
            ]
            if ball_predictions:
                frames_with_ball += 1

            writer.write(draw_detections(frame, ball_predictions))

            processed_frames += 1
            progress = (
                f"{processed_frames}/{total_frames}"
                if total_frames > 0
                else str(processed_frames)
            )
            print(f"\rProcessed frames: {progress}", end="", flush=True)
    finally:
        capture.release()
        writer.release()

    print(
        f"\nBall detected in {frames_with_ball} out of "
        f"{processed_frames} frames."
    )
    print(f"Finished. Output saved to {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
