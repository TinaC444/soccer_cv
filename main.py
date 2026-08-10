from pathlib import Path

import cv2
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
INPUT_VIDEO = BASE_DIR / "videos" / "test_15s.mp4"
OUTPUT_VIDEO = BASE_DIR / "output" / "output_test_15s.mp4"
MODEL_NAME = "yolo11n.pt"
FONT_SCALE = 0.7
POINT_TRACKER = True


def draw_detections(frame, result):
    annotated_frame = frame.copy()

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        label = f"{result.names[class_id]} {confidence:.2f}"

        if POINT_TRACKER:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(annotated_frame, (center_x, center_y), 6, (0, 255, 255), -1)
            text_position = (center_x + 10, center_y + 5)
        else:
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text_position = (x1, max(y1 - 8, 20))

        cv2.putText(
            annotated_frame,
            label,
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return annotated_frame


def main() -> None:
    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_NAME)
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

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO), fourcc, fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {OUTPUT_VIDEO}")

    print(f"Processing {INPUT_VIDEO.name} ({width}x{height} at {fps:.2f} FPS)")

    processed_frames = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            result = model.predict(frame, classes=[0], verbose=False)[0]
            annotated_frame = draw_detections(frame, result)
            writer.write(annotated_frame)

            processed_frames += 1
            if total_frames > 0:
                progress = f"{processed_frames}/{total_frames}"
            else:
                progress = str(processed_frames)
            print(f"\rProcessed frames: {progress}", end="", flush=True)
    finally:
        capture.release()
        writer.release()

    print(f"\nFinished. Output saved to {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
