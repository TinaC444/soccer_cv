from pathlib import Path

import cv2
from ultralytics import YOLO

#tests existing yolo model's "sports ball"
#also reports how many frames contain the ball


BASE_DIR = Path(__file__).resolve().parent
INPUT_VIDEO = BASE_DIR / "videos" / "test_15s.mp4"
OUTPUT_VIDEO = BASE_DIR / "output" / "output_test_15s_ball.mp4"
MODEL_NAME = "yolo11n.pt"
SPORTS_BALL_CLASS_ID = 32
TEST_BALL = False
CONF = 0.01


def test_ball_det() -> None:
    model = YOLO(MODEL_NAME)
    capture = cv2.VideoCapture(str(INPUT_VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {INPUT_VIDEO}")

    processed_frames = 0
    frames_with_ball = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            result = model.predict(
                frame, classes=[SPORTS_BALL_CLASS_ID], verbose=False, conf=CONF
            )[0]
            processed_frames += 1

            if len(result.boxes) > 0:
                frames_with_ball += 1
    finally:
        capture.release()

    print(
        f"Ball detected in {frames_with_ball} out of "
        f"{processed_frames} frames."
    )


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

            result = model.predict(
                frame, classes=[SPORTS_BALL_CLASS_ID], verbose=False, conf=CONF
            )[0]
            annotated_frame = result.plot()
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
    if TEST_BALL:
        test_ball_det()
    else:
        main()
