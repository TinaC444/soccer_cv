from pathlib import Path

import cv2

#processes every video in videos then saves 10th frame as jpg in dataset/raw

BASE_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = BASE_DIR / "videos"
OUTPUT_DIR = BASE_DIR / "dataset" / "raw"
FRAME_INTERVAL = 10
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def extract_frames(video_path: Path) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_name = f"{video_path.stem}_{video_path.suffix[1:].lower()}"
    frame_number = 0
    extracted_count = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            if frame_number % FRAME_INTERVAL == 0:
                output_name = f"{source_name}_frame_{frame_number:06d}.jpg"
                output_path = OUTPUT_DIR / output_name
                if not cv2.imwrite(str(output_path), frame):
                    raise RuntimeError(f"Could not write frame: {output_path}")
                extracted_count += 1

            frame_number += 1
    finally:
        capture.release()

    return extracted_count


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(
        path
        for path in VIDEOS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )

    total_extracted = 0
    for video_path in video_paths:
        extracted_count = extract_frames(video_path)
        total_extracted += extracted_count
        print(f"{video_path.name}: extracted {extracted_count} frames")

    print(f"Extracted {total_extracted} frames in total.")


if __name__ == "__main__":
    main()
