#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect pose+face landmark features from webcam using MediaPipe Holistic.

Example
-------
python src/collect_landmarks.py --class-name Happy --output data/processed/coords.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp

from mediapipe_utils import (
    draw_landmarks,
    extract_pose_face_keypoints,
    feature_columns,
    mediapipe_detection,
)


def ensure_header(path: Path):
    """Create CSV with header if it does not exist."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(feature_columns())


def main():
    parser = argparse.ArgumentParser(description="Collect emotion landmarks from webcam.")
    parser.add_argument("--class-name", required=True, type=str, help="Emotion label to record.")
    parser.add_argument("--output", default="data/processed/coords.csv", type=str)
    parser.add_argument("--camera", default=0, type=int)
    parser.add_argument("--max-frames", default=0, type=int, help="0 means unlimited until q is pressed.")
    parser.add_argument("--min-detection-confidence", default=0.5, type=float)
    parser.add_argument("--min-tracking-confidence", default=0.5, type=float)
    args = parser.parse_args()

    output_path = Path(args.output)
    ensure_header(output_path)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    mp_holistic = mp.solutions.holistic
    frame_count = 0

    print("[Collect] Press 'q' to stop.")
    with mp_holistic.Holistic(
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as holistic:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_pose_face_keypoints(results)

            with open(output_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([args.class_name] + keypoints.tolist())

            image = draw_landmarks(image, results)
            cv2.putText(
                image,
                f"Collecting: {args.class_name} | Frames: {frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Emotion Landmark Collection", image)

            frame_count += 1
            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[Collect] Saved {frame_count} frames to {output_path}")


if __name__ == "__main__":
    main()
