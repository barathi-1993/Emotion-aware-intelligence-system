#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time emotion recognition using webcam + MediaPipe + trained ML classifier.

Example
-------
python src/realtime_inference.py --model models/gradient_boosting_emotion_model.pkl --camera 0
"""

from __future__ import annotations

import argparse
import pickle

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from mediapipe_utils import (
    draw_landmarks,
    extract_pose_face_keypoints,
    feature_columns,
    mediapipe_detection,
)


def main():
    parser = argparse.ArgumentParser(description="Run real-time emotion recognition.")
    parser.add_argument("--model", default="models/gradient_boosting_emotion_model.pkl", type=str)
    parser.add_argument("--camera", default=0, type=int)
    parser.add_argument("--min-detection-confidence", default=0.5, type=float)
    parser.add_argument("--min-tracking-confidence", default=0.5, type=float)
    args = parser.parse_args()

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    feature_names = feature_columns()[1:]

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    mp_holistic = mp.solutions.holistic

    print("[Inference] Press 'q' to quit.")
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

            X = pd.DataFrame([keypoints], columns=feature_names)
            pred = model.predict(X)[0]

            conf = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                conf = float(np.max(proba))

            image = draw_landmarks(image, results)
            text = f"{pred}" if conf is None else f"{pred}: {conf:.2f}"

            cv2.rectangle(image, (0, 0), (520, 60), (245, 117, 16), -1)
            cv2.putText(
                image,
                text,
                (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Real-Time Emotion Recognition", image)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
