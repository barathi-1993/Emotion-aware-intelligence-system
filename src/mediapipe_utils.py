#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MediaPipe utilities for the Emotion-Aware Intelligence System.

The uploaded trained model and dataset use a 2004-dimensional feature vector:
501 landmarks × 4 values = pose (33) + face mesh (468), with x/y/z/visibility.

The IEEE Access manuscript discusses MediaPipe face, hand, and body landmarks
as the broader system design. This implementation keeps the model-compatible
pose+face extractor used by the released training data and pretrained model.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


POSE_LANDMARKS = 33
FACE_LANDMARKS = 468
FEATURE_DIM = (POSE_LANDMARKS + FACE_LANDMARKS) * 4


def mediapipe_detection(frame, holistic_model):
    """Run MediaPipe Holistic detection on one BGR frame."""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def _landmark_to_xyzw(landmark):
    """Convert a MediaPipe landmark to [x, y, z, visibility]."""
    visibility = getattr(landmark, "visibility", 1.0)
    return [landmark.x, landmark.y, landmark.z, visibility]


def extract_pose_face_keypoints(results) -> np.ndarray:
    """
    Extract the 2004-dimensional pose+face feature vector.

    Returns
    -------
    np.ndarray
        Shape [2004]. Missing landmarks are filled with zeros.
    """
    if results.pose_landmarks:
        pose = np.array(
            [_landmark_to_xyzw(lm) for lm in results.pose_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        pose = np.zeros(POSE_LANDMARKS * 4, dtype=np.float32)

    if results.face_landmarks:
        face = np.array(
            [_landmark_to_xyzw(lm) for lm in results.face_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        face = np.zeros(FACE_LANDMARKS * 4, dtype=np.float32)

    keypoints = np.concatenate([pose, face]).astype(np.float32)

    if keypoints.shape[0] != FEATURE_DIM:
        raise ValueError(f"Expected {FEATURE_DIM} features, got {keypoints.shape[0]}")

    return keypoints


def draw_landmarks(image, results):
    """Draw face, pose, and hand landmarks for visualization."""
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
        )

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
        )

    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
        )

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )

    return image


def feature_columns() -> List[str]:
    """Return CSV feature column names matching extract_pose_face_keypoints."""
    columns = ["class"]
    for i in range(1, POSE_LANDMARKS + FACE_LANDMARKS + 1):
        columns.extend([f"x{i}", f"y{i}", f"z{i}", f"v{i}"])
    return columns
