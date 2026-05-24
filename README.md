<h1 align="center">  
  Emotion-Aware Intelligence System <br>
  for Real-Time Personalized Healthcare
</h1>

<p align="center">
  📄 <b>Based on the IEEE Access paper: Digital Twin Model: A Real-Time Emotion Recognition System for Personalized Healthcare</b>
</p>

<p align="center">
  <b>😊 Emotion Recognition</b> &nbsp;|&nbsp;
  <b>📍 MediaPipe Holistic</b> &nbsp;|&nbsp;
  <b>🏥 Personalized Healthcare</b> &nbsp;|&nbsp;
  <b>🧬 Digital Twin</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Paper-IEEE%20Access-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Task-Real--Time%20Emotion%20Recognition-teal?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Feature%20Extractor-MediaPipe-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Model-Gradient%20Boosting-red?style=for-the-badge" />
</p>

<p align="center">
  Official implementation-style repository for an emotion-aware intelligence system that detects patient emotional states in real time using webcam input and MediaPipe landmark features.
</p>

---

## 📚 Table of Contents

- [📌 Overview](#-overview)
- [🧠 Key Idea](#-key-idea)
- [🏗️ System Architecture](#️-system-architecture)
- [📍 MediaPipe Feature Extraction](#-mediapipe-feature-extraction)
- [📁 Repository Structure](#-repository-structure)
- [⚙️ Installation](#️-installation)
- [🗂️ Dataset](#️-dataset)
- [🚀 Usage](#-usage)
- [🏋️ Training](#️-training)
- [🎥 Real-Time Inference](#-real-time-inference)
- [📊 Results](#-results)
- [📦 Pretrained Model](#-pretrained-model)
- [📚 Citation](#-citation)
- [📬 Contact](#-contact)

---

## 📌 Overview

This repository implements a real-time **emotion-aware intelligence system** for personalized healthcare.

The system uses a webcam and the **MediaPipe Holistic** pipeline to detect and extract landmark keypoints from the face and body. These landmark features are then classified using machine learning algorithms to recognize emotional states such as happy, sad, aggressive, confused, disgust, surprise, and willingness/openness.

The original paper proposes an end-to-end framework that connects real-time emotion recognition with a **digital twin-based personalized healthcare setup**, where the detected emotional state can support clinical monitoring, early diagnosis, and treatment decision support.

---

## 🧠 Key Idea

Emotion recognition in healthcare is useful because a patient's emotional state can provide additional context for:

- stress and depression monitoring
- emergency-room patient assessment
- personalized treatment planning
- doctor-patient interaction support
- digital twin-based healthcare simulation

The pipeline is intentionally lightweight and cost-effective:

1. A webcam captures RGB frames.
2. MediaPipe detects facial/body landmarks.
3. Landmark coordinates are stored in CSV format.
4. Multiple ML classifiers are trained.
5. Gradient Boosting is used for real-time emotion prediction.

---

## 🏗️ System Architecture

The full system contains three major stages:

### 1. Data Acquisition

Images are collected using a webcam at different lighting conditions, backgrounds, and body/face postures.

### 2. Detection and Feature Extraction

MediaPipe Holistic extracts landmark features from:

- face
- body pose
- hands, when available

The released dataset/model in this repository uses a model-compatible pose+face representation:

```text
33 pose landmarks + 468 face landmarks = 501 landmarks
501 landmarks × 4 values = 2004 features
```

Each landmark is represented as:

```text
x, y, z, visibility
```

### 3. Emotion Recognition and Classification

Several machine-learning algorithms are trained and compared:

- Logistic Regression
- Random Forest
- Gradient Boosting
- K-Nearest Neighbor
- Support Vector Machine
- Decision Tree
- Ridge Classifier
- Naive Bayes

The paper reports Gradient Boosting as the best-performing classifier for the real-time deployment.

---

## 📍 MediaPipe Feature Extraction

MediaPipe Holistic is used to extract landmark coordinates from each frame.

The implementation in this repository extracts:

```text
pose landmarks: 33 × 4
face landmarks: 468 × 4
total features: 2004
```

Feature CSV format:

```text
class,x1,y1,z1,v1,x2,y2,z2,v2,...,x501,y501,z501,v501
```

The `class` column stores the emotion label.

---

## 📁 Repository Structure

```text
Emotion-aware-intelligence-system/
├── assets/
│   ├── gb_confusion_matrix.png
│   ├── gb_learning_curve.png
│   └── gb_precision_recall.png
│
├── data/
│   ├── raw/
│   └── processed/
│       └── coords.zip
│
├── models/
│   └── gradient_boosting_emotion_model.pkl
│
├── notebooks/
│   └── Emotion_detection_original.ipynb
│
├── src/
│   ├── mediapipe_utils.py
│   ├── collect_landmarks.py
│   ├── train.py
│   ├── evaluate.py
│   ├── plot_learning_curve.py
│   └── realtime_inference.py
│
├── results/
│   ├── confusion_matrix/
│   ├── learning_curve/
│   ├── precision_recall/
│   ├── roc_curve/
│   └── micro_avg_score/
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/barathi-1993/Emotion-aware-intelligence-system.git
cd Emotion-aware-intelligence-system
```

Create a virtual environment:

```bash
conda create -n emotion_dt python=3.8 -y
conda activate emotion_dt
```

Install dependencies:

```bash
pip install -r requirements.txt
```

> The pretrained `body_language.pkl` model was created with an older scikit-learn version. For best compatibility, this repository pins `scikit-learn==1.0.2`.

---

## 🗂️ Dataset

The paper reports a custom real-time emotion dataset containing **5,991 labelled images** collected from three volunteers under different lighting conditions, backgrounds, nationalities, and genders.

The dataset contains 10 emotion classes:

| Class | Emotion | Samples |
|---:|---|---:|
| 0 | Happy | 991 |
| 1 | Sad | 553 |
| 2 | Aggressive/Angry | 421 |
| 3 | Focused/Pay Attention | 556 |
| 4 | Bored | 646 |
| 5 | Shock/Surprise | 392 |
| 6 | Anxiety/Unfriendliness | 655 |
| 7 | Openness/Willingness | 515 |
| 8 | Confused | 512 |
| 9 | Disgust | 412 |

Data split:

```text
Training: 70%
Testing : 30%
```

---

## 🚀 Usage

### 1. Collect New Emotion Landmark Data

```bash
python src/collect_landmarks.py \
  --class-name Happy \
  --output data/processed/coords.csv \
  --camera 0
```

Press `q` to stop recording.

---

### 2. Train Emotion Recognition Models

Train all candidate classifiers and save the best model:

```bash
python src/train.py \
  --data data/processed/coords.zip \
  --model-out models/gradient_boosting_emotion_model.pkl \
  --metrics-out results/metrics.json
```

---

### 3. Evaluate the Trained Model

```bash
python src/evaluate.py \
  --data data/processed/coords.zip \
  --model models/gradient_boosting_emotion_model.pkl \
  --output-dir results/evaluation
```

---

### 4. Plot Learning Curve

```bash
python src/plot_learning_curve.py \
  --data data/processed/coords.zip \
  --output results/evaluation/gb_learning_curve.png
```

---

## 🎥 Real-Time Inference

Run webcam-based real-time emotion recognition:

```bash
python src/realtime_inference.py \
  --model models/gradient_boosting_emotion_model.pkl \
  --camera 0
```

During inference:

1. Webcam frames are captured.
2. MediaPipe extracts pose and face landmarks.
3. A 2004-dimensional feature vector is generated.
4. The trained classifier predicts the emotion.
5. The predicted emotion and confidence are displayed on the screen.

---

## 📊 Results

The paper reports that the proposed ER system achieved high real-time recognition performance using a lightweight webcam-based setup.

### Average Classification Accuracy

| Algorithm | Accuracy |
|---|---:|
| Logistic Regression | 99.1% |
| Random Forest | 99.6% |
| Gradient Boosting | **99.9%** |
| KNN | 99.7% |
| SVM | 98.1% |
| Decision Tree | 98.6% |
| Naive Bayes | 76.5% |
| Ridge Regression | 98.1% |

### Gradient Boosting Classification Report

| Emotion | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Aggressive | 1.00 | 1.00 | 1.00 |
| Anxiety | 1.00 | 1.00 | 1.00 |
| Bored | 1.00 | 1.00 | 1.00 |
| Confused | 1.00 | 1.00 | 1.00 |
| Disgust | 1.00 | 1.00 | 1.00 |
| Focused | 1.00 | 1.00 | 1.00 |
| Happy | 1.00 | 0.99 | 1.00 |
| Sad | 0.99 | 1.00 | 1.00 |
| Willingness | 1.00 | 0.99 | 1.00 |
| Surprise | 1.00 | 1.00 | 1.00 |

The paper reports approximately **15 FPS** real-time performance using a single webcam and an average real-time recognition accuracy of approximately **99%**.

---

## 📦 Pretrained Model

The pretrained model is included as:

```text
models/gradient_boosting_emotion_model.pkl
```

If the pickle file fails to load because of scikit-learn version mismatch, retrain using:

```bash
python src/train.py --data data/processed/coords.zip
```

---

## 📚 Citation

If you use this repository, please cite:

```bibtex
@article{subramanian2022digitaltwin,
  title   = {Digital Twin Model: A Real-Time Emotion Recognition System for Personalized Healthcare},
  author  = {Subramanian, Barathi and Kim, Jeonghong and Maray, Mohammed and Paul, Anand},
  journal = {IEEE Access},
  volume  = {10},
  pages   = {81155--81165},
  year    = {2022},
  doi     = {10.1109/ACCESS.2022.3193941}
}
```

---

## ⚠️ Notes

This repository is intended for research and educational use.

Real-time emotion recognition performance may vary depending on:

- lighting conditions
- camera quality
- face visibility
- posture visibility
- occlusion
- background clutter
- domain shift across users

This system should not be used as a standalone medical diagnostic system. It is a decision-support research prototype.

---

## 📬 Contact

For questions, please open an issue in this repository.

---

<p align="center">
  ⭐ If this project is useful, please consider starring the repository. ⭐
</p>
