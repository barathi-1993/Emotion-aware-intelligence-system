# Repository Notes

This package reorganizes the original notebook-based Emotion-aware-intelligence-system repository into a cleaner research-code layout.

Original uploaded files preserved:
- `notebooks/Emotion_detection_original.ipynb`
- `data/processed/coords.zip`
- `models/gradient_boosting_emotion_model.pkl`
- `results/` figures

Important compatibility note:
- The released `coords.csv` contains 2004 features per row: 501 landmarks × 4 values.
- This corresponds to pose + face landmarks in the original notebook workflow.
- The IEEE Access manuscript describes MediaPipe face, hand, and body landmarks as the broader framework.
- The scripts in `src/` are aligned with the released trained model and dataset.

Recommended workflow:
1. `python src/train.py --data data/processed/coords.zip`
2. `python src/evaluate.py --data data/processed/coords.zip`
3. `python src/realtime_inference.py --model models/gradient_boosting_emotion_model.pkl`
