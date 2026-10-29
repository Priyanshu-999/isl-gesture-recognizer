"""Live webcam demo: recognise gestures in real time.

Run on a Windows PC with a webcam, after training:
    python train.py
    python demo.py

How it works: MediaPipe tracks one hand each frame and appends its 21
landmarks to a rolling window of the last 30 frames. Once the window is
full, the trained model predicts a gesture, and the prediction is
smoothed -- the label shown is the most common prediction of the last
5 windows -- so a single flickering frame does not change the answer.
The predicted word, its confidence, and a small "hand detected" state
are drawn on the video. Press Q to quit.

What this demo cannot do (read before demoing to anyone): it only
knows the 8 gestures it was trained on, only one hand, only one signer
if trained on one signer, and it needs good lighting and the full hand
in frame. A word it has never seen will still be forced into one of
the 8 labels -- that is a classifier limitation, not understanding.
If you train on synthetic data only, this demo will NOT recognise real
hands well; record real samples with collect_data.py first.
"""

from __future__ import annotations

import argparse
import collections
import pathlib

import numpy as np

from common import NUM_LANDMARKS, SEQUENCE_LENGTH

MODEL_PATH = pathlib.Path("models") / "gesture_model.joblib"
SMOOTHING_WINDOW = 5  # show the majority vote of the last 5 predictions


def smooth_predictions(history: collections.deque) -> str | None:
    """Return the most common label in recent predictions, or None.

    Majority voting over a short history removes single-frame flicker
    without adding noticeable lag: at 30 fps, 5 windows is a fraction
    of a second of extra stability.
    """
    if not history:
        return None
    counts = collections.Counter(history)
    return counts.most_common(1)[0][0]


def main() -> None:
    """CLI: open the webcam and label gestures live (see module docstring)."""
    import cv2
    import joblib
    import mediapipe as mp

    from common import sequence_to_features

    parser = argparse.ArgumentParser(description="Live gesture recognition demo.")
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    if not pathlib.Path(args.model).exists():
        raise SystemExit(
            f"Model not found: {args.model}\nRun 'python train.py' first."
        )
    bundle = joblib.load(args.model)
    model, labels = bundle["model"], bundle["labels"]

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Check it is connected and not in use.")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)
    window: list[np.ndarray] = []
    history: collections.deque = collections.deque(maxlen=SMOOTHING_WINDOW)
    current_label, current_confidence = "Show your hand...", 0.0
    print("Demo running. Press Q in the video window to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS)
                landmarks = np.array([[p.x, p.y, p.z] for p in hand.landmark], dtype=float)
                window.append(landmarks)
                window = window[-SEQUENCE_LENGTH:]
                if len(window) == SEQUENCE_LENGTH:
                    features = sequence_to_features(np.stack(window)).reshape(1, -1)
                    prediction = model.predict(features)[0]
                    probabilities = model.predict_proba(features)[0]
                    confidence = float(np.max(probabilities))
                    history.append(prediction)
                    smoothed = smooth_predictions(history)
                    if smoothed is not None:
                        current_label, current_confidence = smoothed, confidence
            else:
                window.clear()
                history.clear()
                current_label, current_confidence = "No hand detected", 0.0

            cv2.rectangle(frame, (10, 10), (460, 90), (0, 0, 0), -1)
            cv2.putText(frame, f"Gesture: {current_label}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {current_confidence:.2f}  (Q to quit)",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("ISL Gesture Recognizer - press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
