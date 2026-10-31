"""Live webcam demo: recognise gestures in real time.

Run on a Windows PC with a webcam, after training:
    python train.py
    python demo.py

This version uses MediaPipe's Tasks API (HandLandmarker). The old
``mp.solutions`` API was removed from newer MediaPipe releases, so
``demo.py`` no longer imports it. Before running, download the hand
model file into this folder (once):

    curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

How it works: the landmarker tracks one hand each frame and appends
its 21 landmarks to a rolling window of the last 30 frames. Once the
window is full, the trained model predicts a gesture, and the
prediction is smoothed -- the label shown is the most common
prediction of the last 5 windows -- so a single flickering frame does
not change the answer. The predicted word, its confidence, and a
small "hand detected" state are drawn on the video. Press Q to quit.

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
import time

import numpy as np

from common import NUM_LANDMARKS, SEQUENCE_LENGTH

MODEL_PATH = pathlib.Path("models") / "gesture_model.joblib"
MODEL_ASSET = pathlib.Path(__file__).parent / "hand_landmarker.task"
SMOOTHING_WINDOW = 5  # show the majority vote of the last 5 predictions

# MediaPipe hand topology (21 landmarks), used to draw the skeleton
# by hand since the old drawing_utils module went away with solutions.
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


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


def create_landmarker(model_path: pathlib.Path = MODEL_ASSET):
    """Build a HandLandmarker in VIDEO mode (one hand, 0.5 thresholds).

    Raises SystemExit with the download command when the model file is
    missing, instead of failing deep inside MediaPipe.
    """
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if not pathlib.Path(model_path).exists():
        raise SystemExit(
            f"Hand model not found: {model_path}\n"
            "Download it into this folder (once):\n"
            "curl -L -o hand_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect_hand_landmarks(landmarker, rgb_frame: np.ndarray, timestamp_ms: int):
    """Detect the first hand in an RGB frame.

    Returns a (21, 3) float array of normalised landmarks, or None
    when no hand is visible. ``timestamp_ms`` must keep increasing
    across calls (VIDEO mode requirement); the caller guarantees this.
    """
    import mediapipe as mp

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect_for_video(image, timestamp_ms)
    if not result.hand_landmarks:
        return None
    return np.array(
        [[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=float
    )


def draw_hand(frame: np.ndarray, landmarks: np.ndarray) -> None:
    """Draw the hand skeleton on a BGR frame, in place.

    Landmarks are normalised (0..1), so they are scaled to the frame
    size first. Green bones, red joints.
    """
    import cv2

    height, width = frame.shape[:2]
    points = [(int(p[0] * width), int(p[1] * height)) for p in landmarks]
    for start, end in _HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


def main() -> None:
    """CLI: open the webcam and label gestures live (see module docstring)."""
    import cv2
    import joblib

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
    landmarker = create_landmarker()
    window: list[np.ndarray] = []
    history: collections.deque = collections.deque(maxlen=SMOOTHING_WINDOW)
    current_label, current_confidence = "Show your hand...", 0.0
    last_timestamp_ms = 0
    print("Demo running. Press Q in the video window to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(time.monotonic() * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms
            landmarks = detect_hand_landmarks(landmarker, rgb, timestamp_ms)
            if landmarks is not None:
                draw_hand(frame, landmarks)
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
