"""Record your own gesture samples from a webcam.

Run this on a Windows PC with a webcam. It needs the packages in
requirements.txt (opencv-python, mediapipe). It is NOT needed to run
train.py on the synthetic sample data.

This version uses MediaPipe's Tasks API (HandLandmarker). The old
``mp.solutions`` API was removed from newer MediaPipe releases, so
this file no longer imports it. Before running, download the hand
model file into this folder (once):

    curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

How a recording session works:
  1. Sit in good, even light, with your hand clearly in frame.
  2. Run:  python collect_data.py --gesture hello --samples 20
  3. Press S to start a sample. A 3-2-1 countdown shows on screen.
  4. Perform the gesture once, steadily, for about one second.
     The script captures SEQUENCE_LENGTH (30) frames of landmarks.
  5. Repeat until you have the requested number of samples.
  6. Press Q at any time to stop early; samples already saved are kept.

What it saves: one .npy file per sample, shaped (30, 21, 3), under
data/<gesture>/, named <gesture>_<YYYYMMDD_HHMMSS>_<index>.npy.
Recording one signer only is a known limit (see README.md): for a
stronger demo, record at least 20 samples per gesture, on two or three
different days / in different lighting, and ask a friend to add samples
too -- then retrain and compare the held-out accuracy.

Tips that actually matter: keep the same gesture speed each time, keep
your whole hand in frame, and do not change what the gesture means
half-way through collecting (consistency beats quantity).
"""

from __future__ import annotations

import argparse
import datetime
import pathlib
import time

import numpy as np

from common import GESTURES, NUM_LANDMARKS, SEQUENCE_LENGTH

MODEL_ASSET = pathlib.Path(__file__).parent / "hand_landmarker.task"

# Imports for camera/hand-tracking are inside functions so that importing
# this module (for example to check its help) does not require a webcam
# or the heavy packages to be installed.


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


def record_sample(cap, landmarker, countdown: int = 3) -> np.ndarray | None:
    """Record one gesture: countdown, then 30 frames of landmarks.

    Returns a (30, 21, 3) array, or None if the hand was lost for too
    many frames (fewer than 25 usable frames) or the user pressed Q.
    Frames with no hand are filled with the last good frame so the
    sequence length stays fixed -- a brief dropout should not ruin a
    sample, but a mostly-missing hand should.
    """
    import cv2  # local import: only needed when actually recording

    for number in range(countdown, 0, -1):
        ok, frame = cap.read()
        if not ok:
            return None
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Get ready... {number}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imshow("Collect gestures - press Q to quit", frame)
        if cv2.waitKey(1000) & 0xFF == ord("q"):
            return None

    frames: list[np.ndarray] = []
    last_good: np.ndarray | None = None
    last_timestamp_ms = 0
    while len(frames) < SEQUENCE_LENGTH:
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
            last_good = landmarks
            frames.append(landmarks)
        elif last_good is not None:
            frames.append(last_good)  # brief dropout: repeat last frame
        cv2.putText(frame, f"Recording... {len(frames)}/{SEQUENCE_LENGTH}",
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("Collect gestures - press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return None

    if len(frames) < 25:  # too much of the gesture is missing
        return None
    # Pad short sequences (26-29 frames) by repeating the last frame.
    while len(frames) < SEQUENCE_LENGTH:
        frames.append(frames[-1])
    return np.stack(frames[:SEQUENCE_LENGTH])


def main() -> None:
    """CLI: record samples for one gesture (see module docstring)."""
    import cv2

    parser = argparse.ArgumentParser(description="Record gesture samples from webcam.")
    parser.add_argument("--gesture", required=True, choices=GESTURES)
    parser.add_argument("--samples", type=int, default=20,
                        help="How many samples to record (default 20).")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default 0).")
    args = parser.parse_args()

    out_dir = pathlib.Path("data") / args.gesture
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("*.npy")))
    print(f"Recording '{args.gesture}'. Already have {existing} samples in {out_dir}.")
    print("Press S to start each sample, Q to quit. Keep your hand in frame.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Check it is connected and not in use.")
    landmarker = create_landmarker()
    try:
        recorded = 0
        while recorded < args.samples:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"{args.gesture}: {recorded}/{args.samples}  (S=start, Q=quit)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Collect gestures - press Q to quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key != ord("s"):
                continue
            sample = record_sample(cap, landmarker)
            if sample is None or sample.shape != (SEQUENCE_LENGTH, NUM_LANDMARKS, 3):
                print("Sample skipped (hand lost). Try again.")
                continue
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = out_dir / f"{args.gesture}_{stamp}_{existing + recorded:03d}.npy"
            np.save(path, sample)
            recorded += 1
            print(f"Saved {path} ({recorded}/{args.samples})")
            time.sleep(0.5)
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"Done. {recorded} new samples for '{args.gesture}'.")


if __name__ == "__main__":
    main()
