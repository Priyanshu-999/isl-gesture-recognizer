"""Record your own gesture samples from a webcam.

Run this on a Windows PC with a webcam. It needs the packages in
requirements.txt (opencv-python, mediapipe). It is NOT needed to run
train.py on the synthetic sample data.

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

# Imports for camera/hand-tracking are inside functions so that importing
# this module (for example to check its help) does not require a webcam
# or the heavy packages to be installed.


def extract_landmarks(results) -> np.ndarray | None:
    """Pull the first detected hand's landmarks from a MediaPipe result.

    Returns a (21, 3) float array, or None when no hand is visible.
    Only the first hand is used: this is a single-hand demo.
    """
    if not results.multi_hand_landmarks:
        return None
    hand = results.multi_hand_landmarks[0]
    return np.array([[p.x, p.y, p.z] for p in hand.landmark], dtype=float)


def record_sample(cap, hands, countdown: int = 3) -> np.ndarray | None:
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
    while len(frames) < SEQUENCE_LENGTH:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = extract_landmarks(hands.process(rgb))
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
    import mediapipe as mp

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
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)
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
            sample = record_sample(cap, hands)
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
