"""Generate a CLEARLY LABELLED SYNTHETIC landmark dataset.

Why this file exists: this repo was built on a machine with no webcam and
no signer, so no real gestures could be recorded here. Rather than fake
real data, the pipeline is built and tested on synthetic sequences whose
shapes and motions are deliberately different per gesture. The numbers
reported from this data measure "does the pipeline work", NOT "how well
does it recognise a real person's ISL".

How each synthetic gesture is made: a base hand pose (which fingers are
extended, by different amounts per gesture) plus a motion pattern over
the 30 frames (for example hello waves side-to-side, yes bobs up-down,
no shakes side-to-side, thank_you moves forward, please draws a small
circle, help rises, sorry moves in a small loop, bye waves faster and
smaller). Random noise, scale and shift are added per sample so samples
are not identical copies.

Usage:
    python generate_sample_data.py --samples-per-gesture 60 --seed 42

Output: data/sample/sample_landmarks.npz with arrays X (n, 30, 21, 3)
and y (n,) of gesture names, plus a small README note in data/sample/.
Replace this file's output with real recordings from collect_data.py
and retrain -- that is the intended next step, see README.md.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

from common import COORDS_PER_LANDMARK, GESTURES, NUM_LANDMARKS, SEQUENCE_LENGTH

# Finger "openness" per gesture, one value per finger group in order:
# thumb, index, middle, ring, pinky. 1.0 = fully extended, 0.2 = curled.
# These are invented demo poses used ONLY to make synthetic classes
# separable; they are not certified ISL handshapes.
_POSES = {
    "hello": [0.9, 1.0, 1.0, 1.0, 1.0],      # open hand
    "thank_you": [0.8, 0.9, 0.9, 0.7, 0.7],  # open hand, fingers together
    "yes": [0.3, 0.4, 0.4, 0.3, 0.3],        # closed fist
    "no": [0.9, 1.0, 0.3, 0.3, 0.3],         # index pointing
    "please": [0.9, 1.0, 1.0, 1.0, 0.4],     # open hand, pinky curled
    "help": [0.7, 0.8, 0.8, 0.8, 0.8],       # both-hands idea -> one open hand
    "sorry": [0.3, 0.5, 0.5, 0.4, 0.4],      # loose fist
    "bye": [0.9, 1.0, 1.0, 0.9, 0.9],        # open hand (faster wave)
}

# Which landmarks belong to which finger (MediaPipe ordering).
_FINGERS = {
    0: [1, 2, 3, 4],        # thumb
    1: [5, 6, 7, 8],        # index
    2: [9, 10, 11, 12],     # middle
    3: [13, 14, 15, 16],    # ring
    4: [17, 18, 19, 20],    # pinky
}


def _base_hand(pose: list[float]) -> np.ndarray:
    """Build one (21, 3) hand from finger-openness values.

    Landmarks are laid out in a simple flat-hand skeleton: knuckles in a
    row, each finger extending outward by its openness amount. Good
    enough to give each class a distinct shape; it is not anatomy.
    """
    hand = np.zeros((NUM_LANDMARKS, COORDS_PER_LANDMARK))
    hand[0] = [0.0, 0.0, 0.0]  # wrist
    # Knuckle positions across the palm.
    knuckles = {5: (-0.06, -0.18), 9: (-0.02, -0.20), 13: (0.02, -0.19), 17: (0.06, -0.16)}
    for landmark, (x, y) in knuckles.items():
        hand[landmark] = [x, y, 0.0]
    for finger_idx, landmarks in _FINGERS.items():
        openness = pose[finger_idx]
        start = hand[landmarks[0]]
        direction = np.array([start[0] * 0.4, -1.0, 0.0])
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        for step, landmark in enumerate(landmarks, start=1):
            hand[landmark] = start + direction * 0.05 * step * openness
    hand[1] = [-0.08, -0.05, 0.0]  # thumb base, kept near the palm
    return hand


def _motion(gesture: str, frame: int) -> np.ndarray:
    """Per-frame (x, y, z) shift for a gesture, frame in 0..29."""
    t = frame / SEQUENCE_LENGTH
    wave = np.sin(2 * np.pi * t)
    if gesture == "hello":
        return np.array([0.08 * wave, 0.0, 0.0])
    if gesture == "bye":
        return np.array([0.05 * np.sin(4 * np.pi * t), 0.0, 0.0])
    if gesture == "yes":
        return np.array([0.0, 0.04 * np.abs(wave), 0.0])
    if gesture == "no":
        return np.array([0.06 * np.sign(wave) * (0.5 + 0.5 * abs(wave)), 0.0, 0.0])
    if gesture == "thank_you":
        return np.array([0.0, 0.0, 0.05 * t])
    if gesture == "please":
        return np.array([0.04 * np.sin(2 * np.pi * t), 0.04 * np.cos(2 * np.pi * t), 0.0])
    if gesture == "help":
        return np.array([0.0, 0.06 * t, 0.0])
    if gesture == "sorry":  # small loop, opposite phase to please
        return np.array([0.03 * np.cos(2 * np.pi * t), 0.03 * np.sin(2 * np.pi * t), 0.0])
    return np.zeros(3)


def generate_sample_data(samples_per_gesture: int = 60, seed: int = 42):
    """Generate synthetic sequences. Returns (X, y): X (n, 30, 21, 3), y (n,)."""
    rng = np.random.default_rng(seed)
    sequences, labels = [], []
    for gesture in GESTURES:
        base = _base_hand(_POSES[gesture])
        for _ in range(samples_per_gesture):
            scale = rng.uniform(0.9, 1.1)
            shift = rng.uniform(-0.05, 0.05, size=3)
            sample = np.stack(
                [base * scale + _motion(gesture, f) + shift for f in range(SEQUENCE_LENGTH)]
            )
            sample += rng.normal(0, 0.008, size=sample.shape)  # sensor-like noise
            sequences.append(sample)
            labels.append(gesture)
    return np.stack(sequences), np.array(labels)


def main() -> None:
    """CLI: write data/sample/sample_landmarks.npz (see module docstring)."""
    parser = argparse.ArgumentParser(description="Generate synthetic sample landmarks.")
    parser.add_argument("--samples-per-gesture", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/sample/sample_landmarks.npz")
    args = parser.parse_args()

    X, y = generate_sample_data(args.samples_per_gesture, args.seed)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y)
    (out.parent / "README.txt").write_text(
        "SYNTHETIC SAMPLE DATA -- NOT real recorded gestures.\n"
        "Generated by generate_sample_data.py for pipeline testing only.\n"
        "Accuracy measured on this data is NOT real-gesture accuracy.\n"
        "Record real samples with collect_data.py, retrain with train.py,\n"
        "and report accuracy on held-out REAL samples instead.\n"
    )
    print(f"Wrote {out}  shape={X.shape}  gestures={sorted(set(y))}")


if __name__ == "__main__":
    main()
