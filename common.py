"""Shared constants and landmark helpers for the ISL gesture recognizer.

A "sample" here is one gesture performed once: a sequence of frames, and
each frame is the 21 MediaPipe hand landmarks (x, y, z).

Keeping the vocabulary and the maths in one place means collect_data.py,
train.py and demo.py can never disagree about sequence length or labels.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- vocabulary
# Eight everyday communicative intents, chosen because they are useful in a
# first conversation and because each one has a visibly different hand
# motion. These are DEMO labels, not a claim to cover Indian Sign Language
# grammar: real ISL uses two hands, facial expression and context as well.
# See README.md ("Honest limits") before presenting this as a translator.
GESTURES = [
    "hello",
    "thank_you",
    "yes",
    "no",
    "please",
    "help",
    "sorry",
    "bye",
]

# ------------------------------------------------------------- sequence shape
SEQUENCE_LENGTH = 30  # frames captured for one gesture (about 1 second at 30 fps)
NUM_LANDMARKS = 21    # MediaPipe Hands returns exactly 21 points per hand
COORDS_PER_LANDMARK = 3  # x, y, z
FEATURES_PER_FRAME = NUM_LANDMARKS * COORDS_PER_LANDMARK  # 63
FEATURES_PER_SAMPLE = SEQUENCE_LENGTH * FEATURES_PER_FRAME  # 1,890


def normalize_frame(landmarks: np.ndarray) -> np.ndarray:
    """Normalise one frame of landmarks so position/size do not dominate.

    What it does: subtracts the wrist (landmark 0) so the hand's position
    in the camera frame no longer matters, then divides by the distance
    from wrist to middle-finger knuckle (landmark 9) so a hand close to
    the camera and the same hand further away look similar.
    Why: without this, the model would learn "hand was on the left" instead
    of "hand moved like this".
    What it returns: a (21, 3) array of floats, wrist at the origin.
    """
    landmarks = np.asarray(landmarks, dtype=float)
    if landmarks.shape != (NUM_LANDMARKS, COORDS_PER_LANDMARK):
        raise ValueError(f"Expected shape (21, 3), got {landmarks.shape}")
    centred = landmarks - landmarks[0]
    hand_size = np.linalg.norm(centred[9])  # wrist -> middle knuckle
    if hand_size < 1e-6:  # degenerate frame (all points on top of each other)
        return centred
    return centred / hand_size


def sequence_to_features(sequence: np.ndarray) -> np.ndarray:
    """Flatten a (30, 21, 3) landmark sequence into one 1,890-number vector.

    The classifier in train.py works on fixed-size vectors, so a whole
    gesture -- its shape at every moment, in order -- becomes one row.
    Order is preserved by flattening frame by frame, so motion over time
    is still in the numbers, just not modelled as a recurrent network.
    Returns: a (1890,) float array.
    """
    sequence = np.asarray(sequence, dtype=float)
    if sequence.shape != (SEQUENCE_LENGTH, NUM_LANDMARKS, COORDS_PER_LANDMARK):
        raise ValueError(
            f"Expected shape (30, 21, 3), got {sequence.shape}"
        )
    normalised = np.stack([normalize_frame(frame) for frame in sequence])
    return normalised.reshape(-1)


def features_to_sequence(features: np.ndarray) -> np.ndarray:
    """Inverse of sequence_to_features, for inspection/debugging only."""
    return np.asarray(features, dtype=float).reshape(
        SEQUENCE_LENGTH, NUM_LANDMARKS, COORDS_PER_LANDMARK
    )
