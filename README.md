# Indian Sign Language Gesture Recognizer — Demo

A working demo that recognises a small set of Indian Sign Language (ISL)
gestures from a webcam. It is **demo-grade, not a production translator**:
it knows 8 gestures, from one hand, and it needs good lighting and a
clear view of the hand.

## The problem

Sign language is a full, visual language, but most everyday software
only understands typing and speech. This project is a first step toward
closing that gap: point a webcam at a hand, and the computer names the
gesture being signed, in real time.

## How it works

1. **See the hand** — MediaPipe Hands finds 21 landmarks on the hand
   in each video frame (fingertips, knuckles, wrist), as x, y, z numbers.
2. **Clean the numbers** — each frame is centred on the wrist and scaled
   by hand size, so where the hand sits in the frame and how close it is
   to the camera matter far less than the shape and motion itself.
3. **Collect a gesture** — one gesture = 30 frames in a row (about 1
   second). Flattened, that is 30 × 21 × 3 = **1,890 numbers**.
4. **Classify** — a RandomForest classifier (StandardScaler + 300 trees)
   learns which 1,890-number pattern belongs to which gesture. No GPU
   is needed; training on a few hundred samples takes seconds.
5. **Show it live** — the demo keeps a rolling 30-frame window, predicts
   every frame, and shows the majority vote of the last 5 predictions so
   the label does not flicker.

Why not an LSTM? A recurrent network models time more directly, but it
needs TensorFlow, longer training and more data than a first demo has.
The flattened classifier is the honest, runnable version of "a simple
classifier over landmark sequences", and `train.py:build_model()` is the
one place to swap in an LSTM later without touching data loading or
the demo.

## Vocabulary (8 gestures)

| Gesture | Demo motion idea |
|---|---|
| `hello` | Open hand, wave side to side |
| `thank_you` | Open hand moving forward |
| `yes` | Fist bobbing down (nodding motion) |
| `no` | Pointing hand shaking side to side |
| `please` | Open hand in a small circle |
| `help` | Open hand rising |
| `sorry` | Loose fist in a small loop |
| `bye` | Open hand waving faster and smaller |

These are everyday communicative intents with visibly different motions,
chosen for a first demo. They are **not a certified ISL dictionary**:
real ISL also uses two hands, facial expression, body movement and
grammar. If you present this, say "demo vocabulary" and check the
handshapes against a proper ISL resource before claiming more.

## Quick start (Windows PC with a webcam)

```bash
pip install -r requirements.txt
```

### Option A — test the pipeline right now (synthetic data, no webcam)

```bash
python generate_sample_data.py   # creates data/sample/sample_landmarks.npz
python train.py                   # trains + evaluates + saves models/gesture_model.joblib
```

### Option B — record real gestures, then train (the real demo)

```bash
# Record at least 20 samples of each gesture (do all 8):
python collect_data.py --gesture hello --samples 20
python collect_data.py --gesture thank_you --samples 20
# ... repeat for yes, no, please, help, sorry, bye

python train.py    # automatically prefers your real data over synthetic
python demo.py     # webcam window: perform a gesture, read the label, Q to quit
```

In `collect_data.py`: press **S** to start a sample (after a 3-2-1
countdown, perform the gesture once, steadily, for about a second),
**Q** to quit. Keep your whole hand in frame, in even light, and keep
the gesture the same every time — consistency beats quantity.

## Measured results — read the caveat first

| | |
|---|---|
| Dataset | **SYNTHETIC sample landmarks** (480 samples: 60 per gesture, seed 42) |
| Split | Stratified 80/20 — 384 train, 96 held-out test |
| Model | StandardScaler + RandomForest, 300 trees, seed 42 |
| Held-out accuracy | **1.0000 (96/96)** |
| Per-gesture F1 | 1.00 for all 8 gestures |

**This 100% is NOT real-gesture accuracy, and it must not be quoted as
if it were.** The synthetic classes were deliberately built with
different hand poses and motions, so a perfect score mostly proves the
pipeline runs end to end (loading → features → training → evaluation
→ saved model → demo prediction path). Real hands vary far more: hand
shape, speed, lighting, camera angle and signer all change the numbers.
The honest number to report is the held-out accuracy **after retraining
on real recorded samples** — replace the table above with that number,
and keep the synthetic number labelled as synthetic.

Charts: `results/confusion_matrix.png`, `results/per_class_f1.png`
Numbers: `results/metrics.json`

## Honest limits

- **Small vocabulary** — 8 gestures only. Anything else is forced into
  one of these 8 labels; the model never says "I don't know".
- **Single signer, single hand** — trained on one person's samples, it
  will be weaker on other people. Record more signers to fix this.
- **Good lighting needed** — MediaPipe loses the hand in dim light,
  cluttered backgrounds or fast motion blur.
- **No face, no grammar** — real ISL meaning also lives in facial
  expression, two-handed signs and sentence context. This demo ignores
  all three.
- **Webcam not verified in the build environment** — the code was built
  and tested on a machine with no webcam, so `collect_data.py` and
  `demo.py` still need a live webcam test on your PC (see below).
- **Synthetic data is not a substitute for real data** — it only proves
  the pipeline works.

## What was verified here vs what needs your webcam

Verified in the build environment (29 Oct 2026):

- All five Python files import cleanly and compile.
- Synthetic data generation: 480 samples, shape (480, 30, 21, 3).
- Training end to end: 1.0000 held-out accuracy on synthetic data
  (see caveat above), confusion matrix and F1 charts generated.
- Saved model loads and predicts correctly through the same feature
  path `demo.py` uses (hello → hello, yes → yes, no → no on samples).

Still needs your Windows PC + webcam:

- `collect_data.py` recording real samples (all 8 gestures).
- Retraining on real data and reporting real held-out accuracy.
- `demo.py` live test: does it label your real gestures, in your
  lighting, without flicker?

## Project structure

```
common.py                 Vocabulary, sequence shape, landmark normalisation
generate_sample_data.py   Clearly-labelled SYNTHETIC data (pipeline testing only)
collect_data.py           Webcam recording: one .npy sample per gesture performance
train.py                  Train, held-out evaluation, charts, saved model
demo.py                   Live webcam demo with prediction smoothing
data/sample/              Synthetic landmarks + README.txt warning label
results/                  metrics.json, confusion_matrix.png, per_class_f1.png
docs/UNDERSTANDING.pdf    Simple-language explainer + interview prep
requirements.txt          Core + webcam packages (no GPU needed)
```

## Skills shown

Python · OpenCV · MediaPipe hand landmarks · feature engineering
(normalisation, sequence flattening) · scikit-learn classification ·
train/test evaluation, confusion matrix, per-class metrics · honest
reporting of synthetic vs real results · Git/GitHub workflow.

## Push to GitHub (run these yourself — no credentials are stored here)

```bash
cd isl-gesture-recognizer
git remote add origin https://github.com/Priyanshu-999/isl-gesture-recognizer.git
git branch -M main
git push -u origin main
```

If the repository does not exist yet on GitHub, create an empty repo
named `isl-gesture-recognizer` first (no README, no .gitignore — this
project already has them), then run the commands above.
