# Music Structure Analysis — SpecTNT Replication & Extension

Replication and extension of:
> Wang, J.-C., Hung, Y.-N., & Smith, J. B. L. (2022).
> "To Catch a Chorus, Verse, Intro, or Anything Else:
> Analyzing a Song with Structural Functions."
> ICASSP 2022. DOI: 10.1109/ICASSP43922.2022.9747252

---

## What This Project Does

Automatically labels song sections from raw audio:

0:00 - 0:14  →  intro
0:14 - 0:45  →  verse
0:45 - 1:18  →  chorus
1:18 - 1:50  →  verse
1:50 - 2:23  →  chorus
2:23 - 2:45  →  outro

---

## My Contribution

The paper tested only harmonic representation.
This project compares 6 different spectrogram representations
under identical conditions — a controlled experiment the paper
did not perform.

| # | Feature | What it captures |
|---|---|---|
| 1 | Harmonic | Paper baseline — musical notes, overtones |
| 2 | Log-Mel | Perceptual loudness across mel bands |
| 3 | CQT | Musical pitch scale |
| 4 | Chroma STFT | Harmony and chords |
| 5 | MFCC | Timbral texture |
| 6 | Linear | Raw STFT — lower bound |

---

## Models Implemented

### Harmonic-CNN (paper's instant model)
- 7 two-dimensional convolutional layers
- 2 dense layers
- Two output heads: function + boundary
- Loss: 0.9 × boundary + 0.1 × function

### SpecTNT (paper's best model)
- ResNet front-end + Spectral-Temporal Transformer
- Spectral encoder: 4 attention heads
- Temporal encoder: 8 attention heads
- CTL loss for smooth ordered predictions
- Multi-point: one label per time frame

---

## Key Implementation Details

- Algorithm 1: exact label conversion from paper
- CTL loss: enforces sequential section ordering
- Augmentation: noise, gain, high/low-pass filters
- Feature caching: pre-computed .npy files
- 6 evaluation metrics via mir_eval
- Dataset: SALAMI-pop (174 popular songs)

---

## Results

| Feature | HR.5F | ACC | PWF | Sf | CHR.5F | CF1 |
|---|---|---|---|---|---|---|
| Paper (SpecTNT) | 0.490 | 0.544 | 0.651 | 0.632 | 0.357 | 0.811 |
| Harmonic | - | - | - | - | - | - |
| Log-Mel | 0.475 | 0.880 | 0.944 | 0.920 | 0.599 | 0.993 |


> Note: Results on training set. Paper uses cross-dataset evaluation.
> Focus is relative feature comparison under identical conditions.

---

## Project Structure