# ECG Arrhythmia Classifier

A machine learning system that detects ventricular arrhythmias from ECG signals with **99% F1-score on unseen patients**, built on real clinical data from the MIT-BIH Arrhythmia Database.

---

## The Problem

Ventricular arrhythmias are abnormal heart rhythms that can lead to sudden cardiac death if undetected. Manual ECG interpretation requires trained cardiologists and is impractical for continuous monitoring. This project builds an automated classifier that can distinguish normal beats from ventricular ectopic beats in real time.

---

## What Makes This Different

Most ECG classifiers treat the signal as a raw array and feed it directly into a neural network. This project takes a signal-processing-first approach:

- Raw ECG is bandpass filtered to remove baseline wander and muscle noise before any ML is applied
- R-peaks are detected using the Pan-Tompkins-inspired algorithm via NeuroKit2
- Individual beats are segmented into 200-sample windows centered on each R-peak
- **18 hand-crafted features** are extracted — combining time-domain statistics, segmental energy analysis (P-wave, QRS, T-wave zones), and FFT-based frequency features

This approach comes from an Electronics and Telecommunication Engineering background, where signal processing is applied before classification — not instead of it.

---

## Dataset

**MIT-BIH Arrhythmia Database** (PhysioNet) — 48 half-hour recordings of ambulatory ECGs collected at Beth Israel Hospital, Boston. Sampled at 360 Hz with expert cardiologist annotations.

Records used: `100`, `106`, `119`, `200` (training) | `208` (held-out test patient)

| Class | Count |
|-------|-------|
| Normal (N) | 8,608 |
| Ventricular (V) | 2,767 |
| **Total** | **11,375** |

---

## Results

Evaluated on **record 208 — a completely unseen patient not used in training**.

| Metric | Normal | Ventricular |
|--------|--------|-------------|
| Precision | 1.00 | 0.97 |
| Recall | 0.98 | 1.00 |
| F1-score | 0.99 | 0.99 |
| **Overall accuracy** | | **99%** |

**Key clinical result: zero missed ventricular beats (0 false negatives).** In cardiac monitoring, failing to detect a real arrhythmia is more dangerous than a false alarm. The model is tuned toward this priority.

---

## Top Features by Importance

The Random Forest feature importance reveals which signal characteristics drive classification:

1. `energy_twave` — ventricular beats have a distinctly large T-wave energy
2. `skewness` — ventricular beats are morphologically asymmetric
3. `min` — deep negative deflection characteristic of V beats
4. `qrs_width` — ventricular beats are wider than normal beats
5. `power_lf` — low-frequency spectral power differs between beat types

These features are clinically meaningful — they match what a cardiologist looks for when identifying ventricular ectopy on a paper ECG.

---

## Methodology

```
Raw ECG signal
      │
      ▼
Bandpass filter (0.5–40 Hz)     ← removes baseline wander + noise
      │
      ▼
R-peak detection                ← identifies each heartbeat location
      │
      ▼
Beat segmentation (200 samples) ← 90 samples before, 110 after R-peak
      │
      ▼
Feature extraction (18 features) ← time + frequency domain
      │
      ▼
Random Forest classifier
      │
      ▼
Normal / Ventricular prediction
```

---

## Project Structure

```
ECG-Medical-Project/
│
├── notebooks/
│   └── 01_explore.ipynb     # Full exploration, analysis, and modelling
│
├── src/
│   ├── preprocess.py        # Signal loading, filtering, segmentation
│   └── model.py             # Feature extraction, training, evaluation
│
├── model.joblib             # Trained Random Forest model
├── requirements.txt         # Dependencies
└── README.md
```

---

## Reproducing the Results

```bash
# Clone the repo
git clone https://github.com/AfeefSiddique/ECG-Medical-Project.git
cd ECG-Medical-Project

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run full training and evaluation pipeline
cd src
python model.py
```

Data is downloaded automatically from PhysioNet on first run — no manual download required.

---

## Requirements

```
wfdb
neurokit2
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
joblib
```

---

## Known Limitations and Future Work

- **Patient-specific variability:** The model is trained on 4 patients and tested on 1. Performance may vary on patients with unusual morphology or comorbidities.
- **Class imbalance:** The dataset contains more normal beats than arrhythmias. Future work could apply SMOTE or cost-sensitive learning.
- **Single lead:** Only Lead I is used. A multi-lead approach would improve robustness.
- **Next step — 1D CNN:** A convolutional neural network operating directly on the raw beat waveform could learn features automatically and potentially generalise better across patients.

---

## Author

**Afeef Siddique**   
[GitHub](https://github.com/AfeefSiddique)
