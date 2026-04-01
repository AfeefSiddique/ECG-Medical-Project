import numpy as np
import pandas as pd
import joblib
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


def extract_features(beat, fs=360):
    """
    Extract hand-crafted signal features from a single beat window.
    Combines time-domain statistics with frequency-domain analysis.
    """
    features = {}

    # Time domain
    features['mean']     = np.mean(beat)
    features['std']      = np.std(beat)
    features['max']      = np.max(beat)
    features['min']      = np.min(beat)
    features['range']    = np.max(beat) - np.min(beat)
    features['skewness'] = skew(beat)
    features['kurtosis'] = kurtosis(beat)
    features['rms']      = np.sqrt(np.mean(beat ** 2))

    # QRS width — samples where signal exceeds 50% of peak amplitude
    peak_val = np.max(np.abs(beat))
    features['qrs_width'] = len(np.where(np.abs(beat) > 0.5 * peak_val)[0])

    # Segmental energy (P-wave, QRS, T-wave zones)
    features['energy_pwave'] = np.sum(beat[0:40] ** 2)
    features['energy_qrs']   = np.sum(beat[70:110] ** 2)
    features['energy_twave'] = np.sum(beat[110:180] ** 2)
    features['qrs_t_ratio']  = (
        features['energy_qrs'] / (features['energy_twave'] + 1e-8)
    )

    # Frequency domain (FFT)
    fft_vals  = np.abs(np.fft.rfft(beat))
    fft_freqs = np.fft.rfftfreq(len(beat), d=1 / fs)

    lf = (fft_freqs >= 0.5) & (fft_freqs < 10)
    mf = (fft_freqs >= 10)  & (fft_freqs < 25)
    hf = (fft_freqs >= 25)  & (fft_freqs < 40)

    features['power_lf']         = np.sum(fft_vals[lf] ** 2)
    features['power_mf']         = np.sum(fft_vals[mf] ** 2)
    features['power_hf']         = np.sum(fft_vals[hf] ** 2)
    features['dominant_freq']    = fft_freqs[np.argmax(fft_vals)]
    features['spectral_entropy'] = -np.sum(
        (fft_vals / np.sum(fft_vals)) *
        np.log(fft_vals / np.sum(fft_vals) + 1e-8)
    )

    return features


def beats_to_features(beats):
    """Convert array of beat windows into a feature DataFrame."""
    return pd.DataFrame([extract_features(b) for b in beats])


def train(X_beats, y_labels):
    """Train Random Forest on extracted features. Returns fitted model."""
    X = beats_to_features(X_beats)
    y = (y_labels == 'V').astype(int)
    clf = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    clf.fit(X, y)
    return clf


def evaluate(clf, X_beats, y_labels):
    """Print classification report for given beats and true labels."""
    X = beats_to_features(X_beats)
    y = (y_labels == 'V').astype(int)
    y_pred = clf.predict(X)
    print(classification_report(y, y_pred,
          target_names=['Normal', 'Ventricular']))
    return confusion_matrix(y, y_pred)


def save_model(clf, path='model.joblib'):
    """Persist trained model to disk."""
    joblib.dump(clf, path)
    print(f"Model saved to {path}")


def load_model(path='model.joblib'):
    """Load model from disk."""
    return joblib.load(path)


if __name__ == '__main__':
    from preprocess import load_dataset

    print("Loading training data...")
    X_train, y_train = load_dataset(['100', '106', '119', '200'])

    print("Loading test data (unseen patient)...")
    X_test, y_test = load_dataset(['208'])

    print("Training model...")
    clf = train(X_train, y_train)

    print("\nEvaluation on unseen patient:")
    evaluate(clf, X_test, y_test)

    save_model(clf)