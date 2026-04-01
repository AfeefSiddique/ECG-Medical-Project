import numpy as np
import wfdb
import neurokit2 as nk


def load_record(record_name, pn_dir='mitdb'):
    """Load a single MIT-BIH record and its annotations."""
    record = wfdb.rdrecord(record_name, pn_dir=pn_dir)
    annotation = wfdb.rdann(record_name, 'atr', pn_dir=pn_dir)
    signal = record.p_signal[:, 0]
    fs = record.fs
    return signal, annotation, fs


def clean_signal(signal, fs=360):
    """Apply bandpass filter to remove baseline wander and muscle noise."""
    return nk.ecg_clean(signal, sampling_rate=fs)


def detect_rpeaks(cleaned_signal, fs=360):
    """Detect R-peak positions in cleaned ECG signal."""
    _, info = nk.ecg_peaks(cleaned_signal, sampling_rate=fs)
    return info['ECG_R_Peaks']


def segment_beats(cleaned_signal, r_peaks, annotation,
                  before=90, after=110, valid_labels=('N', 'V')):
    """
    Extract individual beat windows centered on each R-peak.
    Returns beats array and corresponding labels.
    """
    ann_map = dict(zip(annotation.sample, annotation.symbol))
    beats, labels = [], []

    for peak in r_peaks:
        if peak - before < 0 or peak + after >= len(cleaned_signal):
            continue
        closest = min(annotation.sample, key=lambda x: abs(x - peak))
        if abs(closest - peak) >= 50:
            continue
        label = ann_map[closest]
        if label in valid_labels:
            beats.append(cleaned_signal[peak - before: peak + after])
            labels.append(label)

    return np.array(beats), np.array(labels)


def load_dataset(record_names, valid_labels=('N', 'V')):
    """
    Load and process multiple MIT-BIH records into a
    single beats array and labels array.
    """
    all_beats, all_labels = [], []

    for name in record_names:
        print(f"  Processing record {name}...")
        signal, annotation, fs = load_record(name)
        cleaned = clean_signal(signal, fs)
        r_peaks = detect_rpeaks(cleaned, fs)
        beats, labels = segment_beats(
            cleaned, r_peaks, annotation,
            valid_labels=valid_labels
        )
        all_beats.append(beats)
        all_labels.append(labels)
        print(f"    -> {len(beats)} beats extracted")

    return np.vstack(all_beats), np.concatenate(all_labels)


if __name__ == '__main__':
    print("Testing preprocess pipeline...")
    beats, labels = load_dataset(['100', '106'])
    print(f"beats shape : {beats.shape}")
    print(f"label counts: N={sum(labels=='N')}, V={sum(labels=='V')}")