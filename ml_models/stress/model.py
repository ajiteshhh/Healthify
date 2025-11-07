# model.py
"""
Train and save a stress classifier using either:
A) Precomputed features CSVs (RMSSD, SDNN, LF_HF, temp, bpm, spo2, condition)
B) Raw ECG in millivolts at 360 Hz + vitals (ecg_mv, bpm, spo2, temp, condition)
   -> features are computed per window (RMSSD, SDNN, LF/HF + window means)

Saves:
- stress_predictor_model.pkl
- stress_predictor_scaler.pkl
- (optional) stress_classes.pkl if labels are strings
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

# Try xgboost first; else fallback to RF
try:
    from xgboost import XGBClassifier
    USE_XGB = True
except Exception:
    from sklearn.ensemble import RandomForestClassifier
    USE_XGB = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# ECG processing
import neurokit2 as nk
from scipy import signal


FEATURES = ["RMSSD", "SDNN", "LF_HF", "temp", "bpm", "spo2"]
TARGET = "condition"

MODEL_PATH = "stress_predictor_model.pkl"
SCALER_PATH = "stress_predictor_scaler.pkl"
CLASSES_PATH = "stress_classes.pkl"


# ---------- ECG utils ----------
def bandpass_ecg(x, fs, low=0.5, high=40.0, order=4):
    b, a = signal.butter(order, [low/(fs/2), high/(fs/2)], btype="band")
    return signal.filtfilt(b, a, x)

def notch_filter(x, fs, f0=50.0, Q=30.0):
    # Use 60 Hz if your mains is 60
    b, a = signal.iirnotch(w0=f0/(fs/2), Q=Q)
    return signal.filtfilt(b, a, x)

def ecg_to_three_features(ecg_mv_segment, fs):
    """Return RMSSD, SDNN, LF/HF from a window of ECG in mV."""
    # Detrend and clean
    x = ecg_mv_segment - np.median(ecg_mv_segment)
    x = notch_filter(x, fs, f0=50.0)
    x = bandpass_ecg(x, fs, 0.5, 40.0)

    # R-peaks
    signals, info = nk.ecg_process(x, sampling_rate=fs)
    rpeaks = np.where(signals["ECG_R_Peaks"] == 1)[0]
    if len(rpeaks) < 2:
        return {"RMSSD": np.nan, "SDNN": np.nan, "LF_HF": np.nan}

    rr_ms = np.diff(rpeaks) / fs * 1000.0

    # Time domain
    rmssd = np.sqrt(np.mean(np.diff(rr_ms)**2)) if len(rr_ms) > 1 else np.nan
    sdnn  = np.std(rr_ms, ddof=1) if len(rr_ms) > 1 else np.nan

    # Frequency domain via NeuroKit
    hrv_freq = nk.hrv_frequency({"ECG_R_Peaks": rpeaks}, sampling_rate=fs, psd_method="welch", show=False)
    lf = float(hrv_freq["HRV_LF"].iloc[0])
    hf = float(hrv_freq["HRV_HF"].iloc[0])
    lf_hf = (lf / hf) if hf > 0 else np.nan

    return {"RMSSD": float(rmssd), "SDNN": float(sdnn), "LF_HF": float(lf_hf)}


def make_feature_table_from_raw(df, fs=360, win_sec=60, step_sec=30,
                                ecg_col="ecg_mv", bpm_col="bpm", spo2_col="spo2", temp_col="temp", label_col=TARGET):
    """
    Build per-window features from raw ECG mV + vitals streams.
    Required columns: ecg_col, bpm_col, spo2_col, temp_col
    Optional: label_col
    """
    assert ecg_col in df.columns, f"Missing ECG column '{ecg_col}'"
    n = len(df)
    win = int(win_sec * fs)
    step = int(step_sec * fs)

    rows = []
    for start in range(0, max(1, n - win + 1), step):
        end = start + win
        if end > n: break
        seg = df.iloc[start:end]

        feats = ecg_to_three_features(seg[ecg_col].to_numpy(dtype=float), fs)
        row = {
            **feats,
            "temp": float(np.nanmean(seg[temp_col])) if temp_col in df else np.nan,
            "bpm":  float(np.nanmean(seg[bpm_col]))  if bpm_col in df else np.nan,
            "spo2": float(np.nanmean(seg[spo2_col])) if spo2_col in df else np.nan,
        }
        if label_col in df.columns:
            mode_val = seg[label_col].mode(dropna=True)
            row[label_col] = mode_val.iloc[0] if len(mode_val) else np.nan

        rows.append(row)

    feat = pd.DataFrame(rows)
    keep = FEATURES.copy()
    if label_col in feat.columns:
        keep += [label_col]
    feat = feat[keep]
    return feat


# ---------- Training paths ----------
def _fit_and_save(X, y, label_is_str=False):
    classes = None
    if label_is_str:
        classes = sorted(pd.Series(y).dropna().unique().tolist())
        mapping = {c:i for i,c in enumerate(classes)}
        y = np.array([mapping.get(v, np.nan) for v in y], dtype=float)

    good = ~np.isnan(y)
    X, y = X[good], y[good].astype(int)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.25, random_state=42, stratify=y)

    if USE_XGB:
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=42, n_jobs=-1
        )
    else:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1
        )

    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)

    print("Accuracy:", accuracy_score(yte, ypred))
    print(classification_report(yte, ypred))

    # Save
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    if classes is not None:
        joblib.dump(classes, CLASSES_PATH)
        print(f"[OK] saved {MODEL_PATH}, {SCALER_PATH}, {CLASSES_PATH}")
    else:
        print(f"[OK] saved {MODEL_PATH}, {SCALER_PATH}")


def train_from_feature_csv(train_csv, test_csv=None):
    """Train using precomputed features CSV(s)."""
    train_df = pd.read_csv(train_csv)
    if test_csv is not None:
        test_df = pd.read_csv(test_csv)
        df = pd.concat([train_df, test_df], ignore_index=True)
    else:
        df = train_df

    assert set(FEATURES).issubset(df.columns), f"Missing some of {FEATURES}"
    assert TARGET in df.columns, f"Missing label column '{TARGET}'"

    X = df[FEATURES].values
    y = df[TARGET].values
    _fit_and_save(X, y, label_is_str=(y.dtype==object))


def train_from_raw_csv(raw_csv, fs=360, win_sec=60, step_sec=30,
                       ecg_col="ecg_mv", bpm_col="bpm", spo2_col="spo2", temp_col="temp", label_col=TARGET):
    """Train from raw ECG mV + vitals. One row per sample, 360 Hz sampling preferred."""
    df = pd.read_csv(raw_csv)
    feat = make_feature_table_from_raw(df, fs, win_sec, step_sec, ecg_col, bpm_col, spo2_col, temp_col, label_col)
    feat = feat.dropna(subset=FEATURES).reset_index(drop=True)

    assert label_col in feat.columns, "Label column required for training."
    X = feat[FEATURES].values
    y = feat[label_col].values
    _fit_and_save(X, y, label_is_str=(y.dtype==object))


if __name__ == "__main__":
    if os.path.exists("stress_minimal_train_16k.csv"):
        train_from_feature_csv("stress_minimal_train_16k.csv", "stress_minimal_test_4k.csv")
