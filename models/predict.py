import os
import json
import joblib
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy import signal

# ---------- CONFIG ----------
FEATURES = ["RMSSD", "SDNN", "LF_HF", "temp", "bpm", "spo2"]
MODEL_PATH = "stress_predictor_model.pkl"
SCALER_PATH = "stress_predictor_scaler.pkl"
CLASSES_PATH = "stress_classes.pkl"  # optional
JSON_PATH = "/input.json"

# ---------- ECG UTILS ----------
def bandpass_ecg(x, fs, low=0.5, high=40.0, order=4):
    b, a = signal.butter(order, [low/(fs/2), high/(fs/2)], btype="band")
    return signal.filtfilt(b, a, x)

def notch_filter(x, fs, f0=50.0, Q=30.0):
    b, a = signal.iirnotch(w0=f0/(fs/2), Q=Q)
    return signal.filtfilt(b, a, x)

def ecg_to_three_features(ecg_mv_segment, fs, mains=50.0):
    x = np.asarray(ecg_mv_segment, dtype=float)
    x = x - np.median(x)
    if mains:
        x = notch_filter(x, fs, f0=float(mains))
    x = bandpass_ecg(x, fs, 0.5, 40.0)

    signals, info = nk.ecg_process(x, sampling_rate=fs)
    rpeaks = np.where(signals["ECG_R_Peaks"] == 1)[0]
    if len(rpeaks) < 2:
        return {"RMSSD": np.nan, "SDNN": np.nan, "LF_HF": np.nan}

    rr_ms = np.diff(rpeaks) / fs * 1000.0
    rmssd = np.sqrt(np.mean(np.diff(rr_ms)**2)) if len(rr_ms) > 1 else np.nan
    sdnn = np.std(rr_ms, ddof=1) if len(rr_ms) > 1 else np.nan

    hrv_freq = nk.hrv_frequency({"ECG_R_Peaks": rpeaks}, sampling_rate=fs, psd_method="welch", show=False)
    lf = float(hrv_freq["HRV_LF"].iloc[0])
    hf = float(hrv_freq["HRV_HF"].iloc[0])
    lf_hf = (lf / hf) if hf > 0 else np.nan
    return {"RMSSD": rmssd, "SDNN": sdnn, "LF_HF": lf_hf}


# ---------- MODEL LOADING ----------
def _load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Missing model/scaler .pkl files.")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    classes = joblib.load(CLASSES_PATH) if os.path.exists(CLASSES_PATH) else None
    return model, scaler, classes


# ---------- MAIN FUNCTION ----------
def predict_from_json(json_path=JSON_PATH):
    with open(json_path, "r") as f:
        data = json.load(f)

    fs = int(data.get("fs", 360))
    win_sec = int(data.get("win_sec", 60))
    step_sec = int(data.get("step_sec", 30))

    ecg_mv = np.array(data["ecg_mv"], dtype=float)
    bpm = np.array(data["bpm"], dtype=float)
    spo2 = np.array(data["spo2"], dtype=float)
    temp = np.array(data["temp"], dtype=float)

    model, scaler, classes = _load_artifacts()

    # Upsample vitals to match ECG sampling rate (1Hz → 360Hz)
    bpm = np.repeat(bpm, fs)
    spo2 = np.repeat(spo2, fs)
    temp = np.repeat(temp, fs)

    min_len = min(len(ecg_mv), len(bpm))
    ecg_mv, bpm, spo2, temp = ecg_mv[:min_len], bpm[:min_len], spo2[:min_len], temp[:min_len]

    win = int(win_sec * fs)
    step = int(step_sec * fs)
    rows = []

    for start in range(0, max(1, len(ecg_mv) - win + 1), step):
        end = start + win
        if end > len(ecg_mv):
            break

        ecg_seg = ecg_mv[start:end]
        feats = ecg_to_three_features(ecg_seg, fs)
        row = {
            **feats,
            "temp": float(np.nanmean(temp[start:end])),
            "bpm": float(np.nanmean(bpm[start:end])),
            "spo2": float(np.nanmean(spo2[start:end]))
        }
        rows.append(row)

    feat_df = pd.DataFrame(rows, columns=FEATURES)
    Xs = scaler.transform(feat_df)
    pred = model.predict(Xs)

    if classes:
        labels = [classes[i] for i in pred]
    else:
        labels = ["stress" if p == 1 else "no_stress" for p in pred]

    feat_df["pred"] = labels
    last_label = labels[-1]

    print(f"\n🧠 Predicted condition: {last_label.upper()}")
    print(feat_df.tail(1).to_string(index=False))

    return last_label


# ---------- RUN ----------
if __name__ == "__main__":
    predict_from_json(JSON_PATH)