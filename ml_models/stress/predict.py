
from pathlib import Path

predict_py = r

import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy import signal

FEATURES = ["RMSSD", "SDNN", "LF_HF", "temp", "bpm", "spo2"]
MODEL_PATH = "stress_predictor_model.pkl"
SCALER_PATH = "stress_predictor_scaler.pkl"
CLASSES_PATH = "stress_classes.pkl"  # optional


# ---------- ECG utils ----------
def bandpass_ecg(x, fs, low=0.5, high=40.0, order=4):
    b, a = signal.butter(order, [low/(fs/2), high/(fs/2)], btype="band")
    return signal.filtfilt(b, a, x)

def notch_filter(x, fs, f0=50.0, Q=30.0):
    b, a = signal.iirnotch(w0=f0/(fs/2), Q=Q)
    return signal.filtfilt(b, a, x)

def ecg_to_three_features(ecg_mv_segment, fs, mains=50.0):
    # center & clean
    x = np.asarray(ecg_mv_segment, dtype=float)
    x = x - np.median(x)
    if mains:
        x = notch_filter(x, fs, f0=float(mains))
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

    # Frequency domain with NeuroKit
    hrv_freq = nk.hrv_frequency({"ECG_R_Peaks": rpeaks}, sampling_rate=fs, psd_method="welch", show=False)
    lf = float(hrv_freq["HRV_LF"].iloc[0])
    hf = float(hrv_freq["HRV_HF"].iloc[0])
    lf_hf = (lf / hf) if hf > 0 else np.nan
    return {"RMSSD": float(rmssd), "SDNN": float(sdnn), "LF_HF": float(lf_hf)}


# ---------- Load model/scaler/classes ----------
def _load_artifacts(model_path=MODEL_PATH, scaler_path=SCALER_PATH, classes_path=CLASSES_PATH):
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Missing model/scaler .pkl files in current directory.")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    classes = joblib.load(classes_path) if os.path.exists(classes_path) else None
    return model, scaler, classes


# ---------- Predictors ----------
def predict_from_features(sample_dict, artifacts=None):
    model, scaler, classes = artifacts or _load_artifacts()
    # Coerce to float in correct order
    row = {k: float(sample_dict[k]) for k in FEATURES}
    df = pd.DataFrame([row], columns=FEATURES)
    Xs = scaler.transform(df)
    pred = model.predict(Xs)[0]
    label = classes[pred] if classes else ("stress" if pred == 1 else "no_stress")
    # Also include probability if available
    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(Xs)[0, 1]) if (not classes or len(getattr(model, "classes_", []))==2) else None
    return {"input": row, "pred": int(pred) if not classes else label, "label": label, "stress_prob": proba}


def predict_from_raw_buffers(ecg_mv, bpm, spo2, temp, fs=360, win_sec=60, step_sec=30):
    """
    ecg_mv: 360Hz array (float, length = win_sec * fs)
    bpm/spo2/temp: 1Hz arrays (float, length = win_sec)
    """
    model, scaler, classes = _load_artifacts()

    # --- make sure vitals match duration ---
    ecg_mv = np.asarray(ecg_mv, dtype=float)
    n = len(ecg_mv)
    expected_secs = n / fs
    t = np.linspace(0, expected_secs, n)

    # upsample vitals (1Hz → 360Hz) by repeating each value fs times
    bpm = np.repeat(bpm, fs)
    spo2 = np.repeat(spo2, fs)
    temp = np.repeat(temp, fs)

    # trim vitals if slightly longer
    min_len = min(len(ecg_mv), len(bpm))
    ecg_mv, bpm, spo2, temp = ecg_mv[:min_len], bpm[:min_len], spo2[:min_len], temp[:min_len]

    # --- sliding window (60s, step 30s) ---
    win = int(win_sec * fs)
    step = int(step_sec * fs)
    rows = []

    for start in range(0, max(1, n - win + 1), step):
        end = start + win
        if end > n:
            break

        ecg_seg = ecg_mv[start:end]
        feats = ecg_to_three_features(ecg_seg, fs)

        # vitals: mean of their 60s worth of repeated values
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

    out = feat_df.copy()
    out["pred"] = labels
    return out


# ---------- CLI ----------
def parse_kv_list(kvs):
    # Parse key=value pairs
    out = {}
    for kv in kvs:
        if "=" not in kv:
            raise ValueError(f"Bad key=value: {kv}")
        k,v = kv.split("=",1)
        out[k] = v
    # ensure all features provided
    missing = [k for k in FEATURES if k not in out]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    return out

def main():
    ap = argparse.ArgumentParser(description="Stress predictor (features or raw ECG)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--from-features", nargs="+", help="key=val pairs for RMSSD,SDNN,LF_HF,temp,bpm,spo2")
    g.add_argument("--from-raw-csv", type=str, help="CSV with columns: ecg_mv,bpm,spo2,temp")
    g.add_argument("--from-arrays", type=str, help="NPZ file with arrays: ecg_mv,bpm,spo2,temp")
    ap.add_argument("--fs", type=int, default=360, help="Sampling rate for ECG (Hz)")
    ap.add_argument("--win", type=int, default=60, help="Window sec")
    ap.add_argument("--step", type=int, default=30, help="Step sec")
    ap.add_argument("--mains", type=float, default=50.0, help="Mains notch (50 or 60)")
    ap.add_argument("--save", type=str, default=None, help="Optional path to save per-window predictions CSV/JSON")

    args = ap.parse_args()
    artifacts = _load_artifacts()

    if args.from_features:
        sample = parse_kv_list(args.from_features)
        res = predict_from_features(sample, artifacts=artifacts)
        print(json.dumps(res, indent=2))
        if args.save:
            pd.DataFrame([res]).to_csv(args.save, index=False)
        return

    if args.from_raw_csv:
        df = pd.read_csv(args.from_raw_csv)
        required = ["ecg_mv","bpm","spo2","temp"]
        if not set(required).issubset(df.columns):
            missing = list(set(required)-set(df.columns))
            raise ValueError(f"Missing columns in CSV: {missing}")
        out = predict_from_raw_arrays(
            df["ecg_mv"].to_numpy(), df["bpm"].to_numpy(), df["spo2"].to_numpy(), df["temp"].to_numpy(),
            fs=args.fs, win_sec=args.win, step_sec=args.step, mains=args.mains, artifacts=artifacts
        )
        print(out.tail().to_string(index=False))
        if args.save:
            out.to_csv(args.save, index=False)
        return

    if args.from_arrays:
        data = np.load(args.from_arrays)
        for k in ["ecg_mv","bpm","spo2","temp"]:
            if k not in data:
                raise ValueError(f"NPZ missing array: {k}")
        out = predict_from_raw_arrays(
            data["ecg_mv"], data["bpm"], data["spo2"], data["temp"],
            fs=args.fs, win_sec=args.win, step_sec=args.step, mains=args.mains, artifacts=artifacts
        )
        print(out.tail().to_string(index=False))
        if args.save:
            out.to_csv(args.save, index=False)
        return

if __name__ == "__main__":
    main()

Path("/mnt/data/predict.py").write_text(predict_py)
print("/mnt/data/predict.py")
