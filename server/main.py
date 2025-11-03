import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from database import get_db_connection
from datetime import datetime
import numpy as np
import json
import time
import os
import struct
import joblib
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "..", "models"))
from predict import predict_from_json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("✅ FastAPI routes loaded:", [route.path for route in app.router.routes])


@app.get("/")
def home():
    return {"message": "Health Monitor API running 🚀", "status": "online"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


vitals_clients = []
vitals_buffer = []
last_save_time = time.time()

WIN_SEC = 60
FS = 360
ECG_TARGET = WIN_SEC * FS
VITALS_TARGET = 60

vitals_window = {"bpm": [], "spo2": [], "temp": []}
ecg_window = []


def save_model_input_json():
    save_path = os.path.join(BASE_DIR, "..", "models", "input.json")

    payload = {
        "fs": FS,
        "win_sec": WIN_SEC,
        "step_sec": 30,
        "ecg_mv": ecg_window.copy(),
        "bpm": vitals_window["bpm"].copy(),
        "spo2": vitals_window["spo2"].copy(),
        "temp": vitals_window["temp"].copy(),
    }

    try:
        with open(save_path, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"✅ Saved model input → {save_path}")
        stress_result = predict_from_json(save_path)
        print(f"🧠 Stress prediction → {stress_result}")

        for c in ecg_clients:
            try:
                asyncio.create_task(
                    c.send_json(
                        {
                            "type": "stress",
                            "class": stress_result,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                )
            except:
                if c in ecg_clients:
                    ecg_clients.remove(c)

    except Exception as e:
        print(f"❌ Error in save_model_input_json: {e}")


async def save_avg_to_db():
    global vitals_buffer, last_save_time

    if time.time() - last_save_time >= 5 and len(vitals_buffer) > 0:
        try:
            avg_hr = sum(v[0] for v in vitals_buffer) / len(vitals_buffer)
            avg_spo2 = sum(v[1] for v in vitals_buffer) / len(vitals_buffer)
            avg_temp = sum(v[2] for v in vitals_buffer) / len(vitals_buffer)

            if avg_hr > 30 and avg_spo2 > 50 and avg_temp > 20:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO vitals (heart_rate, spo2, temperature) VALUES (%s, %s, %s)",
                    (avg_hr, avg_spo2, avg_temp),
                )
                conn.commit()
                cur.close()
                conn.close()

                print(
                    f"✅ Saved vitals avg: HR={avg_hr:.1f}, SpO2={avg_spo2:.1f}, Temp={avg_temp:.1f}"
                )

            vitals_buffer = []
            last_save_time = time.time()
        except Exception as e:
            print(f"❌ Error saving vitals to DB: {e}")


@app.websocket("/ws/esp32/vitals")
async def esp32_vitals_ws(websocket: WebSocket):
    global vitals_buffer

    await websocket.accept()
    print("✅ ESP32 vitals socket connected")

    try:
        while True:
            raw_data = await websocket.receive_text()

            try:
                data = json.loads(raw_data)
                bpm = data.get("bpm")
                spo2 = data.get("spo2")
                temp = data.get("temp")

                if bpm is None or spo2 is None or temp is None:
                    print("⚠️ Incomplete vitals data received")
                    continue

                vitals_buffer.append((bpm, spo2, temp))
                vitals_window["bpm"].append(bpm)
                vitals_window["spo2"].append(spo2)
                vitals_window["temp"].append(temp)

                for key in vitals_window:
                    if len(vitals_window[key]) > VITALS_TARGET:
                        vitals_window[key].pop(0)

                if (
                    len(vitals_window["bpm"]) == VITALS_TARGET
                    and len(ecg_window) == ECG_TARGET
                ):
                    save_model_input_json()

                disconnected = []
                for c in vitals_clients:
                    try:
                        await c.send_json(
                            {
                                "type": "vitals",
                                "bpm": bpm,
                                "spo2": spo2,
                                "temp": temp,
                                "timestamp": datetime.now().isoformat(),
                            }
                        )
                    except Exception as e:
                        print(f"⚠️ Error sending to client: {e}")
                        disconnected.append(c)

                for c in disconnected:
                    if c in vitals_clients:
                        vitals_clients.remove(c)

                await save_avg_to_db()

            except json.JSONDecodeError as e:
                print(f"❌ Invalid JSON from ESP32: {e}")
                continue

    except WebSocketDisconnect:
        print("❌ ESP32 vitals disconnected")
    except Exception as e:
        print(f"❌ ESP32 vitals error: {e}")


@app.websocket("/ws/client/vitals")
async def client_vitals_ws(websocket: WebSocket):
    await websocket.accept()
    vitals_clients.append(websocket)
    print(f"🖥️ Vitals client connected (total: {len(vitals_clients)})")

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in vitals_clients:
            vitals_clients.remove(websocket)
        print(f"❌ Vitals client disconnected (remaining: {len(vitals_clients)})")
    except Exception as e:
        if websocket in vitals_clients:
            vitals_clients.remove(websocket)
        print(f"❌ Vitals client error: {e}")


ecg_clients = []
ECG_BUFFER = np.array([], dtype=float)
MAX_BUFFER = 2000
BATCH_SIZE = 50


def detect_beats_and_extract_187(samples):
    global ECG_BUFFER

    samples = np.array(samples, dtype=float)
    ECG_BUFFER = np.concatenate((ECG_BUFFER, samples))

    if len(ECG_BUFFER) > MAX_BUFFER:
        ECG_BUFFER = ECG_BUFFER[-MAX_BUFFER:]

    diff = np.diff(ECG_BUFFER, prepend=ECG_BUFFER[0])
    squared = diff ** 2
    win = int(0.1 * 360)
    mwa = np.convolve(squared, np.ones(win) / win, mode="same")

    threshold = np.mean(mwa) + 2 * np.std(mwa)
    peaks = [
        i
        for i in range(1, len(mwa) - 1)
        if mwa[i] > threshold and mwa[i] > mwa[i - 1] and mwa[i] > mwa[i + 1]
    ]

    beats = []
    r = 93

    for p in peaks:
        if p - r >= 0 and p + r < len(ECG_BUFFER):
            beat = ECG_BUFFER[p - r : p + r + 1]
            if len(beat) == 187:
                beats.append(beat.tolist())

    return beats


arr_labels = {
    0: "Normal beat",
    1: "Supraventricular premature beat",
    2: "Ventricular premature beat",
    3: "Fusion beat",
    4: "Unclassifiable beat",
}


async def run_model_on_beat(beat_187):
    beat_arr = np.array(beat_187).reshape(1, -1)
    arr_class = arr_model.predict(beat_arr)[0]
    arr = arr_labels.get(arr_class, str(arr_class))
    return arr


@app.websocket("/ws/client/ecg")
async def client_ecg_ws(websocket: WebSocket):
    await websocket.accept()
    ecg_clients.append(websocket)
    print(f"🖥️ ECG client connected (total: {len(ecg_clients)})")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in ecg_clients:
            ecg_clients.remove(websocket)
        print("❌ ECG client disconnected")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "arrhythmia_model.pkl")

arr_model = joblib.load(MODEL_PATH)
print(f"✅ Model loaded successfully from {MODEL_PATH}")


@app.websocket("/ws/esp32/ecg")
async def esp32_ecg_ws(websocket: WebSocket):
    await websocket.accept()
    print("✅ ESP32 ECG connected")

    temp_batch = []

    try:
        while True:
            message = await websocket.receive_bytes()

            if len(message) != 4:
                continue

            (mV,) = struct.unpack("<f", message)
            temp_batch.append(mV)
            ecg_window.append(mV)

            if len(ecg_window) > ECG_TARGET:
                ecg_window.pop(0)

            if len(ecg_window) == ECG_TARGET and len(vitals_window["bpm"]) == VITALS_TARGET:
                save_model_input_json()

            if len(temp_batch) >= BATCH_SIZE:
                samples = temp_batch[:]
                temp_batch = []

                for c in ecg_clients:
                    try:
                        await c.send_json({"type": "ecg", "samples": samples})
                    except:
                        ecg_clients.remove(c)

                beats = detect_beats_and_extract_187(samples)
                for beat in beats:
                    pred = await run_model_on_beat(beat)
                    for c in ecg_clients:
                        try:
                            await c.send_json(
                                {
                                    "type": "prediction",
                                    "label": "arrhythmia",
                                    "class": pred,
                                    "beat": beat,
                                }
                            )
                        except:
                            ecg_clients.remove(c)

    except WebSocketDisconnect:
        print("❌ ESP32 ECG disconnected")
    except Exception as e:
        print(f"❌ Error: {e}")


@app.websocket("/ws/esp32")
async def ws_esp32_legacy(websocket: WebSocket):
    print("⚠️ Using legacy /ws/esp32 endpoint, consider using /ws/esp32/ecg")
    await esp32_ecg_ws(websocket)


@app.websocket("/ws/client")
async def ws_client_legacy(websocket: WebSocket):
    print("⚠️ Using legacy /ws/client endpoint, consider using /ws/client/ecg")
    await client_ecg_ws(websocket)


@app.get("/api/vitals")
def get_all_vitals():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, heart_rate, spo2, temperature, timestamp FROM vitals ORDER BY id ASC"
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()

        data = []
        for row in rows:
            data.append(
                {
                    "id": row[0],
                    "heart_rate": row[1],
                    "spo2": row[2],
                    "temperature": row[3],
                    "timestamp": row[4].isoformat() if row[4] else None,
                }
            )

        return {"count": len(data), "vitals": data}

    except Exception as e:
        print(f"❌ Error fetching vitals: {e}")
        return {"error": "Failed to fetch vitals"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting server on port {port}")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )