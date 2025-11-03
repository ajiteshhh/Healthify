from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from database import get_db_connection
from datetime import datetime
import numpy as np
import json
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Health Monitor API running 🚀"}

# connected clients (browsers)
clients = []

# buffer for last 5 seconds of ESP32 data
vitals_buffer = []
last_save_time = time.time()

async def save_avg_to_db():
    global vitals_buffer, last_save_time

    # every 5 seconds store avg
    if time.time() - last_save_time >= 5 and len(vitals_buffer) > 0:
        avg_hr = sum(v[0] for v in vitals_buffer) / len(vitals_buffer)
        avg_spo2 = sum(v[1] for v in vitals_buffer) / len(vitals_buffer)
        avg_temp = sum(v[2] for v in vitals_buffer) / len(vitals_buffer)

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO vitals (heart_rate, spo2, temperature) VALUES (%s, %s, %s)",
            (avg_hr, avg_spo2, avg_temp)
        )
        conn.commit()
        cur.close()
        conn.close()

        vitals_buffer = []
        last_save_time = time.time()


@app.websocket("/ws/esp32/vitals")
async def esp32_ws(websocket: WebSocket):
    global vitals_buffer

    await websocket.accept()
    print("✅ ESP32 vitals socket connected")

    try:
        while True:
            data = json.loads(await websocket.receive_text())
            bpm = data.get("bpm")
            spo2 = data.get("spo2")
            temp = data.get("temp")
            # push to buffer
            vitals_buffer.append((bpm, spo2, temp))

            # broadcast to UI
            for c in clients:
                await c.send_json({
                    "bpm": bpm,
                    "spo2": spo2,
                    "temp": temp
                })

            # periodically average and store
            await save_avg_to_db()

    except WebSocketDisconnect:
        print("❌ ESP32 disconnected")


@app.websocket("/ws/client/vitals")
async def client_ws(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    print("🖥️ Client connected")

    try:
        while True:
            await websocket.receive_text()  # not needed, UI just listens
    except WebSocketDisconnect:
        clients.remove(websocket)
        print("❌ Client disconnected")

# ===========================
# ✅ ECG WebSocket Handling
# ===========================

web_clients = []
ECG_BUFFER = np.array([])
active_sessions = {}  # websocket -> session_id


def save_ecg_batch(session_id, samples, rate, ts):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO ecg_batches (session_id, timestamp_start, sampling_rate, samples)
        VALUES (%s, %s, %s, %s)
        RETURNING batch_id
        """,
        (session_id, ts, rate, json.dumps(samples))
    )
    batch_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return batch_id


def save_beat_prediction(session_id, batch_id, beat, beat_index, arr_pred, stress_pred):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO ecg_beats_predictions
        (session_id, batch_id, beat_index, beat, arrhythmia_prediction, stress_prediction)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (session_id, batch_id, beat_index, json.dumps(beat), arr_pred, stress_pred)
    )
    conn.commit()
    cur.close()
    conn.close()


def process_ecg_batch(samples, fs=360):
    global ECG_BUFFER
    samples = np.array(samples)

    ECG_BUFFER = np.concatenate((ECG_BUFFER, samples))
    if len(ECG_BUFFER) > 800:
        ECG_BUFFER = ECG_BUFFER[-800:]

    diff = np.diff(ECG_BUFFER, prepend=ECG_BUFFER[0])
    squared = diff**2
    win = int(0.1 * fs)
    mwa = np.convolve(squared, np.ones(win)/win, mode='same')

    threshold = np.mean(mwa) + 2*np.std(mwa)
    peaks = [i for i in range(1, len(mwa)-1)
             if mwa[i] > threshold and mwa[i] > mwa[i-1] and mwa[i] > mwa[i+1]]

    beats = []
    wb = wa = 93

    for r in peaks:
        if r-wb >= 0 and r+wa < len(ECG_BUFFER):
            beat = ECG_BUFFER[r-wb:r+wa+1]
            if len(beat) == 187:
                beats.append(beat.tolist())

    return beats


@app.websocket("/ws/esp32")
async def ws_esp32(websocket: WebSocket):
    await websocket.accept()
    print("✅ ESP32 connected")

    try:
        while True:
            data = json.loads(await websocket.receive_text())

            session_id = data["session_id"]
            active_sessions[websocket] = session_id

            samples = data["samples"]
            ts = datetime.fromtimestamp(data["timestamp_start"])
            rate = data["sampling_rate"]

            batch_id = save_ecg_batch(session_id, samples, rate, ts)

            for c in web_clients:
                await c.send_json({"type": "ecg", "samples": samples})

            beats = process_ecg_batch(samples)

            for i, beat in enumerate(beats):
                arr = "Normal"
                stress = "No Stress"

                save_beat_prediction(session_id, batch_id, beat, i, arr, stress)

                for c in web_clients:
                    await c.send_json({
                        "type": "prediction",
                        "beat_index": i,
                        "arrhythmia": arr,
                        "stress": stress
                    })

    except WebSocketDisconnect:
        active_sessions.pop(websocket, None)
        print("❌ ESP32 disconnected")


@app.websocket("/ws/client")
async def ws_client(websocket: WebSocket):
    await websocket.accept()
    web_clients.append(websocket)
    print("🖥️ Client connected")

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        web_clients.remove(websocket)
        print("❌ Client disconnected")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
