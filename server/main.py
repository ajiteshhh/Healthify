from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from database import get_db_connection
from datetime import datetime
import numpy as np
import json

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


# ===========================
# ✅ HR + SpO2 Endpoints
# ===========================

@app.post("/api/v1/vitals/heartrate_spo2")
async def post_hr_spo2(request: Request):
    data = await request.json()
    hr = data.get("heartRate")
    spo2 = data.get("spo2")

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO vitals_heartrate_spo2 (heart_rate, spo2) VALUES (%s, %s)",
        (hr, spo2)
    )
    conn.commit()
    cur.close()
    conn.close()

    return {"success": True, "message": "HR & SpO2 saved"}


@app.get("/api/v1/vitals/heartrate_spo2")
def get_hr_spo2():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM vitals_heartrate_spo2 ORDER BY id DSC")
    rows = cur.fetchall()

    result = [{"id": r[0], "heart_rate": r[1], "spo2": r[2], "timestamp": r[3]} for r in rows]

    cur.close()
    conn.close()
    return result


# ===========================
# ✅ Temperature Endpoints
# ===========================

@app.post("/api/v1/vitals/temperature")
async def post_temperature(request: Request):
    data = await request.json()
    temp = data.get("temperature")

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO vitals_temperature (temperature) VALUES (%s)",
        (temp,)
    )
    conn.commit()
    cur.close()
    conn.close()

    return {"success": True, "message": "Temperature saved"}


@app.get("/api/v1/vitals/temperature")
def get_temperature():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM vitals_temperature ORDER BY id DSC")
    rows = cur.fetchall()

    result = [{"id": r[0], "temperature": r[1], "timestamp": r[2]} for r in rows]

    cur.close()
    conn.close()
    return result


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
