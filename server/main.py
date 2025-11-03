from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from database import get_db_connection
from datetime import datetime
import numpy as np
import json
import time
import os
import asyncio

app = FastAPI()

# CORS configuration for Render deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("✅ FastAPI routes loaded:", [route.path for route in app.router.routes])

# Health check endpoints for Render
@app.get("/")
def home():
    return {"message": "Health Monitor API running 🚀", "status": "online"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# ===========================
# ✅ Vitals WebSocket Handling
# ===========================

# Connected clients (browsers)
vitals_clients = []

# Buffer for last 5 seconds of ESP32 data
vitals_buffer = []
last_save_time = time.time()

async def save_avg_to_db():
    """Save averaged vitals to database every 5 seconds"""
    global vitals_buffer, last_save_time

    # Every 5 seconds store avg
    if time.time() - last_save_time >= 5 and len(vitals_buffer) > 0:
        try:
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

            print(f"✅ Saved vitals avg: HR={avg_hr:.1f}, SpO2={avg_spo2:.1f}, Temp={avg_temp:.1f}")

            vitals_buffer = []
            last_save_time = time.time()
        except Exception as e:
            print(f"❌ Error saving vitals to DB: {e}")


@app.websocket("/ws/esp32/vitals")
async def esp32_vitals_ws(websocket: WebSocket):
    """WebSocket endpoint for ESP32 to send vitals data"""
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
                
                # Validate data
                if bpm is None or spo2 is None or temp is None:
                    print("⚠️ Incomplete vitals data received")
                    continue
                
                # Push to buffer
                vitals_buffer.append((bpm, spo2, temp))

                # Broadcast to UI clients
                disconnected = []
                for c in vitals_clients:
                    try:
                        await c.send_json({
                            "type": "vitals",
                            "bpm": bpm,
                            "spo2": spo2,
                            "temp": temp,
                            "timestamp": datetime.now().isoformat()
                        })
                    except Exception as e:
                        print(f"⚠️ Error sending to client: {e}")
                        disconnected.append(c)
                
                # Remove disconnected clients
                for c in disconnected:
                    if c in vitals_clients:
                        vitals_clients.remove(c)

                # Periodically average and store
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
    """WebSocket endpoint for web clients to receive vitals data"""
    await websocket.accept()
    vitals_clients.append(websocket)
    print(f"🖥️ Vitals client connected (total: {len(vitals_clients)})")

    try:
        while True:
            # Keep connection alive, client just listens
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in vitals_clients:
            vitals_clients.remove(websocket)
        print(f"❌ Vitals client disconnected (remaining: {len(vitals_clients)})")
    except Exception as e:
        if websocket in vitals_clients:
            vitals_clients.remove(websocket)
        print(f"❌ Vitals client error: {e}")


# ===========================
# ✅ ECG WebSocket Handling
# ===========================

ecg_clients = []
ECG_BUFFER = np.array([])
active_sessions = {}  # websocket -> session_id


def save_ecg_batch(session_id, samples, rate, ts):
    """Save ECG batch to database"""
    try:
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
    except Exception as e:
        print(f"❌ Error saving ECG batch: {e}")
        return None


def save_beat_prediction(session_id, batch_id, beat, beat_index, arr_pred, stress_pred):
    """Save beat prediction to database"""
    try:
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
    except Exception as e:
        print(f"❌ Error saving beat prediction: {e}")


def process_ecg_batch(samples, fs=360):
    """Process ECG samples to detect beats"""
    global ECG_BUFFER
    
    try:
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
    except Exception as e:
        print(f"❌ Error processing ECG batch: {e}")
        return []


@app.websocket("/ws/esp32/ecg")
async def esp32_ecg_ws(websocket: WebSocket):
    """WebSocket endpoint for ESP32 to send ECG data"""
    await websocket.accept()
    print("✅ ESP32 ECG connected")

    try:
        while True:
            raw_data = await websocket.receive_text()
            
            try:
                data = json.loads(raw_data)

                session_id = data.get("session_id")
                if not session_id:
                    print("⚠️ No session_id in ECG data")
                    continue
                    
                active_sessions[websocket] = session_id

                samples = data.get("samples", [])
                timestamp_start = data.get("timestamp_start")
                rate = data.get("sampling_rate", 360)

                if not samples or not timestamp_start:
                    print("⚠️ Incomplete ECG data received")
                    continue

                ts = datetime.fromtimestamp(timestamp_start)

                # Save batch to database
                batch_id = save_ecg_batch(session_id, samples, rate, ts)
                
                if not batch_id:
                    continue

                # Broadcast raw ECG to clients
                disconnected = []
                for c in ecg_clients:
                    try:
                        await c.send_json({
                            "type": "ecg",
                            "samples": samples,
                            "timestamp": datetime.now().isoformat()
                        })
                    except Exception as e:
                        print(f"⚠️ Error sending ECG to client: {e}")
                        disconnected.append(c)
                
                # Remove disconnected clients
                for c in disconnected:
                    if c in ecg_clients:
                        ecg_clients.remove(c)

                # Process ECG to detect beats
                beats = process_ecg_batch(samples)

                # Save predictions for each beat
                for i, beat in enumerate(beats):
                    arr = "Normal"
                    stress = "No Stress"

                    save_beat_prediction(session_id, batch_id, beat, i, arr, stress)

                    # Broadcast predictions
                    for c in ecg_clients:
                        try:
                            await c.send_json({
                                "type": "prediction",
                                "beat_index": i,
                                "arrhythmia": arr,
                                "stress": stress,
                                "timestamp": datetime.now().isoformat()
                            })
                        except Exception as e:
                            print(f"⚠️ Error sending prediction to client: {e}")
                            
            except json.JSONDecodeError as e:
                print(f"❌ Invalid JSON from ESP32 ECG: {e}")
                continue

    except WebSocketDisconnect:
        active_sessions.pop(websocket, None)
        print("❌ ESP32 ECG disconnected")
    except Exception as e:
        active_sessions.pop(websocket, None)
        print(f"❌ ESP32 ECG error: {e}")


@app.websocket("/ws/client/ecg")
async def client_ecg_ws(websocket: WebSocket):
    """WebSocket endpoint for web clients to receive ECG data"""
    await websocket.accept()
    ecg_clients.append(websocket)
    print(f"🖥️ ECG client connected (total: {len(ecg_clients)})")

    try:
        while True:
            # Keep connection alive, client just listens
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in ecg_clients:
            ecg_clients.remove(websocket)
        print(f"❌ ECG client disconnected (remaining: {len(ecg_clients)})")
    except Exception as e:
        if websocket in ecg_clients:
            ecg_clients.remove(websocket)
        print(f"❌ ECG client error: {e}")


# ===========================
# ✅ Legacy endpoints (kept for backward compatibility)
# ===========================

@app.websocket("/ws/esp32")
async def ws_esp32_legacy(websocket: WebSocket):
    """Legacy ESP32 endpoint - redirects to /ws/esp32/ecg"""
    print("⚠️ Using legacy /ws/esp32 endpoint, consider using /ws/esp32/ecg")
    await esp32_ecg_ws(websocket)


@app.websocket("/ws/client")
async def ws_client_legacy(websocket: WebSocket):
    """Legacy client endpoint - redirects to /ws/client/ecg"""
    print("⚠️ Using legacy /ws/client endpoint, consider using /ws/client/ecg")
    await client_ecg_ws(websocket)


if __name__ == "__main__":
    import uvicorn
    # Get port from environment variable (Render sets this automatically)
    port = int(os.environ.get("PORT", 8000))
    
    print(f"🚀 Starting server on port {port}")
    
    # Disable reload in production (Render deployment)
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Set to False for production
        log_level="info"
    )