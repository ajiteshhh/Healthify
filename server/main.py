from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from database import get_db_connection
from datetime import datetime
import numpy as np
import json
import time
import os
import struct

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

            if avg_hr > 30 and avg_spo2 > 50 and avg_temp > 20:
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


# ================================
# ✅ ECG Stream + Beat Detection
# ================================

ecg_clients = []
ECG_BUFFER = np.array([], dtype=float)
MAX_BUFFER = 2000  # ~5.5 seconds buffer @ 360Hz
BATCH_SIZE = 50    # Send + process in chunks


def detect_beats_and_extract_187(samples):
    global ECG_BUFFER

    samples = np.array(samples, dtype=float)
    ECG_BUFFER = np.concatenate((ECG_BUFFER, samples))

    if len(ECG_BUFFER) > MAX_BUFFER:
        ECG_BUFFER = ECG_BUFFER[-MAX_BUFFER:]

    diff = np.diff(ECG_BUFFER, prepend=ECG_BUFFER[0])
    squared = diff**2
    win = int(0.1 * 360)
    mwa = np.convolve(squared, np.ones(win)/win, mode="same")

    threshold = np.mean(mwa) + 2*np.std(mwa)
    peaks = [
        i for i in range(1, len(mwa)-1)
        if mwa[i] > threshold and mwa[i] > mwa[i-1] and mwa[i] > mwa[i+1]
    ]

    beats = []
    r = 93

    for p in peaks:
        if p-r >= 0 and p+r < len(ECG_BUFFER):
            beat = ECG_BUFFER[p-r:p+r+1]
            if len(beat) == 187:
                beats.append(beat.tolist())

    return beats


async def run_model_on_beat(beat_187):
    # TODO: replace with real model call
    return int(0)  # class prediction placeholder


@app.websocket("/ws/esp32/ecg")
async def esp32_ecg_ws(websocket: WebSocket):
    await websocket.accept()
    print("✅ ESP32 ECG connected")

    temp_batch = []

    try:
        while True:
            message = await websocket.receive_bytes()

            # Must be exactly 4 bytes (float32)
            if len(message) != 4:
                continue

            # Decode little-endian float32
            (mV,) = struct.unpack("<f", message)
            temp_batch.append(mV)

            # Process + forward in small groups
            if len(temp_batch) >= BATCH_SIZE:
                samples = temp_batch[:]
                temp_batch = []

                # ✅ Forward raw samples to UI
                for c in ecg_clients:
                    try:
                        await c.send_json({"type": "ecg", "samples": samples})
                    except:
                        ecg_clients.remove(c)

                # ✅ R-peak → 187-sample beat → classify
                beats = detect_beats_and_extract_187(samples)
                for beat in beats:
                    pred = await run_model_on_beat(beat)
                    for c in ecg_clients:
                        try:
                            await c.send_json({
                                "type": "prediction",
                                "class": pred,
                                "beat": beat
                            })
                        except:
                            ecg_clients.remove(c)

    except WebSocketDisconnect:
        print("❌ ESP32 ECG disconnected")
    except Exception as e:
        print(f"❌ Error: {e}")

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

# ===========================
# ✅ REST Endpoints to get stored vitals
# ===========================

@app.get("/api/vitals")
def get_all_vitals():
    """Return all stored vitals from DB"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, heart_rate, spo2, temperature, timestamp FROM vitals ORDER BY id ASC")
        rows = cur.fetchall()
        cur.close()
        conn.close()

        data = []
        for row in rows:
            data.append({
                "id": row[0],
                "heart_rate": row[1],
                "spo2": row[2],
                "temperature": row[3],
                "timestamp": row[4].isoformat() if row[4] else None
            })

        return {"count": len(data), "vitals": data}

    except Exception as e:
        print(f"❌ Error fetching vitals: {e}")
        return {"error": "Failed to fetch vitals"}

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