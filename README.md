# Healthify - Health Monitor Dashboard

Real-time health monitoring dashboard displaying vital signs from smartwatch sensors.

## Features

- Real-time vital monitoring (heart rate, SpO2, temperature, ECG)
- Interactive dashboard with charts and status indicators
- ML-powered stress prediction
- WebSocket communication for live data
- PostgreSQL data persistence
- Connection status monitoring
- Health status alerts

## Quick Start

### Prerequisites
- Node.js (v16+)
- Python 3.8+
- PostgreSQL database

### Installation

1. **Backend Setup**
   ```bash
   cd server
   pip install -r requirements.txt
   python database.py  # Initialize database
   python main.py      # Start backend server
   ```

2. **Frontend Setup**
   ```bash
   npm install
   npm run dev          # Start development server
   ```

3. **Hardware** (Optional)
   - Upload `hardware/healthify_driver.ino` to ESP32
   - Configure WiFi credentials
   - Connect sensors

## API Endpoints

- `GET /` - Health check
- `GET /api/v1/vitals` - Latest vitals
- `ws://localhost:8000/ws/esp32/vitals` - ESP32 connection
- `ws://localhost:8000/ws/client/vitals` - Web client connection

## Health Thresholds

- **Heart Rate**: Normal 60-100 BPM
- **SpO2**: Normal ≥95%
- **Temperature**: Normal 36.1-37.2°C

## Usage

1. Start backend and frontend servers
2. Open `http://localhost:3000`
3. Connect ESP32 device (optional)
4. Monitor real-time vitals

## Project Structure

```
Healthify/
├── App.tsx              # React frontend
├── server/              # FastAPI backend
│   ├── main.py         # API server
│   ├── database.py     # Database setup
│   └── requirements.txt
└── hardware/           # ESP32 Arduino code
    └── healthify_driver.ino
```

## Tech Stack

- Frontend: React, TypeScript, Tailwind CSS
- Backend: FastAPI, Python, PostgreSQL
- Communication: WebSocket
- Hardware: ESP32 with sensors

---

**Note**: Educational project. Additional security and validation required for production medical use.