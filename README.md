Overview

Healthify is an IoT-based real-time health monitoring system that leverages biomedical sensors, machine learning, and web dashboard integration to provide continuous health tracking and intelligent health condition detection.

Objective:

Real-Time Health Monitoring:
Capture SpO₂, BPM (Heart Rate), Body Temperature, and ECG signals using biomedical sensors, streamed continuously for live analysis.

Live Data Display:
Visualize current sensor readings on an SSD1306 OLED display for instant user feedback.

Machine Learning–Based Detection:
Trained ML models analyze physiological data to detect:
Hypoxemia

Arrhythmia

Stress Levels

Fever

Tachycardia / Bradycardia

Web Dashboard Integration:
Display real-time vitals and ML-based results through an interactive, user-friendly dashboard for remote monitoring.

Technology Stack:
Microcontroller: ESP32

Sensors:

MAX30100 Pulse Oximeter (SpO₂, BPM)

AD8232 ECG Sensor

DS18B20 Temperature Sensor

Display: SSD1306 OLED

Software:
Arduino IDE (for embedded programming)

Python / Flask (for data processing and backend)

HTML, CSS, JavaScript (for dashboard frontend)

Machine Learning:
Trained models for vital sign classification and anomaly detection.

Hardware Components: Component -> 	Function
ESP32	Core microcontroller with Wi-Fi connectivity
MAX30100	Measures SpO₂ and pulse rate
AD8232	Captures ECG signals
DS18B20	Reads body temperature
SSD1306 OLED	Displays live vitals
Breadboard & Jumper Wires	Used for prototyping connections
Button & USB Cable	Used for user input and data/power interface

System Architecture:
Sensor Data Acquisition: Sensors capture physiological parameters.

Data Transmission: ESP32 streams data to the web server via Wi-Fi.

Real-Time Visualization: OLED and web dashboard display live vitals.

ML Processing: Collected data is fed to trained models for condition detection.

Alert System: Generates emergency alerts for abnormal conditions.

Applications:
Remote patient monitoring with real-time vitals

Early detection of health anomalies using machine learning

Fitness and stress tracking for daily wellness insights

Emergency alert system for critical conditions

Telemedicine and live doctor–patient data sharing
