# 🛡 AI-Based Wildlife Poaching Detection System

## 📖 Abstract

### 🐾 Problem Statement
Wildlife poaching remains a critical threat to biodiversity, especially in remote forest regions where continuous human monitoring is unfeasible. Conventional surveillance systems are either reactive, energy-intensive, or incapable of differentiating between poachers, wildlife, and vehicles. These limitations hinder rapid intervention and compromise wildlife protection efforts.


### 💡 Proposed Solution
We propose a low-power, real-time AI-based poaching detection system designed for remote forest surveillance. The solution uses a trained object detection model to analyze video frames and identify the presence of poachers, animals, or vehicles. Upon detecting a poacher, the system sends an immediate SMS alert to forest authorities using a GSM module. The device is motion-triggered to optimize energy usage and is built with solar-powered hardware for continuous operation in off-grid environments.


## ⚙ System Features

- 🎯 *AI-Powered Object Detection*: Detects poachers, animals, and vehicles using deep learning.
- 🔔 *Real-Time Alerts*: Sends SMS notifications when poachers are detected.
- ⚡ *Motion-Activated Inference*: PIR sensor triggers image processing only when movement is detected.
- ☀ *Solar Powered Hardware*: Supports 24/7 deployment in remote areas.
- 🌧 *Weatherproof Enclosure*: Rugged design suitable for forest conditions.

---

## 🧰 Tech Stack

### 🎓 AI/ML:
- *Model Architecture*: YOLOv5 Nano / YOLOv8n
- *Frameworks*: PyTorch, TensorFlow Lite
- *Data Annotation*: LabelImg, Roboflow
- *Preprocessing*: OpenCV (for frame extraction and manipulation)

### 💻 Edge & Hardware:
- *Primary Compute Device*: NVIDIA Jetson Nano / Raspberry Pi 4 + Coral USB Accelerator
- *Microcontroller*: ESP32 (for motion sensing and GSM control)
- *Sensors*: 
  - PIR Motion Sensor (HC-SR501)
  - Optional GPS Module (for location tagging)
- *Communication Module*: SIM800L GSM Module
- *Power Supply*: Li-Ion Batteries + Solar Panel + MPPT Charge Controller

### 🧑‍💻 Programming Languages:
- Python (AI + Control Logic)
- C/C++ (Microcontroller Firmware)
- Arduino IDE (Firmware Development)

---

## 🚀 Workflow

1. *Training Phase*
   - Collect and annotate images of forests, poachers, animals, and vehicles.
   - Train a lightweight object detection model suitable for edge inference.

2. *Deployment Phase*
   - Live video is captured and split into frames.
   - AI model runs inference on each frame.
   - If a poacher is detected, ESP32 triggers SMS alert via GSM module.

3. *Optimization*
   - Motion sensors reduce energy use by triggering inference only on movement.
   - Solar-powered system ensures sustainability in remote locations.

---

## 🌿 Future Scope

- 🔭 Integration of infrared/thermal vision for night surveillance
- 🛰 Satellite or LoRaWAN communication for ultra-remote monitoring
- 📊 Cloud-based dashboard for data logging and real-time visual monitoring
- 🎥 Live video streaming with edge buffering

---

## 👨‍🔬 Team Vision

This project embodies our passion for combining *Artificial Intelligence, **Embedded Systems, and **Environmental Conservation*. By deploying smart systems in the wild, we aim to create meaningful impact and contribute to the preservation of our planet's biodiversity.

---

> "Technology, when directed with purpose, can become a guardian of the wild."
