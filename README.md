# DEMI: Deodorizing Engineering Majors Intelligently 🧴🤖

> *A fully functional robotics system built at Cal Hacks 12.0 to keep Berkeley engineers clean, confident, and computationally fresh.*

[![Devpost](https://img.shields.io/badge/Devpost-DEMI-blue?logo=devpost)](https://devpost.com/software/demi-deodorizing-engineering-majors-intelligently)
[![Hackathon](https://img.shields.io/badge/Built%20at-Cal%20Hacks%2012.0-%230066cc)](https://calhacks.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🧠 Overview

**DEMI** (*Deodorizing Engineering Majors Intelligently*) is a backpack-mounted personal hygiene robot that detects body odor and automatically applies deodorant — either to you or toward the source of the smell around you.

Built in 36 hours at **Cal Hacks 12.0 (UC Berkeley)**, DEMI combines:
- **Environmental sensing** via a TVOC gas sensor  
- **Computer vision & depth estimation** for spatial awareness  
- **Multithreaded embedded processing** on a Raspberry Pi  
- **6-DOF robotic arm control** via LeRobot’s SDK  

What began as a lighthearted idea — “a robot that sprays deodorant when things get rough” — became a fully working, sensor-driven robotics platform that blends embedded systems, AI perception, and robotic actuation.

---

## 💡 Inspiration

From our first day in Cory Hall, we knew Berkeley engineering life would be a grind — late nights, impossible exams, and caffeine replacing sleep. Personal hygiene? Sometimes… optional.  
So instead of continuing that cycle, we decided to *automate it*.

DEMI was born: **Deodorizing Engineering Majors Intelligently**. What started as a joke evolved into a serious engineering challenge in sensor fusion, embedded software, and real-time robotics.

---

## 🧩 What It Does

DEMI keeps users (and their peers) hygienic through two intelligent behaviors:

### 🫧 1. Self-Application of Deodorant  
A **TVOC (Total Volatile Organic Compounds)** sensor sits near the user’s body and monitors air quality. When VOC concentrations surpass a threshold — correlating with body odor — DEMI activates a **self-application trajectory** on its robotic arm, which smoothly moves the deodorant can to the user and sprays.

### 🌫️ 2. Maintaining a “Sphere of Cleanliness”  
A **front-mounted webcam** (on the user’s shirt) continuously detects people using CNN-based vision models.  
If nearby individuals are detected *and* ambient VOC levels are high, DEMI determines the odor’s likely direction, selects an appropriate **directional spray trajectory**, and deploys deodorant toward the source — keeping the local air quality (and social interactions) fresh.

---

## 🛠️ How We Built It

DEMI integrates **three major computing paradigms**: computer vision, embedded systems, and robotics control.

### 👁️ Computer Vision
- A **Logitech webcam** captures live RGB frames in real time.  
- Using **OpenCV** for preprocessing and **MobileNetV3** for object detection, DEMI identifies humans in its field of view.  
- To infer distance, DEMI employs **Intel’s DPT-SwinV2-Tiny-256** depth estimation model, producing per-pixel depth maps from single RGB images.  
- Combining bounding boxes with depth data gives precise **3D localization** of nearby people, allowing targeted deodorant trajectories based on distance and orientation.

### 💻 Embedded Systems
- A **Raspberry Pi 4B** runs all sensor and vision logic concurrently using multithreading.  
- **Threads:**
  - `sensor_thread`: Reads and filters TVOC data using an *n-point moving average low-pass filter (LPF)*, calibrated through an affine sensor model for stable odor detection.
  - `camera_thread`: Continuously streams and preprocesses camera frames.
  - `comm_thread`: Maintains an **SSH connection (via Paramiko)** to a remote MacBook server.
- During actuation, both sensor and camera threads **pause automatically** while the arm executes a spray trajectory, then resume when a *“trajectory complete”* flag is received.
- All data (sensor, depth, detections) are serialized into **JSON packets** and transmitted for real-time fusion and logging.

### 🤖 Robotics
- DEMI’s actuator is a **LeRobot S0-101 6-DOF robotic arm** controlled through **LeRobot’s Python SDK**.
- The team recorded multiple **joint-space trajectories** for different spray actions:
  - Self-application  
  - Directional spray (left, right, forward)
- The **MacBook server** receives processed odor and camera data from the Pi and decides which trajectory to execute.
- This architecture cleanly separates **low-level sensing** (on Pi) from **high-level actuation** (on server), enabling robust real-time control and safe concurrency.

---

## 🧰 Hardware Components

| Component | Function |
|------------|-----------|
| **Raspberry Pi 4B** | Embedded processing, multithreaded sensing & comms |
| **TVOC Sensor** | Odor detection and air-quality monitoring |
| **Webcam (Logitech)** | Frontal camera for detection & depth estimation |
| **MacBook (Server)** | High-level control & trajectory execution |
| **LeRobot S0-101 Arm** | 6-DOF robotic actuator for deodorant application |
| **Deodorant Can & Mount** | Spray payload |
| **Backpack Rig** | Mounting for portable integration |

---

## 🧱 Software Stack

| Layer | Tools / Libraries |
|--------|------------------|
| **Language** | Python 3.8+ |
| **Computer Vision** | OpenCV, PyTorch, torchvision, MobileNetV3 |
| **Depth Estimation** | Intel DPT-SwinV2-Tiny-256 |
| **Embedded I/O** | `rpi.gpio`, I²C interface |
| **Networking** | Paramiko (SSH + JSON packets) |
| **Robotics SDK** | LeRobot (teleoperative control + trajectory playback) |
| **Signal Processing** | NumPy-based affine calibration + LPF filtering |
| **Threading** | `threading`, `asyncio` |
| **Logging** | Real-time JSON serialization and remote logging |
