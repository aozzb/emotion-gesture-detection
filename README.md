# Real-Time Face Expression Detection (OpenCV + MediaPipe)

A lightweight real-time facial expression detector built using **OpenCV** and **MediaPipe Face Mesh**.  
The system tracks key facial landmarks, computes geometric ratios, smooths them over time, and classifies expressions such as:

- 😊 **Smile**
- 😮 **Mouth Open**
- 😴 **Eyes Closed**
- 😐 **Neutral**

This project is part of a bigger emotion + gesture → emote GIF mapping system (in progress).

---

## 🚀 Features

### ✔ Face Detection
- Uses **MediaPipe FaceDetection**  
- Returns bounding box `(x, y, w, h)`  
- Optimized for webcam-range faces

### ✔ Facial Landmark Extraction
- Uses **MediaPipe FaceMesh** (468 landmarks)
- Extracts key points around:
  - Mouth
  - Eyes
  - Lips
  - Smile curvature

### ✔ Expression Classification
Based on geometric thresholds & smoothing:
- `smile`
- `mouth_open`
- `eyes_closed`
- `neutral`

### ✔ Real-Time Pipeline
- Captures frames from webcam
- Detects face
- Computes expression
- Draws bounding box & labels
- Displays **FPS**

---

## 🧠 How Expression Detection Works

| Metric | Description |
|--------|-------------|
| **Mouth Open Ratio** | lip distance ÷ mouth width |
| **Smile Ratio** | mouth width ÷ lip height |
| **Smile Curvature** | vertical raise of mouth corners |
| **EAR (Eye Aspect Ratio)** | eye openness measure |

All metrics are smoothed using a **moving average** to avoid jitter.

---

## 📂 Project Structure

project/
└── src/
    ├── face_detector.py       # Face detection using MediaPipe
    ├── face_expression.py     # Expression analysis using Face Mesh
    └── main.py                # Real-time webcam loop

---

## 🛠 Installation

1️⃣ Clone or Download the Repository
    git clone https://github.com/aozzb/emotion-gesture-detection.git
    cd emotion-gesture-detection/src
2️⃣ Install Dependencies
    pip install opencv-python mediapipe numpy

---

## ▶️ Running the Project

Inside the src directory:
    python main.py
This will open your webcam and show:
    Bounding box
    Expression text
    FPS value

---

## 📈 Output Overview

The display includes:
  A green face bounding box
  Expression label:
      "smile"
      "mouth_open"
      "eyes_closed"
      "neutral"
  FPS counter
  Console prints detected expression each frame

---

## 🔧 Adjustable Thresholds

You can modify these lines in FaceExpression.get_expression():

  if ear < 5:
      return "eyes_closed"
  elif mouth_open_ratio > 0.55:
      return "mouth_open"
  elif smile_ratio > 2.8 and self._smile_curvature(pts) > 5:
      return "smile"
  else:
      return "neutral"

---

## 🤝 Built With

  Python
  OpenCV
  MediaPipe Face Detection
  MediaPipe Face Mesh
  NumPy

---




