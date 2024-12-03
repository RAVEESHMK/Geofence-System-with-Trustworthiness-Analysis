# Geofence-System-with-Trustworthiness-Analysis

Overview

This project implements a real-time object detection and geofence monitoring system using the YOLO model. It also evaluates and analyzes the system's trustworthiness by simulating various attacks, assessing potential security and privacy risks, and benchmarking the system for fairness and bias issues.

Features

1. Object Detection: Detects objects in real-time using the YOLO model.
2. Geofence Monitoring: Monitors object movement inside a designated geofence.
3. Privacy Protection: Blurs faces in the video feed to protect privacy.
4. Security Evaluation: Simulates adversarial attacks (e.g., Gaussian noise) to test robustness.
5. Fairness Assessment: Analyzes detection performance across object classes to identify bias.
6. Performance Benchmarking: Logs frame rates (FPS) and system efficiency.

Technologies Used

Backend: Flask, OpenCV, YOLO (Darknet), gTTS
Frontend: HTML, CSS, JavaScript, Bootstrap
Other: FFmpeg for audio playback

.
├── trust_worthy_ai.py                # Flask backend
├── templates/
│   └── index.html        # Frontend HTML
├── yolo-coco/            # YOLO model configuration
│   ├── coco.names        # Class labels
│   ├── yolov3.cfg        # YOLO configuration
│   └── yolov3.weights    # Pre-trained YOLO weights
├── ffmpeg
    ├── ffplay
└── README.md             # Documentation

Installation

Prerequisites:

Python 3.8+
Pip package manager
FFmpeg installed on your system
YOLOv3 model files (yolov3.cfg, yolov3.weights, coco.names) in the yolo-coco/ directory.

Steps:
1. Clone the repository
2. Install dependencies
3. Download YOLO files
4. run python trust_worty_ai.py
5. Access the application: Open your browser and navigate to http://127.0.0.1:5000/.

Usage

View the live video feed and monitor the geofence status on the web dashboard.
Geofence alerts will be displayed, and corresponding audio notifications will play.
Logs of object movements are displayed on the dashboard for reference.

Contact
For inquiries, please contact rm159@usf.edu
