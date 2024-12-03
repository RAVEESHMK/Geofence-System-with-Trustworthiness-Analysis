from flask import Flask, render_template, Response, jsonify
import numpy as np
import cv2
import os
import subprocess
import threading
import math
from gtts import gTTS
import time
import psutil

# Define your FFmpeg path and labels
ffmpeg_path = "./FFmpeg"
LABELS = open("./yolo-coco/coco.names").read().strip().split("\n")

# Load YOLO model
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet("./yolo-coco/yolov3.cfg", "./yolo-coco/yolov3.weights")
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Initialize webcam feed
cap = cv2.VideoCapture(0)
audio_playing = False

# Define a larger geofence area on the right side of the frame (adjusted to span a larger area)
frame_width, frame_height = 640, 480  # Assuming the frame size is 640x480

# Increase the width to make the geofence larger
geofence_width = 240  # Set the width to a larger value, like 240 pixels
geofence_height = frame_height  # Set the height to cover the entire window height
geofence_x1 = frame_width - geofence_width  # Align it to the right side
geofence_y1 = 0  # Start at the top
geofence_x2 = geofence_x1 + geofence_width  # End at the right edge of the frame
geofence_y2 = geofence_y1 + geofence_height  # Bottom edge of the geofence

# Object tracking dictionary to store previous states (inside or outside the geofence)
object_states = {}

# Geofence status and logs
geofence_status = "No objects in geofence"
movement_logs = []

# Benchmarking variables
start_time = time.time()
fps = 0

# Function to play audio alerts
def play_audio(description):
    global audio_playing
    audio_playing = True
    tts = gTTS(description, lang='en')
    tts.save("tts.mp3")
    subprocess.call([os.path.join(ffmpeg_path, "ffplay.exe"), "-nodisp", "-autoexit", "tts.mp3"])
    audio_playing = False

# Function to calculate distance between two points
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Track objects within the geofence
def check_geofence(centerX, centerY):
    if geofence_x1 < centerX < geofence_x2 and geofence_y1 < centerY < geofence_y2:
        return True  # Object is inside the geofence
    return False  # Object is outside the geofence

# Function to log benchmarking data
def log_benchmark():
    global start_time, fps
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = 1 / elapsed_time if elapsed_time > 0 else 0
    start_time = end_time
    print(f"FPS: {fps:.2f}")

# Function to add Gaussian noise to the frame to simulate adversarial attacks
def add_noise(frame):
    noise = np.random.normal(0, 25, frame.shape)  # Mean=0, Std=25
    noisy_frame = np.uint8(np.clip(frame + noise, 0, 255))  # Ensure values are within the valid range
    return noisy_frame

# Function to blur the face (use a face detection model)
def blur_faces(frame):
    # Load pre-trained Haar-cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Blur the faces detected in the frame
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
        frame[y:y+h, x:x+w] = blurred_face  # Replace the face area with the blurred version

    return frame

# Function to generate video frames
def generate_frames():
    global geofence_status, movement_logs, fps
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (frame_width, frame_height))

        # Step 1: Blur faces before adding adversarial noise
        frame = blur_faces(frame)

        # Step 2: Add noise to the frame to simulate adversarial attack
        frame = add_noise(frame)

        # Create a geofence boundary (rectangle) on the right side of the frame
        cv2.rectangle(frame, (geofence_x1, geofence_y1), (geofence_x2, geofence_y2), (255, 0, 0), 2)  # Blue rectangle

        # Prepare the image for YOLO detection
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        boxes, confidences, classIDs = [], [], []
        current_objects = set()

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > 0.5:
                    box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y, width, height) = boxes[i]
                label = f"{LABELS[classIDs[i]]}: {confidences[i]:.2f}"
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Track the object state (whether inside or outside geofence)
                inside_geofence = check_geofence(centerX, centerY)
                object_key = LABELS[classIDs[i]] + f"_{i}"  # Unique key for each object

                if object_key not in object_states:
                    object_states[object_key] = inside_geofence  # Initialize state

                # Detect entering or exiting
                if inside_geofence and not object_states[object_key]:
                    print(f"Object {LABELS[classIDs[i]]} has entered the geofence!")
                    movement_logs.append(f"Object {LABELS[classIDs[i]]} has entered the geofence.")
                    geofence_status = f"Object {LABELS[classIDs[i]]} inside geofence"
                    if not audio_playing:
                        description = f"{LABELS[classIDs[i]]} has entered the geofence."
                        threading.Thread(target=play_audio, args=(description,)).start()

                if not inside_geofence and object_states[object_key]:
                    print(f"Object {LABELS[classIDs[i]]} has exited the geofence!")
                    movement_logs.append(f"Object {LABELS[classIDs[i]]} has exited the geofence.")
                    geofence_status = "No objects in geofence"
                    if not audio_playing:
                        description = f"{LABELS[classIDs[i]]} has exited the geofence."
                        threading.Thread(target=play_audio, args=(description,)).start()

                # Update object state
                object_states[object_key] = inside_geofence

        # Log performance
        log_benchmark()

        # Display the frame with geofence
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Flask app to serve video feed and status
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/geofence_status')
def geofence_status_route():
    return jsonify({
        'status': geofence_status,
        'logs': movement_logs,
        'fps': fps
    })

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
