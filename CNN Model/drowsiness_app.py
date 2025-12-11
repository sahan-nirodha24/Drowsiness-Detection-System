import os
import cv2
import numpy as np
import tensorflow as tf
import threading
import time
import sys

# Audio Libraries
try:
    import winsound
except ImportError:
    winsound = None

try:
    import pyttsx3
    voice_engine = pyttsx3.init()
except ImportError:
    voice_engine = None
    print("Warning: 'pyttsx3' not found. Voice alerts disabled. Install with `pip install pyttsx3`.")

# --- Mute TensorFlow Logs ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Configuration ---
MODEL_PATH = "Models/drowsiness_model_mobilenet.h5"
LABELS = ["Drowsy", "Non-Drowsy"] 
IMG_SIZE = (224, 224)

# Thresholds & Parameters
DROWSY_THRESHOLD = 0.5 
EYE_CLOSED_FRAMES_THRESH = 10  # Consecutive frames with 0 eyes to trigger alert
WARMUP_DURATION = 5.0          # Seconds to wait before alerting
ALERT_COOLDOWN = 3.0           # Seconds between alerts

# --- State Management ---
alert_active = False 
last_alert_time = 0 
start_time = time.time()
closed_frames = 0
invert_logic = False 

def play_alert_thread(message, beep_enabled=True):
    """
    Handles audio alerts in a separate thread.
    """
    global alert_active
    if alert_active: return
    alert_active = True
    
    try:
        # 1. Beep Sound
        if beep_enabled and winsound:
            winsound.Beep(2500, 1000) 
        
        # 2. Voice Command
        if voice_engine:
            voice_engine.say(message)
            voice_engine.runAndWait()

    except Exception as e:
        print(f"Audio Error: {e}")
    finally:
        alert_active = False

def trigger_alert(message, is_serious):
    global last_alert_time
    current_time = time.time()
    
    # Only alert if warmup is complete
    if (current_time - start_time > WARMUP_DURATION):
        # Check cooldown
        if (current_time - last_alert_time > ALERT_COOLDOWN):
            last_alert_time = current_time
            threading.Thread(target=play_alert_thread, args=(message, is_serious), daemon=True).start()

def main():
    global invert_logic, closed_frames

    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    print("Loading CNN Model...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model Loaded Successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 2. Setup Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # 3. Detectors
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    print("--- CNN-Based Drowsiness Detection with Hybrid Logic ---")
    print("Press 'q' to Quit | 'i' to Invert Logic")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1) # Mirror
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Face Detection
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # --- Mechanism 1: Geometric Eye Detection ---
            roi_gray = gray[y:y+h, x:x+w]
            # Focus on upper half for eyes usually, but full face is safer for general cascade
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4, minSize=(20, 20))
            
            if len(eyes) == 0:
                closed_frames += 1
            else:
                closed_frames = 0
            
            # Draw Eyes
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 1)

            # --- Mechanism 2: CNN Inference ---
            cnn_status = "Unknown"
            is_cnn_drowsy = False
            
            try:
                # Preprocess for MobileNet
                face_img = frame[y:y+h, x:x+w]
                face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(face_img_rgb, IMG_SIZE)
                normalized = tf.keras.applications.mobilenet_v2.preprocess_input(resized.astype('float32'))
                reshaped = np.reshape(normalized, (1, 224, 224, 3))

                prediction = model.predict(reshaped, verbose=0)
                raw_drowsy_score = prediction[0][0]
                raw_non_drowsy_score = prediction[0][1]
                
                # Check Logic
                is_cnn_drowsy = raw_drowsy_score > raw_non_drowsy_score
                if invert_logic: is_cnn_drowsy = not is_cnn_drowsy
                
                score = raw_non_drowsy_score if (invert_logic != is_cnn_drowsy) else raw_drowsy_score # Display logic score
                
                if is_cnn_drowsy:
                    cnn_status = f"DROWSY ({raw_drowsy_score:.2f})"
                else:
                    cnn_status = f"Active ({raw_non_drowsy_score:.2f})"

            except Exception as e:
                print(f"CNN Error: {e}")

            # --- Consolidated Alert Logic ---
            final_status = "Scanning..."
            alert_color = (0, 255, 0)
            
            # Case A: Eyes Closed (Geometric) - Highest Priority
            if closed_frames > EYE_CLOSED_FRAMES_THRESH:
                final_status = "EYES CLOSED!"
                alert_color = (0, 0, 255)
                trigger_alert("Driver Alert! Your eyes are closed.", is_serious=True)
            
            # Case B: CNN Drowsy (Model)
            elif is_cnn_drowsy:
                final_status = f"CNN: {cnn_status}"
                alert_color = (0, 0, 255)
                trigger_alert("Wake up! You are drowsy.", is_serious=True)
                
            # Case C: Active
            else:
                final_status = "Active"
                alert_color = (0, 255, 0)
                # Optional: "Your eyes are open" every few seconds?
                # Using the generic trigger logic, we can do it if needed, 
                # but to match the snippet's "Active" flow, we'll keep it silent or minimal.
                # Actually, user liked "Your eyes are open".
                # Let's add it with valid cooldown.
                trigger_alert("Driver Alert! Your eyes are open.", is_serious=False)

            # Debug Overlay
            cv2.putText(frame, final_status, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)
            
            # System Info
            cv2.putText(frame, f"Closed Frames: {closed_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(frame, f"Model: {cnn_status}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            # Warmup Timer
            current_time = time.time()
            if current_time - start_time < WARMUP_DURATION:
                 cv2.putText(frame, f"WARMUP: {int(WARMUP_DURATION - (current_time - start_time))}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('Drowsiness Detector (CNN Hybrid)', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('i'): 
            invert_logic = not invert_logic
            print(f"Logic Inverted: {invert_logic}")

    cap.release()
    cv2.destroyAllWindows()
    print("System Shutdown.")

if __name__ == "__main__":
    main()
