import tensorflow as tf
import cv2
import numpy as np
import collections
import os
import csv
from datetime import datetime

# --- Main Configuration ---
# --- Motion Extraction Settings ---
MOTION_DELAY_SECONDS = 0.1 

# --- Edge Detection & Model Settings ---
# Path to your newly trained student model.
STUDENT_MODEL_PATH = "delta_trace_student_model.h5"
# The input size your model was trained on.
IMG_WIDTH = 224
IMG_HEIGHT = 224
# Canny edge detector thresholds.
CANNY_THRESHOLD_1 = 30
CANNY_THRESHOLD_2 = 150

# --- Optimization & Debugging ---
MOTION_SENSITIVITY = 5
MIN_MOTION_AREA = 50 * 50 
DEBUG_VIEW = False 

# --- File Paths ---
# Corrected for Windows compatibility
LOG_SAVE_PATH = r"C:\Users\DAS_3\Desktop\project"
# --- End Configuration ---


def run_delta_trace():
    """
    Main application to run the final DELTA TRACE system with the trained student model.
    """
    # --- Load Your Trained Student Model ---
    print(f"[INFO] Loading trained student model from '{STUDENT_MODEL_PATH}'...")
    if not os.path.exists(STUDENT_MODEL_PATH):
        print(f"[ERROR] Student model file not found. Please ensure '{STUDENT_MODEL_PATH}' is in the same folder as this script.")
        return
    try:
        student_model = tf.keras.models.load_model(STUDENT_MODEL_PATH)
        print("[INFO] Student model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    # --- Video Capture Setup ---
    print("[INFO] Starting video capture...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open video stream.")
        return

    # --- Motion Extraction Setup ---
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    delay_frames = int(fps * MOTION_DELAY_SECONDS)
    if delay_frames < 1: delay_frames = 1
    frame_buffer = collections.deque(maxlen=delay_frames)

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_pixels = original_width * original_height
    
    # --- Data Logging Setup ---
    log_data = []
    log_headers = ['Timestamp', 'Frame', 'Pixels Processed', 'Total Pixels', 'Saving Percentage', 'Motion Sensitivity']
    
    # --- Window Setup ---
    WINDOW_NAME = 'DELTA TRACE'
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print(f"[INFO] Running on {original_width}x{original_height}. Press 'q' to quit.")
    frame_count = 0

    # --- Main Loop ---
    while True:
        ret, current_frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_buffer.append(current_frame)
        output_frame = current_frame.copy()
        pixels_processed = 0 

        if len(frame_buffer) == delay_frames:
            delayed_frame = frame_buffer[0]
            
            # --- Step 1: Create the Filtered Frame for Display ---
            inverted_delayed_frame = cv2.bitwise_not(delayed_frame)
            filtered_frame = cv2.addWeighted(
                src1=current_frame, alpha=1.0, src2=inverted_delayed_frame,
                beta=0.99, gamma=-225.0
            )
            output_frame = filtered_frame.copy() # The final output will be the filtered view

            # --- Step 2: Isolate Motion ---
            frame_delta = cv2.absdiff(current_frame, delayed_frame)
            gray_delta = cv2.cvtColor(frame_delta, cv2.COLOR_BGR2GRAY)
            _, thresh_mask = cv2.threshold(gray_delta, MOTION_SENSITIVITY, 255, cv2.THRESH_BINARY)
            
            contours, _ = cv2.findContours(thresh_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion_contours = [c for c in contours if cv2.contourArea(c) > MIN_MOTION_AREA]

            # --- Step 3: Use Student Model on Detected Motion ---
            if motion_contours:
                motion_mask = np.zeros_like(thresh_mask)
                cv2.drawContours(motion_mask, motion_contours, -1, (255), cv2.FILLED)
                pixels_processed = cv2.countNonZero(motion_mask)
                
                # Create the processing frame from the filtered image, as this is what the model was trained on
                processing_frame = cv2.bitwise_and(filtered_frame, filtered_frame, mask=motion_mask)
                
                # Prepare the frame for the model (resize, normalize, add batch dimension)
                model_input = cv2.resize(processing_frame, (IMG_WIDTH, IMG_HEIGHT))
                model_input = model_input / 255.0
                model_input = np.expand_dims(model_input, axis=0)

                # Get prediction from your trained model
                prediction = student_model.predict(model_input, verbose=0)[0]
                
                # De-normalize the prediction to get pixel coordinates
                x_center = int(prediction[0] * original_width)
                y_center = int(prediction[1] * original_height)
                box_w = int(prediction[2] * original_width)
                box_h = int(prediction[3] * original_height)
                
                startX = int(x_center - (box_w / 2))
                startY = int(y_center - (box_h / 2))
                endX = startX + box_w
                endY = startY + box_h
                
                # --- Step 4: Perform Precise Edge Trace ---
                # Ensure coordinates are within the frame bounds before slicing
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(original_width - 1, endX), min(original_height - 1, endY))

                human_roi = current_frame[startY:endY, startX:endX]
                if human_roi.size > 0:
                    gray_roi = cv2.cvtColor(human_roi, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray_roi, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)
                    edge_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for c in edge_contours:
                        c += (startX, startY) # Offset back to full screen coordinates
                    
                    cv2.drawContours(output_frame, edge_contours, -1, (0, 255, 255), 2) # Yellow outline

        # --- Step 5: Log Data and Display Metrics ---
        saving_percentage = 100 * (1 - (pixels_processed / total_pixels)) if total_pixels > 0 else 0
        metrics_text = f"Processed: {pixels_processed:,} / {total_pixels:,} pixels ({saving_percentage:.1f}% saving)"
        log_data.append([datetime.now(), frame_count, pixels_processed, total_pixels, saving_percentage, MOTION_SENSITIVITY])
        
        cv2.putText(output_frame, metrics_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(output_frame, metrics_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow(WINDOW_NAME, output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup & Save Log ---
    print("[INFO] Stopping stream and cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    
    if log_data:
        os.makedirs(LOG_SAVE_PATH, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"delta_trace_inference_log_{timestamp_str}.csv"
        full_path = os.path.join(LOG_SAVE_PATH, filename)
        
        print(f"[INFO] Saving performance data to {full_path}...")
        with open(full_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(log_headers)
            writer.writerows(log_data)
        print("[INFO] Save complete.")
    
    print("[INFO] Done.")


if __name__ == '__main__':
    run_delta_trace()
