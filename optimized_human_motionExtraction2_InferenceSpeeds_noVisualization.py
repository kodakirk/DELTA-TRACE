import cv2
import numpy as np
import collections
import os
import csv
from datetime import datetime
import glob
import time

# --- Main Configuration ---
# --- Motion Extraction Settings ---
MOTION_DELAY_SECONDS = 0.1 
BETA = 0.99
GAMMA = -200.0

# --- Human Detection Settings (DNN) ---
CONFIDENCE_THRESHOLD = 0.4

# --- Optimization & Debugging ---
MOTION_SENSITIVITY = 0.1
MIN_MOTION_AREA = 50 * 50 
DEBUG_VIEW = False 

# --- File Paths ---
VIDEO_SOURCE_FOLDER = "/home/koda/Desktop/clips"
LOG_SAVE_PATH = "/home/koda/Desktop/motionExtraction"
# Paths for the MobileNet-SSD model files.
HOME_DIR = os.path.expanduser("~")
prototxt_path = os.path.join(HOME_DIR, "Desktop/test", "deploy.prototxt")
model_path = os.path.join(HOME_DIR, "Desktop/test", "mobilenet_iter_73000.caffemodel")
# --- End Configuration ---


def analyze_videos(video_paths, net, analysis_type="Optimized"):
    """
    Analyzes a list of videos using a specified method ('Optimized' or 'Full_Frame'),
    without displaying the video for maximum processing speed.
    Returns a list of log entries and a dictionary of processing times.
    """
    log_data = []
    processing_times = {}
    
    # MODIFIED: Window creation is disabled for headless operation.
    # WINDOW_NAME = f'DELTA TRACE - {analysis_type} Analysis'
    # cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    for video_path in video_paths:
        print(f"\n[INFO] [{analysis_type}] Processing video: {os.path.basename(video_path)}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[WARNING] Could not open video file: {video_path}. Skipping.")
            continue

        # --- Setup for this video ---
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30
        delay_frames = int(fps * MOTION_DELAY_SECONDS)
        if delay_frames < 1: delay_frames = 1
        frame_buffer = collections.deque(maxlen=delay_frames)

        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_pixels = original_width * original_height
        
        frame_count = 0
        start_time = time.time() # Start timer for this video

        # --- Frame-by-Frame Processing Loop ---
        while True:
            ret, current_frame = cap.read()
            if not ret: break

            frame_count += 1
            frame_buffer.append(current_frame)
            pixels_processed = 0 
            run_detector = False
            inference_time_ms = 0 # Initialize inference time for this frame

            if len(frame_buffer) == delay_frames:
                delayed_frame = frame_buffer[0]
                
                if analysis_type == "Optimized":
                    # Core motion detection logic remains.
                    frame_delta = cv2.absdiff(current_frame, delayed_frame)
                    gray_delta = cv2.cvtColor(frame_delta, cv2.COLOR_BGR2GRAY)
                    _, thresh_mask = cv2.threshold(gray_delta, MOTION_SENSITIVITY, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(thresh_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if any(cv2.contourArea(c) > MIN_MOTION_AREA for c in contours):
                        run_detector = True
                        motion_mask = np.zeros_like(thresh_mask)
                        cv2.drawContours(motion_mask, contours, -1, (255), cv2.FILLED)
                        pixels_processed = cv2.countNonZero(motion_mask)
                        processing_frame = cv2.bitwise_and(current_frame, current_frame, mask=motion_mask)
                
                elif analysis_type == "Full_Frame":
                    # Full_Frame mode processes every frame
                    run_detector = True
                    pixels_processed = total_pixels
                    processing_frame = current_frame

                # --- Run DNN if triggered ---
                if run_detector:
                    (h, w) = processing_frame.shape[:2]
                    blob = cv2.dnn.blobFromImage(cv2.resize(processing_frame, (300, 300)), 0.007843, (300, 300), 127.5)
                    net.setInput(blob)
                    
                    # Measure the inference time
                    inf_start = time.time()
                    detections = net.forward()
                    inf_end = time.time()
                    inference_time_ms = (inf_end - inf_start) * 1000

                    # Detection logic is still performed, but not drawn to a visible window.
                    for i in np.arange(0, detections.shape[2]):
                        confidence = detections[0, 0, i, 2]
                        if confidence > CONFIDENCE_THRESHOLD:
                            idx = int(detections[0, 0, i, 1])
                            if CLASSES[idx] == "person":
                                # The bounding box is found but not drawn to save time.
                                pass

            # --- Log Data ---
            saving_percentage = 100 * (1 - (pixels_processed / total_pixels)) if total_pixels > 0 else 0
            
            log_data.append([
                datetime.now(), os.path.basename(video_path), frame_count, 
                pixels_processed, total_pixels, saving_percentage, 
                MOTION_SENSITIVITY, MOTION_DELAY_SECONDS, CONFIDENCE_THRESHOLD, 
                MIN_MOTION_AREA, BETA, GAMMA, analysis_type, inference_time_ms
            ])
            
            # MODIFIED: All visualization calls are disabled.
            # cv2.imshow(WINDOW_NAME, output_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        end_time = time.time()
        processing_times[os.path.basename(video_path)] = end_time - start_time
        cap.release()
    
    # cv2.destroyAllWindows() # Disabled as no windows are created.
    return log_data, processing_times


def main():
    """
    Main function to run the comparative analysis and save the results.
    """
    # --- Load Model ---
    print("[INFO] Initializing DNN person detector (MobileNet-SSD)...")
    if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
        print(f"[ERROR] Model files not found. Please check paths.")
        return
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    
    # --- Find Videos ---
    video_paths = glob.glob(os.path.join(VIDEO_SOURCE_FOLDER, '*'))
    if not video_paths:
        print(f"[ERROR] No video files found in '{VIDEO_SOURCE_FOLDER}'.")
        return

    # --- Run Both Analyses ---
    optimized_logs, optimized_times = analyze_videos(video_paths, net, "Optimized")
    full_frame_logs, full_frame_times = analyze_videos(video_paths, net, "Full_Frame")

    # --- Combine and Save Results ---
    all_logs = optimized_logs + full_frame_logs
    if not all_logs:
        print("[INFO] No data was logged. Exiting.")
        return

    os.makedirs(LOG_SAVE_PATH, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"comparative_analysis_log_{timestamp_str}.csv"
    full_path = os.path.join(LOG_SAVE_PATH, filename)
    
    print(f"\n[INFO] Saving combined analysis data to {full_path}...")
    
    log_headers = [
        'Timestamp', 'Video File', 'Frame', 'Pixels Processed', 'Total Pixels', 
        'Saving Percentage', 'Motion Sensitivity', 'Motion Delay', 
        'Confidence Threshold', 'Min Motion Area', 'Beta', 'Gamma', 'Analysis Type',
        'Inference Time (ms)'
    ]

    with open(full_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(log_headers)
        writer.writerows(all_logs)

        # --- Append Summary Section ---
        writer.writerow([])
        writer.writerow(["--- Processing Time Summary (seconds) ---"])
        writer.writerow(["Video File", "Optimized Time", "Full Frame Time", "Time Saved (s)"])
        
        for video_name in optimized_times:
            opt_time = optimized_times.get(video_name, 0)
            ff_time = full_frame_times.get(video_name, 0)
            time_saved = ff_time - opt_time if ff_time > 0 else 0
            writer.writerow([video_name, f"{opt_time:.2f}", f"{ff_time:.2f}", f"{time_saved:.2f}"])
            print(f"  - {video_name}: Optimized={opt_time:.2f}s, Full Frame={ff_time:.2f}s")

        # --- Append Inference Time Summary ---
        opt_inf_times = [row[-1] for row in optimized_logs if row[-1] > 0]
        ff_inf_times = [row[-1] for row in full_frame_logs if row[-1] > 0]
        
        avg_opt_inf = sum(opt_inf_times) / len(opt_inf_times) if opt_inf_times else 0
        avg_ff_inf = sum(ff_inf_times) / len(ff_inf_times) if ff_inf_times else 0

        writer.writerow([])
        writer.writerow(["--- Average Inference Time per Frame (ms) ---"])
        writer.writerow(["Analysis Type", "Average Inference Time (ms)"])
        writer.writerow(["Optimized", f"{avg_opt_inf:.2f}"])
        writer.writerow(["Full_Frame", f"{avg_ff_inf:.2f}"])
        
        print(f"\n[INFO] Average Inference Time (Optimized): {avg_opt_inf:.2f} ms")
        print(f"[INFO] Average Inference Time (Full Frame): {avg_ff_inf:.2f} ms")

        # --- Append Average Processing Savings Summary ---
        opt_savings = [row[5] for row in optimized_logs if row[5] < 100.0]
        
        avg_opt_saving = sum(opt_savings) / len(opt_savings) if opt_savings else 0

        writer.writerow([])
        writer.writerow(["--- Average Processing Savings (on motion frames) ---"])
        writer.writerow(["Analysis Type", "Average Savings (%)"])
        writer.writerow(["Optimized", f"{avg_opt_saving:.2f}"])
        
        print(f"\n[INFO] Average Processing Savings (Optimized, on motion frames): {avg_opt_saving:.2f}%")


    print("[SUCCESS] Save complete.")


if __name__ == '__main__':
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]
    main()
