import cv2
import numpy as np
import os
import collections
import glob

# --- Main Configuration ---
# --- Input/Output Folders ---
# The script will process all videos in this folder.
# MODIFIED FOR WINDOWS: Using the full, absolute path to your videos folder.
SOURCE_VIDEO_FOLDER = r"C:\Users\DAS_3\Desktop\project\sourceVideos"
# The script will create and save the dataset here.
OUTPUT_DATASET_FOLDER = "dataset"

# --- Motion Extraction Settings ---
# These settings create the unique visual filter for the student model to learn.
MOTION_DELAY_SECONDS = 0.1 
MOTION_SENSITIVITY = 5

# --- "Teacher" Model Configuration (MobileNet-SSD) ---
# This model finds people in the clean frames to create the labels.
CONFIDENCE_THRESHOLD = 0.5 # Only save detections the teacher is >50% sure about.

# Define the absolute path to your models folder using a raw string (r"...")
# to correctly handle backslashes.
MODEL_DIR = r"C:\Users\DAS_3\Desktop\models"
prototxt_path = os.path.join(MODEL_DIR, "deploy.prototxt")
model_path = os.path.join(MODEL_DIR, "mobilenet_iter_73000.caffemodel")
# --- End Configuration ---


def create_dataset():
    """
    Processes source videos to generate a dataset for training a "student" model.
    For each frame with a detected person, it saves the filtered image and a
    bounding box label file.
    """
    # --- Load the Teacher Model ---
    print("[INFO] Loading 'teacher' model (MobileNet-SSD)...")
    if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
        print(f"[ERROR] Teacher model files not found in '{MODEL_DIR}'.")
        print("[ERROR] Please ensure the path is correct and the files are present.")
        return
    
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]
    print("[INFO] Teacher model loaded successfully.")

    # --- Setup Output Directories ---
    print(f"[INFO] Setting up output directory: '{OUTPUT_DATASET_FOLDER}'")
    filtered_images_path = os.path.join(OUTPUT_DATASET_FOLDER, "filtered_images")
    labels_path = os.path.join(OUTPUT_DATASET_FOLDER, "labels")
    os.makedirs(filtered_images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)
    print("[INFO] Directories are ready.")

    # --- Find and Process Videos ---
    video_paths = glob.glob(os.path.join(SOURCE_VIDEO_FOLDER, '*'))
    if not video_paths:
        print(f"[ERROR] No video files found in '{SOURCE_VIDEO_FOLDER}'. Please add videos and try again.")
        return
        
    print(f"[INFO] Found {len(video_paths)} video(s) to process...")
    total_detections = 0

    for video_path in video_paths:
        print(f"\n[INFO] Processing video: {os.path.basename(video_path)}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[WARNING] Could not open video file: {video_path}. Skipping.")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30
        delay_frames = int(fps * MOTION_DELAY_SECONDS)
        if delay_frames < 1: delay_frames = 1
        
        frame_buffer = collections.deque(maxlen=delay_frames)
        frame_count = 0

        while True:
            ret, current_frame = cap.read()
            if not ret:
                break # End of video

            frame_count += 1
            frame_buffer.append(current_frame)
            
            # We can only process once the buffer for the motion effect is full.
            if len(frame_buffer) == delay_frames:
                delayed_frame = frame_buffer[0]
                
                # --- Step 1: Run the Teacher Model on the CLEAN frame ---
                (h, w) = current_frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(current_frame, (300, 300)), 0.007843, (300, 300), 127.5)
                net.setInput(blob)
                detections = net.forward()

                # --- Step 2: Check for a 'person' detection ---
                for i in np.arange(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > CONFIDENCE_THRESHOLD:
                        idx = int(detections[0, 0, i, 1])
                        if CLASSES[idx] == "person":
                            # A person was found! Now we generate the data.
                            total_detections += 1
                            
                            # --- Step 3: Create the Filtered Image ---
                            inverted_delayed_frame = cv2.bitwise_not(delayed_frame)
                            filtered_frame = cv2.addWeighted(
                                src1=current_frame, alpha=1.0, src2=inverted_delayed_frame,
                                beta=0.99, gamma=-225.0
                            )
                            
                            # --- Step 4: Get Bounding Box and Save Data ---
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")
                            
                            # Normalize coordinates for the label file (common practice)
                            norm_x_center = ((startX + endX) / 2) / w
                            norm_y_center = ((startY + endY) / 2) / h
                            norm_width = (endX - startX) / w
                            norm_height = (endY - startY) / h

                            # Define unique filenames
                            base_filename = f"{os.path.basename(video_path).split('.')[0]}_frame_{frame_count}"
                            image_filename = os.path.join(filtered_images_path, f"{base_filename}.jpg")
                            label_filename = os.path.join(labels_path, f"{base_filename}.txt")
                            
                            # Save the filtered image and the label file
                            cv2.imwrite(image_filename, filtered_frame)
                            with open(label_filename, 'w') as f:
                                # Format: class_index x_center y_center width height
                                f.write(f"0 {norm_x_center} {norm_y_center} {norm_width} {norm_height}\n")
                            
                            # We only process the first person found in a frame to keep it simple.
                            break 
                
                # Provide progress update in the console
                if frame_count % 100 == 0:
                    print(f"  ...scanned {frame_count} frames, found {total_detections} total person detections.")

        cap.release()
        print(f"[INFO] Finished processing {os.path.basename(video_path)}.")

    print(f"\n[SUCCESS] Dataset generation complete. Found and saved {total_detections} total samples.")

if __name__ == '__main__':
    create_dataset()
