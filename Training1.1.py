import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
import math

# --- Main Configuration ---
# --- Dataset and Model Paths ---
# Path to the dataset you generated in Phase 1.
DATASET_PATH = r"C:\Users\DAS_3\dataset"
# Where to save the final, trained student model.
OUTPUT_MODEL_NAME = "delta_trace_student_model.h5"

# --- Training Hyperparameters ---
# How many times the model will see the entire dataset.
EPOCHS = 20
# How many images to process at once.
BATCH_SIZE = 32
# The dimensions to resize images to for the model.
IMG_WIDTH = 224
IMG_HEIGHT = 224
# --- End Configuration ---


def data_generator(image_paths, label_paths, batch_size, target_size):
    """
    A generator that yields batches of images and labels from the disk.
    This avoids loading the entire dataset into memory.
    """
    # Create a dictionary to map image paths to their corresponding label paths
    path_map = dict(zip(image_paths, label_paths))
    num_samples = len(image_paths)

    while True: # Loop forever so the generator never terminates
        # Create a temporary list of paths to shuffle
        temp_paths = list(image_paths)
        np.random.shuffle(temp_paths)
        
        for offset in range(0, num_samples, batch_size):
            # Get a batch of file paths
            batch_paths = temp_paths[offset:offset+batch_size]
            
            images = []
            labels = []
            
            for img_path in batch_paths:
                # Load and process the image
                try:
                    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    images.append(img_array)

                    # Load the corresponding label
                    label_path = path_map[img_path]
                    with open(label_path, 'r') as f:
                        parts = f.read().strip().split()
                        labels.append([float(p) for p in parts[1:]])
                except Exception as e:
                    print(f"\n[WARNING] Skipping file due to error: {e} - Path: {img_path}")
                    continue
            
            # Normalize images and convert to numpy arrays
            X = np.array(images) / 255.0
            y = np.array(labels)
            
            yield X, y


def build_student_model(height, width, channels):
    """
    Defines the architecture for our lightweight "student" model.
    """
    model = Sequential([
        Input(shape=(height, width, channels)),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(4, activation='linear')
    ])
    
    return model


def main():
    """
    Main function to load data, build, train, and save the model.
    """
    # --- Phase 1: Get File Paths ---
    print("[INFO] Getting and verifying image and label file paths...")
    all_image_paths = glob.glob(os.path.join(DATASET_PATH, "filtered_images", "*.jpg"))
    
    synced_image_paths = []
    synced_label_paths = []

    if not all_image_paths:
        print(f"[ERROR] No images found. Please check the path: {os.path.join(DATASET_PATH, 'filtered_images')}")
        return

    # Create a set of label basenames for fast lookup
    label_basenames = {os.path.basename(p) for p in glob.glob(os.path.join(DATASET_PATH, "labels", "*.txt"))}

    for img_path in all_image_paths:
        # Construct the expected corresponding label filename
        label_filename = os.path.basename(img_path).replace(".jpg", ".txt")
        
        # Check if the label file actually exists
        if label_filename in label_basenames:
            synced_image_paths.append(img_path)
            synced_label_paths.append(os.path.join(DATASET_PATH, "labels", label_filename))

    print(f"[INFO] Found {len(synced_image_paths)} synchronized image/label pairs.")

    if not synced_image_paths:
        print("[ERROR] No matching image/label pairs found. Cannot proceed with training.")
        return

    # --- Phase 2: Split Data Paths ---
    # Split the list of file paths, not the data itself.
    print("[INFO] Splitting data paths into training and validation sets...")
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        synced_image_paths, synced_label_paths, test_size=0.2, random_state=42
    )
    print(f"[INFO] Training samples: {len(train_paths)}, Validation samples: {len(val_paths)}")

    # --- Phase 3: Build and Compile Model ---
    print("[INFO] Building the student model...")
    student_model = build_student_model(IMG_HEIGHT, IMG_WIDTH, 3)
    student_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    student_model.summary()

    # --- Phase 4: Create Generators and Train the Model ---
    print("\n[INFO] Creating data generators...")
    train_gen = data_generator(train_paths, train_labels, BATCH_SIZE, (IMG_HEIGHT, IMG_WIDTH))
    val_gen = data_generator(val_paths, val_labels, BATCH_SIZE, (IMG_HEIGHT, IMG_WIDTH))

    print("\n[INFO] Starting model training...")
    print("="*50)
    # The model now trains using the generator.
    history = student_model.fit(
        train_gen,
        steps_per_epoch=math.ceil(len(train_paths) / BATCH_SIZE),
        validation_data=val_gen,
        validation_steps=math.ceil(len(val_paths) / BATCH_SIZE),
        epochs=EPOCHS
    )
    print("="*50)
    print("[INFO] Training complete.")

    # --- Phase 5: Save the Final Model ---
    print(f"[INFO] Saving trained model to '{OUTPUT_MODEL_NAME}'...")
    student_model.save(OUTPUT_MODEL_NAME)
    print("[SUCCESS] Student model has been saved.")


if __name__ == '__main__':
    # Ensure you have tensorflow and scikit-learn installed:
    # pip install tensorflow scikit-learn
    main()
