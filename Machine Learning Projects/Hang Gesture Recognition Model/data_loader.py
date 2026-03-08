import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

def load_dataset(dataset_path, img_size=(64, 64)):
    """
    Load images from the LeapGestRecog dataset structure.
    
    Args:
        dataset_path: Path to the leapGestRecog directory
        img_size: Target size for images (height, width)
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, class_names
    """
    images = []
    labels = []
    class_names = []
    
    # The LeapGestRecog dataset has structure: leapGestRecog/00/01_palm/*.png
    # Where 00-09 are subjects and 01-10 are gesture types
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    subjects = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    
    for subject_idx, subject in enumerate(subjects):
        subject_path = os.path.join(dataset_path, subject)
        
        gesture_folders = sorted([d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))])
        
        for gesture_folder in gesture_folders:
            gesture_path = os.path.join(subject_path, gesture_folder)
            
            # Extract gesture class from folder name (e.g., "01_palm" -> "palm")
            gesture_name = gesture_folder.split('_', 1)[1] if '_' in gesture_folder else gesture_folder
            
            if gesture_name not in class_names:
                class_names.append(gesture_name)
            
            gesture_label = class_names.index(gesture_name)
            
            # Load all images in this gesture folder
            for img_file in os.listdir(gesture_path):
                if img_file.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(gesture_path, img_file)
                    
                    # Read and preprocess image
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    # Convert to RGB (OpenCV loads as BGR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Resize
                    img = cv2.resize(img, img_size)
                    
                    images.append(img)
                    labels.append(gesture_label)
    
    # Convert to numpy arrays
    X = np.array(images, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    y = np.array(labels)
    
    # Split dataset: 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"Dataset loaded successfully!")
    print(f"Classes: {class_names}")
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, class_names


def get_data_augmentation():
    """
    Create data augmentation configuration for training.
    
    Returns:
        ImageDataGenerator configured for augmentation
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    return datagen
