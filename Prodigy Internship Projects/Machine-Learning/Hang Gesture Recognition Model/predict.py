import sys
import numpy as np
import cv2
from tensorflow.keras.models import load_model

def preprocess_image(image_path, img_size=(64, 64)):
    """
    Load and preprocess an image for prediction.
    
    Args:
        image_path: Path to the image file
        img_size: Target size for the image
    
    Returns:
        Preprocessed image array
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, img_size)
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img


def predict_gesture(image_path, model_path='gesture_model.h5', class_names_path='class_names.npy'):
    """
    Predict the gesture from an image.
    
    Args:
        image_path: Path to the input image
        model_path: Path to the trained model
        class_names_path: Path to the class names file
    
    Returns:
        Predicted class name and confidence score
    """
    print("="*60)
    print("HAND GESTURE RECOGNITION - PREDICTION")
    print("="*60)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = load_model(model_path)
    
    # Load class names
    print(f"Loading class names from {class_names_path}...")
    class_names = np.load(class_names_path, allow_pickle=True)
    
    # Preprocess image
    print(f"Processing image: {image_path}...")
    img = preprocess_image(image_path)
    
    # Make prediction
    print("Making prediction...")
    predictions = model.predict(img, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    predicted_class = class_names[predicted_class_idx]
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Predicted Gesture: {predicted_class.upper()}")
    print(f"Confidence: {confidence*100:.2f}%")
    print("="*60)
    
    # Show top 3 predictions
    print("\nTop 3 Predictions:")
    print("-"*60)
    top_indices = np.argsort(predictions[0])[::-1][:3]
    for i, idx in enumerate(top_indices, 1):
        print(f"{i}. {class_names[idx]}: {predictions[0][idx]*100:.2f}%")
    print("="*60 + "\n")
    
    return predicted_class, confidence


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python predict.py <image_path>")
        print("Example: python predict.py test_image.png\n")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    try:
        predicted_class, confidence = predict_gesture(image_path)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure you have:")
        print("1. Trained the model (run train_model.py first)")
        print("2. gesture_model.h5 exists in the current directory")
        print("3. class_names.npy exists in the current directory\n")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}\n")
        sys.exit(1)
