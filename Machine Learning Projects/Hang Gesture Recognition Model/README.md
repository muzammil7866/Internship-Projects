# Hand Gesture Recognition Model

A deep learning project that uses Convolutional Neural Networks (CNNs) to classify hand gestures from images, enabling intuitive human-computer interaction and gesture-based control systems.

## Project Overview

This project implements a CNN-based image classifier trained on the **LeapGestRecog** dataset from Kaggle. The model can accurately identify different hand gestures, making it suitable for touchless interfaces, accessibility applications, and interactive systems.

### Key Features
- **Deep CNN Architecture**: Multi-layer convolutional network with batch normalization and dropout
- **Data Augmentation**: Enhanced generalization through rotation, shifting, and flipping
- **Transfer-ready**: Modular code structure for easy integration into applications
- **Visualization**: Training history plots for performance analysis

## Dataset Setup

### Download the Dataset

1. Visit the Kaggle dataset page: [LeapGestRecog Dataset](https://www.kaggle.com/gti-upm/leapgestrecog)

2. Download the dataset (you'll need a Kaggle account)

3. Extract the downloaded zip file into the project directory:
   ```
   Hang Gesture Recognition Model/
   ├── leapGestRecog/
   │   ├── 00/
   │   ├── 01/
   │   └── ...
   ├── train_model.py
   ├── predict.py
   └── ...
   ```

### Alternative: Using Kaggle API

```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset
kaggle datasets download -d gti-upm/leapgestrecog

# Extract
unzip leapgestrecog.zip -d leapGestRecog
```

## Installation

1. Ensure you have Python 3.8+ installed

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

Run the training script to train the CNN on the LeapGestRecog dataset:

```bash
python train_model.py
```

**What happens during training:**
- Loads and preprocesses images from the dataset
- Splits data into train/validation/test sets (70/15/15)
- Applies data augmentation
- Trains for up to 50 epochs (with early stopping)
- Saves the best model as `gesture_model.h5`
- Generates a training history plot (`training_history.png`)

**Training outputs:**
- `gesture_model.h5`: The trained model
- `best_gesture_model.h5`: Best model checkpoint
- `class_names.npy`: List of gesture class names
- `training_history.png`: Accuracy and loss curves

### Predicting Gestures

Use the trained model to predict gestures from new images:

```bash
python predict.py path/to/your/image.png
```

**Example:**
```bash
python predict.py test_gesture.jpg
```

**Output:**
```
PREDICTION RESULTS
==============================================================
Predicted Gesture: PALM
Confidence: 98.76%
==============================================================
```

## Model Architecture

The CNN consists of:
- **3 Convolutional Blocks**: Each with two Conv2D layers, BatchNormalization, MaxPooling, and Dropout
- **Feature Maps**: 32 → 64 → 128 channels progressively
- **Dense Layers**: 256 → 128 → num_classes
- **Regularization**: Dropout (0.25-0.5) and Batch Normalization
- **Optimizer**: Adam with learning rate scheduling

Total parameters: ~1.5M (depends on number of gesture classes)

## Business Goals Achieved

This hand gesture recognition technology addresses several critical business needs:

### 1. **Touchless Interfaces**
   - **Goal**: Enable gesture-controlled devices for smart homes, public kiosks, and automotive systems
   - **Application**: Users can control TV volume, navigate menus, or adjust smart home settings without physical contact
   - **Value**: Improved hygiene, modern UX, differentiation from competitors

### 2. **Accessibility Solutions**
   - **Goal**: Assist users with mobility impairments or limited dexterity
   - **Application**: People who cannot use traditional input devices can interact with computers through gestures
   - **Value**: Inclusive design, compliance with accessibility standards, expanded user base

### 3. **Gaming & Virtual Reality**
   - **Goal**: Provide natural, immersive control mechanisms
   - **Application**: Hand gestures for character actions, UI navigation, or object manipulation in VR/AR environments
   - **Value**: Enhanced player engagement, intuitive controls, next-gen gaming experiences

### 4. **Healthcare & Sterile Environments**
   - **Goal**: Enable touchless interaction in settings where contamination is a concern
   - **Application**: Surgeons controlling medical imaging displays during operations without breaking sterility
   - **Value**: Patient safety, workflow efficiency, compliance with medical protocols

### 5. **Industrial & Manufacturing**
   - **Goal**: Control machinery or robots in environments where hands are occupied or dirty
   - **Application**: Workers wearing gloves or handling materials can issue commands via gestures
   - **Value**: Increased productivity, reduced equipment wear, improved safety

## Project Structure

```
Hang Gesture Recognition Model/
├── data_loader.py          # Dataset loading and preprocessing
├── model_architecture.py   # CNN model definition
├── train_model.py          # Training script
├── predict.py              # Inference script
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── leapGestRecog/          # Dataset directory (not included, must download)
```

## Requirements

- Python 3.8+
- TensorFlow 2.10+
- NumPy
- OpenCV
- Matplotlib
- scikit-learn

## Notes

- **GPU Recommended**: Training is significantly faster with a CUDA-enabled GPU
- **Training Time**: On CPU, expect 2-3 hours for 50 epochs. On GPU: 15-30 minutes
- **Memory**: Ensure you have at least 8GB RAM for training
- **Dataset Size**: LeapGestRecog contains ~20,000 images across 10 gesture classes

## Troubleshooting

**Issue**: `FileNotFoundError: Dataset path not found`  
**Solution**: Ensure you've downloaded and extracted the LeapGestRecog dataset to the project directory

**Issue**: Out of memory during training  
**Solution**: Reduce `batch_size` in `train_model.py` (try 16 or 8)

**Issue**: Poor accuracy  
**Solution**: Train for more epochs, adjust learning rate, or try different augmentation settings
