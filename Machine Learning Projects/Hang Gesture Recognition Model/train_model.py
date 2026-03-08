import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from data_loader import load_dataset, get_data_augmentation
from model_architecture import create_cnn_model, print_model_summary

def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training and validation accuracy/loss curves.
    
    Args:
        history: Keras training history object
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def train_gesture_model(dataset_path='leapGestRecog', epochs=50, batch_size=32):
    """
    Train the hand gesture recognition model.
    
    Args:
        dataset_path: Path to the LeapGestRecog dataset
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    print("="*60)
    print("HAND GESTURE RECOGNITION - TRAINING")
    print("="*60)
    
    # Load dataset
    print("\n[1/5] Loading dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test, class_names = load_dataset(dataset_path)
    
    # Save class names for later use
    np.save('class_names.npy', class_names)
    print(f"Class names saved to class_names.npy")
    
    # Create model
    print("\n[2/5] Creating CNN model...")
    input_shape = X_train.shape[1:]  # (64, 64, 3)
    num_classes = len(class_names)
    model = create_cnn_model(input_shape=input_shape, num_classes=num_classes)
    print_model_summary(model)
    
    # Setup data augmentation
    print("\n[3/5] Setting up data augmentation...")
    datagen = get_data_augmentation()
    datagen.fit(X_train)
    
    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'best_gesture_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    print(f"\n[4/5] Training model for {epochs} epochs...")
    print(f"Batch size: {batch_size}")
    print("-"*60)
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n[5/5] Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Save final model
    model.save('gesture_model.h5')
    print(f"\nFinal model saved to gesture_model.h5")
    
    # Plot training history
    plot_training_history(history)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return model, history


if __name__ == "__main__":
    # Configuration
    DATASET_PATH = 'leapGestRecog'
    EPOCHS = 50
    BATCH_SIZE = 32
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"\n{'='*60}")
        print("ERROR: Dataset not found!")
        print(f"{'='*60}")
        print(f"\nPlease download the LeapGestRecog dataset from:")
        print("https://www.kaggle.com/gti-upm/leapgestrecog")
        print(f"\nExtract it to: {os.path.abspath(DATASET_PATH)}")
        print(f"\n{'='*60}\n")
    else:
        # Train the model
        model, history = train_gesture_model(
            dataset_path=DATASET_PATH,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )
