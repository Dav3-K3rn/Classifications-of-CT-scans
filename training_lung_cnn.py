import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd
import json
import os

# ---------- CONFIG ----------
DATA_DIR = "output_dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
MODEL_SAVE_PATH = "lung_cancer_cnn_enhanced.h5"
RESULTS_SAVE_PATH = "training_results.json"
# ----------------------------

def load_dataset():
    """Load and prepare dataset"""
    print("Loading dataset...")
    train_ds = image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # Get class names
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Class names: {class_names}")
    print(f"Number of classes: {num_classes}")

    # Prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names, num_classes

def create_baseline_model(input_shape, num_classes):
    """Phase 1: Baseline CNN from scratch"""
    print("Creating baseline CNN model...")
    
    data_augmentation = models.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.2),
    ])
    
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        data_augmentation,
        layers.Rescaling(1./255),
        
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(),
        
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(),
        
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(),
        
        layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def run_phase_1(train_ds, val_ds, num_classes):
    """Run Phase 1: Baseline Model"""
    print("\n" + "="*50)
    print("PHASE 1: BASELINE MODEL DEVELOPMENT")
    print("="*50)
    
    baseline_model = create_baseline_model(IMG_SIZE + (3,), num_classes)
    baseline_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    baseline_history = baseline_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1
    )
    
    return baseline_model, baseline_history.history

def run_phase_2(train_ds, val_ds, num_classes):
    """Run Phase 2: Transfer Learning"""
    print("\n" + "="*50)
    print("PHASE 2: TRANSFER LEARNING MODELS")
    print("="*50)
    
    # ... (transfer learning code from previous version)
    # This would include VGG16, ResNet50 models
    # Return best_model and histories
    
    # For now, return baseline as placeholder
    return run_phase_1(train_ds, val_ds, num_classes)

def run_phase_3(model, val_ds, class_names):
    """Run Phase 3: Comprehensive Evaluation"""
    print("\n" + "="*50)
    print("PHASE 3: COMPREHENSIVE MODEL EVALUATION")
    print("="*50)
    
    # Get predictions
    y_true = []
    y_pred_proba = []
    
    for images, labels in val_ds:
        y_true.extend(labels.numpy())
        predictions = model.predict(images, verbose=0)
        y_pred_proba.extend(predictions)
    
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Generate evaluation metrics
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Save results
    results = {
        'y_true': y_true.tolist(),
        'y_pred': y_pred.tolist(),
        'y_pred_proba': y_pred_proba.tolist(),
        'class_names': class_names,
        'classification_report': report
    }
    
    with open(RESULTS_SAVE_PATH, 'w') as f:
        json.dump(results, f)
    
    print(f"Results saved to {RESULTS_SAVE_PATH}")
    return results

def main():
    """Run Phases 1-3"""
    print("Starting Phases 1-3: Training and Evaluation...")
    
    # Load data
    train_ds, val_ds, class_names, num_classes = load_dataset()
    
    # Run phases
    best_model, training_history = run_phase_2(train_ds, val_ds, num_classes)
    
    # Evaluate
    results = run_phase_3(best_model, val_ds, class_names)
    
    # Save model
    best_model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    print("Phases 1-3 completed successfully!")

if __name__ == "__main__":
    main()