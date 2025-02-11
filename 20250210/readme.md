# CNN Image Classification Model Documentation

## Project Overview

This project implements a Convolutional Neural Network (CNN) for image classification, specifically designed to classify 5 different classes (A, E, I, O, U) of grayscale images.

## Technical Details

### 1. Data Processing & Configuration

```python
# Base Configuration
IMG_SIZE = (240, 55)  # Image dimensions
BATCH_SIZE = 8        # Batch size
EPOCHS = 100          # Number of training epochs
NUM_CLASSES = 5       # Number of classes
```

#### Data Augmentation Strategy

- Rotation: ±5 degrees

- Rotation: ±5 degrees

- Rotation: ±5 degrees

- Rotation: ±5 degrees

- Rotation: ±5 degrees

### 2. Model Architecture

The CNN architecture consists of three convolutional blocks followed by a classification head:

```python
def build_simple_model():
    model = models.Sequential([
        # Input Layer
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),

        # First Convolutional Block
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        # Second Convolutional Block
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        # Third Convolutional Block
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        # Classification Head
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
```

### 3. Training Strategy

- **Optimizer**: Adam with learning rate 1e-4
- **Loss Function**: Categorical Crossentropy
- **Class Weighting**: Implemented to handle class imbalance
- **Callbacks**:
  - Early Stopping (patience=20)
  - Learning Rate Reduction
  - Model Checkpoint

### 4. Evaluation Metrics

The model's performance is evaluated using:

- Accuracy (training and validation)
- Confusion Matrix
- Classification Report
- Per-class Accuracy

### 5. Results Analysis

Current results show:

- Training Accuracy: ~93%
- Validation Accuracy: 100%
- Test Accuracy: 100%

#### Batch Size Experiment Results

| Batch Size | Training Accuracy (%) | Validation Accuracy (%) |
| ---------- | --------------------- | ----------------------- |
| 8          | 93.19                 | 100.00                  |
| 16         | 93.19                 | 100.00                  |
| 32         | 93.19                 | 100.00                  |
| 64         | 93.19                 | 100.00                  |

#### Class-wise Performance

All classes (A, E, I, O, U) achieved 100% accuracy with 40 samples each.

### 6. Future Improvements

1. Implement cross-validation
2. Increase dataset size
3. Add more challenging test cases
4. Verify data preprocessing pipeline
5. Consider model complexity reduction

### 7. Data Management

- Dataset split ratio: 80% training, 10% validation, 10% testing
- Automated directory creation and file management
- Comprehensive data visualization tools

### 8. Visualization Features

- Training history plots (accuracy and loss)
- Confusion matrix visualization
- Sample batch visualization
- Detailed prediction analysis

This implementation provides a robust framework for image classification tasks while maintaining flexibility for future modifications and improvements.
