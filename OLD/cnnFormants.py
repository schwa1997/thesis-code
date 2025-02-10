from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing import image
import datetime
import io

# Build custom CNN model to learn local features of images (e.g. formants)
def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv1_1'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv1_2'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu', padding='same', name='Conv2_1'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same', name='Conv2_2'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(256, (3, 3), activation='relu', padding='same', name='Conv3_1'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same', name='Conv3_2'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Data preparation
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Image normalization
    rotation_range=15,  # Reduced rotation to preserve formant structure
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=False,  # Formants are position-sensitive
    fill_mode='nearest',
    validation_split=0.2  # Split 20% of data for validation
)

# Define the common directory
data_dir = r'C:\Users\huimin.chen\Downloads\thesis-code-20241023T072444Z-001\thesis-code\audio\output'

# Function to create generators with different batch sizes
def create_generators(batch_size):
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(240, 55),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(240, 55),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator

# Get class labels
class_indices = train_datagen.flow_from_directory(data_dir, target_size=(240, 55), class_mode='categorical', subset='training').class_indices
class_labels = {v: k for k, v in class_indices.items()}

print("Class mapping (integer -> class name):", class_labels)

# Calculate class weights to handle data imbalance
train_labels = train_datagen.flow_from_directory(data_dir, target_size=(240, 55), class_mode='categorical', subset='training').classes
class_weights_values = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = {i: weight for i, weight in enumerate(class_weights_values)}
print("Class weights:", class_weights)

# Build custom CNN model
model = build_cnn_model(input_shape=(240, 55, 3), num_classes=len(class_indices))

# Define callbacks
early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, verbose=1)
lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=5, factor=0.2, min_lr=1e-6, verbose=1)

# Define output directory
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(r'C:\Users\huimin.chen\Downloads\thesis-code-20241023T072444Z-001\thesis-code\analysis_results', timestamp)
os.makedirs(output_dir, exist_ok=True)

# Visualize training process
def plot_history(history, output_dir, batch_size):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='training accuracy')
    plt.plot(epochs_range, val_acc, label='validation accuracy')
    plt.legend(loc='lower right')
    plt.title(f'Accuracy (Batch Size: {batch_size})')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='training loss')
    plt.plot(epochs_range, val_loss, label='validation loss')
    plt.legend(loc='upper right')
    plt.title(f'Loss (Batch Size: {batch_size})')
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f'training_history_batch_{batch_size}.png'))
    plt.close()

# Function to train and evaluate model with different batch sizes
def train_and_evaluate(batch_sizes, epochs=150):
    results = []
    for batch_size in batch_sizes:
        print(f"\nTraining with batch size: {batch_size}")
        train_generator, validation_generator = create_generators(batch_size)
        
        # Reset model
        model = build_cnn_model(input_shape=(240, 55, 3), num_classes=len(class_indices))
        
        # Train model
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            class_weight=class_weights,
            callbacks=[early_stopping, lr_reduction]
        )
        
        # Evaluate on validation set
        val_loss, val_accuracy = model.evaluate(validation_generator)
        
        # Save training history plot
        plot_history(history, output_dir, batch_size)
        
        results.append({
            'batch_size': batch_size,
            'train_accuracy': history.history['accuracy'][-1] * 100,
            'val_accuracy': val_accuracy * 100
        })
    
    return results

# Train and evaluate with different batch sizes
batch_sizes = [5,10,15,20]
results = train_and_evaluate(batch_sizes)

# Create report file
report_path = os.path.join(output_dir, 'analysis_report.txt')
with io.open(report_path, 'w', encoding='utf-8') as f:
    f.write("Results for different batch sizes (Epoch = 150):\n\n")
    for result in results:
        f.write(f"Batch Size: {result['batch_size']}\n")
        f.write(f"Training Accuracy: {result['train_accuracy']:.2f}%\n")
        f.write(f"Validation Accuracy: {result['val_accuracy']:.2f}%\n\n")

print(f"Analysis report and images have been saved in folder: {output_dir}")
