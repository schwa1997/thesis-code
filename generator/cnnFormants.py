from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing import image
import datetime
import io

# Build custom CNN model to learn local features of images (e.g. formants)
def build_cnn_model(input_shape, num_classes):
    model = Sequential()
    
    # First convolutional and pooling layer
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Second convolutional and pooling layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Third convolutional and pooling layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Flatten and classify global features
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Data preparation
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Image normalization
    validation_split=0.3,  # Split data into training and validation sets
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    r'C:\Users\huimin.chen\Downloads\thesis-code-20241023T072444Z-001\thesis-code\generator\output',  # Image storage path
    target_size=(128, 128),  # Adjust image size to match model input
    batch_size=32,
    class_mode='categorical',
    subset='training'  # Use training set
)

validation_generator = train_datagen.flow_from_directory(
    r'C:\Users\huimin.chen\Downloads\thesis-code-20241023T072444Z-001\thesis-code\generator\output',  # Same image path
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # Use validation set
)

# Get class labels
class_indices = train_generator.class_indices
class_labels = {v: k for k, v in class_indices.items()}

print("Training set class mapping (integer -> class name):", class_labels)
print("Validation set class mapping (integer -> class name):", {v: k for k, v in validation_generator.class_indices.items()})

# Check if test set classes are in class_labels
test_classes = ['A', 'E', 'I', 'O', 'U']
for cls in test_classes:
    if cls not in class_labels.values():
        print(f"Warning: Test set class '{cls}' not in training set classes. Please check if class names are consistent.")

# Calculate class weights to handle data imbalance
train_labels = train_generator.classes
class_weights_values = class_weight.compute_class_weight('balanced',
                                                          classes=np.unique(train_labels),
                                                          y=train_labels)
class_weights = {i: weight for i, weight in enumerate(class_weights_values)}
print("Class weights:", class_weights)

# Build custom CNN model
model = build_cnn_model(input_shape=(128, 128, 3), num_classes=len(class_indices))

# Define callbacks
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)
lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=5, factor=0.5, min_lr=1e-6, verbose=1)

# Define output directory
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(r'C:\Users\huimin.chen\Downloads\thesis-code-20241023T072444Z-001\thesis-code\generator\analysis_results', timestamp)
os.makedirs(output_dir, exist_ok=True)

# Visualize training process
def plot_history(history, output_dir):
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
    plt.title('accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='training loss')
    plt.plot(epochs_range, val_loss, label='validation loss')
    plt.legend(loc='upper right')
    plt.title('loss')
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

# Use model for prediction
def predict_vowel(image_path, model, class_labels):
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_label = class_labels.get(predicted_index, "Unknown")

    return predicted_label

# Evaluate test set
def evaluate_test_set(model, class_labels, test_dir, output_dir):
    correct = 0
    total = 0
    per_class_correct = {label: 0 for label in class_labels.values()}
    per_class_total = {label: 0 for label in class_labels.values()}
    y_true = []
    y_pred = []

    for class_name in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for file in os.listdir(class_path):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(class_path, file)
                predicted_class = predict_vowel(image_path, model, class_labels)
                true_class = class_name.upper()

                if true_class not in per_class_total:
                    print(f"Warning: Undefined class '{true_class}' found in test set.")
                    continue

                total += 1
                per_class_total[true_class] += 1
                if predicted_class == true_class:
                    correct += 1
                    per_class_correct[true_class] += 1

                y_true.append(true_class)
                y_pred.append(predicted_class)

    # Calculate overall accuracy
    overall_accuracy = correct / total * 100 if total > 0 else 0
    
    # Create report file
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    with io.open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Total correct predictions: {correct} / {total}\n")
        f.write(f"Overall accuracy: {overall_accuracy:.2f}%\n\n")

        f.write("Classification accuracy for each class:\n")
        for label in per_class_total:
            if per_class_total[label] > 0:
                accuracy = per_class_correct[label] / per_class_total[label] * 100
                f.write(f"  Class '{label}': {per_class_correct[label]} / {per_class_total[label]} = {accuracy:.2f}%\n")
            else:
                f.write(f"  Class '{label}': No test samples\n")

        f.write("\nClassification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=sorted(class_labels.values())))

    cm = confusion_matrix(y_true, y_pred, labels=sorted(class_labels.values()))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(class_labels.values()),
                yticklabels=sorted(class_labels.values()))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

# Define test set path
test_dir = r'C:\Users\huimin.chen\Downloads\thesis-code-20241023T072444Z-001\thesis-code\generator\test'

# Train model
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    class_weight=class_weights,
    callbacks=[early_stopping, lr_reduction]
)

# Save training history plot
plot_history(history, output_dir)

# Evaluate test set and save results
evaluate_test_set(model, class_labels, test_dir, output_dir)

print(f"Analysis report and images have been saved in folder: {output_dir}")
