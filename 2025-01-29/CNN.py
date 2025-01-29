import os
import shutil
import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from io import StringIO

# 可选：禁用 oneDNN 警告
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Define directories
base_dir = r"C:\Users\huimin.chen\Downloads\code-20250129T112844Z-001\code"
image_dir = os.path.join(base_dir, 'image')
analysis_dir = os.path.join(base_dir, 'analysis')
os.makedirs(analysis_dir, exist_ok=True)

# Parameters
img_size = (55, 55)
batch_size = 32  # 修改批次大小
epochs = 30      # 修改训练轮次
initial_learning_rate = 0.001 # 修改学习率

# Split data into training and validation sets manually
def create_train_val_dirs(base_dir, train_ratio=0.8):
    # Create directories
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
    
    for class_name in class_dirs:
        # Create class directories in train and val
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        
        # Get all images in the class
        class_path = os.path.join(image_dir, class_name)
        images = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Split images
        train_images, val_images = train_test_split(images, train_size=train_ratio, random_state=42)
        
        # Copy images to respective directories
        for img in train_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(train_dir, class_name, img)
            shutil.copy2(src, dst)
            
        for img in val_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(val_dir, class_name, img)
            shutil.copy2(src, dst)
    
    return train_dir, val_dir

# Create train and validation directories
train_dir, val_dir = create_train_val_dirs(base_dir)

# Create data generators
train_datagen = ImageDataGenerator(
    validation_split=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False
)

val_datagen = ImageDataGenerator()

# Create generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Print dataset information
print("\nDataset Information:")
print(f"Number of training samples: {train_generator.samples}")
print(f"Number of validation samples: {validation_generator.samples}")
print(f"Number of classes: {len(train_generator.class_indices)}")
print("Class mapping:", train_generator.class_indices)

# Adjust batch size for small dataset
batch_size = min(batch_size, train_generator.samples // 2)

# Modified model architecture optimized for spectrograms
def build_model(input_shape, num_classes):
    # 使用预训练的 ResNet50V2
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    
    # 冻结预训练模型的权重
    base_model.trainable = False
    
    model = models.Sequential([
        # 预处理层
        layers.Input(shape=input_shape),
        layers.Rescaling(1./255),  # 归一化
        
        # 预训练模型
        base_model,
        
        # 分类层
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Create and compile model
model = build_model((img_size[0], img_size[1], 3), num_classes=len(train_generator.class_indices))

# 修改回调函数
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.00001
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
]

# 使用混合精度训练
try:
    from tensorflow.keras.mixed_precision import set_global_policy
    set_global_policy('mixed_float16')
except:
    print("Mixed precision training not available")

# 编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 训练策略：先训练分类层
history_1 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    callbacks=callbacks
)

# 解冻部分预训练层进行微调
base_model.trainable = True
for layer in base_model.layers[:-10]:  # 保持前面的层冻结
    layer.trainable = False

# 使用更小的学习率进行微调
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 继续训练
history_2 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,
    callbacks=callbacks
)

# Save the model
model_path = os.path.join(analysis_dir, "vowel_cnn_model.h5")
model.save(model_path)
print(f"Model saved at {model_path}")

# Evaluate the model
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Save accuracy and loss plots
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_1.history['accuracy'], label='Training Accuracy (First 10 epochs)')
plt.plot(history_1.history['val_accuracy'], label='Validation Accuracy (First 10 epochs)')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_1.history['loss'], label='Training Loss (First 10 epochs)')
plt.plot(history_1.history['val_loss'], label='Validation Loss (First 10 epochs)')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Classification report and confusion matrix
y_true = validation_generator.classes
y_pred = np.argmax(model.predict(validation_generator), axis=1)
class_labels = list(validation_generator.class_indices.keys())

report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
report_path = os.path.join(analysis_dir, 'classification_report.json')
with open(report_path, 'w') as f:
    json.dump(report, f, indent=4)
print(f"Classification report saved at {report_path}")

conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure()
plt.imshow(conf_matrix, cmap='coolwarm', interpolation='nearest')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks(ticks=np.arange(len(class_labels)), labels=class_labels)
plt.yticks(ticks=np.arange(len(class_labels)), labels=class_labels)
conf_matrix_path = os.path.join(analysis_dir, 'confusion_matrix.png')
plt.savefig(conf_matrix_path)
print(f"Confusion matrix saved at {conf_matrix_path}")

# Add a separate test data generator
test_datagen = ImageDataGenerator(rescale=1./255)  # Only rescaling for test data

# Create a separate test generator
test_generator = test_datagen.flow_from_directory(
    image_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Important for maintaining order in predictions
)

# After training, add detailed evaluation code
print("\nEvaluating model on test data:")
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Get predictions for individual samples
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

# Print detailed metrics for each class
print("\nDetailed Classification Report:")
class_labels = list(test_generator.class_indices.keys())
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Save sample predictions
print("\nSample Predictions:")
for i in range(min(5, len(predicted_classes))):
    true_class = class_labels[true_classes[i]]
    pred_class = class_labels[predicted_classes[i]]
    confidence = predictions[i][predicted_classes[i]] * 100
    print(f"Sample {i+1}: True: {true_class}, Predicted: {pred_class} (Confidence: {confidence:.2f}%)")

# Save confusion matrix with better visualization
plt.figure(figsize=(10, 8))
cm = confusion_matrix(true_classes, predicted_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_labels, 
            yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
conf_matrix_path = os.path.join(analysis_dir, 'confusion_matrix.png')
plt.savefig(conf_matrix_path)
plt.close()

# Save model summary
def save_model_summary(model, filepath):
    # 使用 StringIO 来捕获摘要
    summary_string = StringIO()
    model.summary(print_fn=lambda x: summary_string.write(x + '\n'))
    
    # 使用 utf-8 编码写入文件
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(summary_string.getvalue())
    except Exception as e:
        print(f"Warning: Could not save model summary due to: {e}")
        # 如果保存失败，至少打印出来
        print("\nModel Summary:")
        model.summary()

# 修改保存训练历史的部分
def save_training_history(history, filepath):
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history.history, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Could not save training history due to: {e}")

# 在训练后使用这些函数
try:
    save_model_summary(model, os.path.join(analysis_dir, 'model_summary.txt'))
    save_training_history(history_1, os.path.join(analysis_dir, 'training_history_first_10_epochs.json'))
    save_training_history(history_2, os.path.join(analysis_dir, 'training_history_last_20_epochs.json'))
except Exception as e:
    print(f"Warning: Error in saving files: {e}")

def predict_single_image(model, image_path):
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=img_size
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    return predicted_class, confidence

# Example usage:
# test_image_path = "path/to/test/image.png"
# predicted_class, confidence = predict_single_image(model, test_image_path)
# print(f"Predicted class: {predicted_class} with confidence: {confidence:.2f}%")

# 修改数据生成器以加载共振峰数据
class VowelDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory, batch_size=32, shuffle=True):
        self.directory = directory
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_files = []
        self.formant_files = []
        self.labels = []
        
        # 加载数据
        for label in os.listdir(directory):
            label_dir = os.path.join(directory, label)
            if os.path.isdir(label_dir):
                for file in os.listdir(label_dir):
                    if file.endswith('.png'):
                        self.image_files.append(os.path.join(label_dir, file))
                        formant_file = os.path.join(label_dir, 
                                      file.replace('.png', '_formants.txt'))
                        self.formant_files.append(formant_file)
                        self.labels.append(label)
        
        self.indices = np.arange(len(self.image_files))
        self.on_epoch_end()
    
    def __len__(self):
        return len(self.image_files) // self.batch_size
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:
                                   (idx + 1) * self.batch_size]
        
        # 加载图像
        batch_images = []
        batch_formants = []
        batch_labels = []
        
        for i in batch_indices:
            # 加载图像
            img = tf.keras.preprocessing.image.load_img(
                self.image_files[i],
                target_size=(55, 55)
            )
            img = tf.keras.preprocessing.image.img_to_array(img)
            batch_images.append(img)
            
            # 加载共振峰数据
            with open(self.formant_files[i], 'r') as f:
                formants = list(map(float, f.read().split(',')))
            batch_formants.append(formants)
            
            # 处理标签
            batch_labels.append(self.labels[i])
        
        return [np.array(batch_images), np.array(batch_formants)], \
               tf.keras.utils.to_categorical(batch_labels)
