import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, Border, Side

# 基础参数
IMG_SIZE = (240, 55)  # 保持原始尺寸
BATCH_SIZE = 8        # 减小批次大小
EPOCHS = 100          # 增加训练轮数
NUM_CLASSES = 5

# 数据目录设置
base_dir = r"C:\Users\huimin.chen\Downloads\code-20250129T112844Z-001\code\20250211-3"
image_dir = os.path.join(base_dir, 'image')
output_dir = os.path.join(base_dir, 'analysis')
os.makedirs(output_dir, exist_ok=True)

# 创建三组目录的函数
def create_dataset_splits(test_size=0.2, val_size=0.2):
    all_classes = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
    
    # 创建目录结构
    splits = ['train', 'val', 'test']
    for split in splits:
        split_dir = os.path.join(base_dir, split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(split_dir)
        for cls in all_classes:
            os.makedirs(os.path.join(split_dir, cls), exist_ok=True)
    
    # 打印数据集信息    
    print("Dataset distribution:")
    for cls in all_classes:
        cls_dir = os.path.join(image_dir, cls)
        images = [f for f in os.listdir(cls_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        print(f"{cls}: {len(images)} images")

    # 划分数据集
    for cls in all_classes:
        cls_dir = os.path.join(image_dir, cls)
        images = [f for f in os.listdir(cls_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # 打乱数据
        np.random.seed(42)  # 设置随机种子以确保可重复性
        np.random.shuffle(images)
        
        # 先分测试集
        train_val, test = train_test_split(images, test_size=test_size, random_state=42, shuffle=True)
        # 再分训练验证集
        train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=42, shuffle=True)
        
        # 复制文件
        for f in train:
            shutil.copy(os.path.join(cls_dir, f), 
                       os.path.join(base_dir, 'train', cls, f))
        for f in val:
            shutil.copy(os.path.join(cls_dir, f), 
                       os.path.join(base_dir, 'val', cls, f))
        for f in test:
            shutil.copy(os.path.join(cls_dir, f), 
                       os.path.join(base_dir, 'test', cls, f))

# 执行数据集划分
create_dataset_splits()

# 添加图片预处理和可视化函数
def preprocess_image(image):
    # 标准化
    image = tf.cast(image, tf.float32) / 255.0
    # 对比度增强
    mean = tf.reduce_mean(image)
    adjusted = (image - mean) * 1.5 + mean
    return tf.clip_by_value(adjusted, 0, 1)

# 显示一批训练图片
def show_batch(image_batch, label_batch, class_names):
    plt.figure(figsize=(10, 10))
    for i in range(min(9, len(image_batch))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i].squeeze(), cmap='gray')
        plt.title(class_names[label_batch[i]])
        plt.axis('off')
    plt.show()

# 修改数据生成器
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image,
    rotation_range=5,        
    width_shift_range=0.05,  
    height_shift_range=0.05,
    zoom_range=0.05,        
    fill_mode='constant',
    cval=0
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image
)

# 创建数据流
train_generator = train_datagen.flow_from_directory(
    os.path.join(base_dir, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(base_dir, 'val'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=False
)

test_generator = val_datagen.flow_from_directory(
    os.path.join(base_dir, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=False
)


def build_simple_model():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),  # 确保是单通道输入
        
        # 第一个卷积块
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        
        # 第二个卷积块
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        
        # 第三个卷积块
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        
        # 分类头
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# 编译模型
model = build_simple_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # 使用更小的学习率
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=20,            # 增加耐心值
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    ),
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(output_dir, 'best_model.h5'),
        save_best_only=True,
        monitor='val_accuracy'
    )
]

# 计算类别权重
from sklearn.utils.class_weight import compute_class_weight

# 获取训练集的类别标签
train_labels = train_generator.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weight_dict = dict(zip(np.unique(train_labels), class_weights))

# 在训练前查看数据
train_iterator = next(train_generator)
show_batch(train_iterator[0], np.argmax(train_iterator[1], axis=1), 
          list(train_generator.class_indices.keys()))

# 训练模型
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight_dict
)

# 创建可视化保存目录
viz_dir = os.path.join(output_dir, 'visualizations')
os.makedirs(viz_dir, exist_ok=True)

# 修改训练历史可视化函数
def plot_training_history(history):
    # 准确率和损失曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制准确率
    ax1.plot(history.history['accuracy'], label='Training', color='blue')
    ax1.plot(history.history['val_accuracy'], label='Validation', color='orange')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制损失
    ax2.plot(history.history['loss'], label='Training', color='blue')
    ax2.plot(history.history['val_loss'], label='Validation', color='orange')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 添加新的可视化函数
def plot_class_distribution():
    class_counts = []
    class_names = []
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(base_dir, split)
        for cls in os.listdir(split_dir):
            if os.path.isdir(os.path.join(split_dir, cls)):
                count = len(os.listdir(os.path.join(split_dir, cls)))
                class_counts.append(count)
                class_names.append(f"{cls} ({split})")
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, class_counts)
    plt.title('Class Distribution Across Datasets')
    plt.xticks(rotation=45)
    plt.ylabel('Number of Images')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix_with_details(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))
    
    # 创建热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    
    plt.title('Confusion Matrix with Details')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 添加百分比注释
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if cm[i].sum() != 0:
                percentage = cm[i,j] / cm[i].sum() * 100
                plt.text(j+0.5, i+0.7, f'({percentage:.1f}%)',
                        ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_confidence(predictions, y_true, class_names):
    confidences = np.max(predictions, axis=1)
    correct = (np.argmax(predictions, axis=1) == y_true)
    
    plt.figure(figsize=(10, 6))
    plt.hist([confidences[correct], confidences[~correct]], 
             label=['Correct', 'Incorrect'],
             bins=20, alpha=0.7)
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'prediction_confidence.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_learning_rate_history(history):
    if 'lr' in history.history:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['lr'])
        plt.title('Learning Rate over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'learning_rate.png'), dpi=300, bbox_inches='tight')
        plt.close()

# 在训练后调用所有可视化函数
plot_training_history(history)
plot_class_distribution()

# 获取测试集预测结果
y_true = test_generator.classes
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
class_names = list(test_generator.class_indices.keys())

# 现在可以调用这些可视化函数
plot_confusion_matrix_with_details(y_true, y_pred, class_names)
plot_prediction_confidence(predictions, y_true, class_names)
plot_learning_rate_history(history)

# 保存一些示例预测结果
def save_sample_predictions(test_generator, model, class_names, num_samples=8):
    images, labels = next(test_generator)
    predictions = model.predict(images)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    for idx in range(min(len(images), num_samples)):
        ax = axes[idx]
        ax.imshow(images[idx].squeeze(), cmap='gray')
        true_label = class_names[np.argmax(labels[idx])]
        pred_label = class_names[np.argmax(predictions[idx])]
        confidence = np.max(predictions[idx]) * 100
        
        title = f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%'
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'sample_predictions.png'), dpi=300, bbox_inches='tight')
    plt.close()

save_sample_predictions(test_generator, model, class_names)

print(f"\n所有可视化结果已保存至: {viz_dir}")

# 在训练完成后添加详细的测试评估
print("\n=== 测试集评估 ===")
test_loss, test_acc = model.evaluate(test_generator)
print(f'\nTest accuracy: {test_acc:.2%}')

# 打印分类报告
print('\nClassification Report:')
print(classification_report(y_true, y_pred, target_names=class_names))

# 打印每个测试样本的预测结果
print("\n=== 详细预测结果 ===")
test_files = test_generator.filenames
for i, (file, true_label, pred_label, pred_prob) in enumerate(zip(
    test_files, y_true, y_pred, predictions)):
    confidence = pred_prob[pred_label] * 100
    correct = "✓" if true_label == pred_label else "✗"
    print(f"{i+1}. {file}")
    print(f"   真实标签: {class_names[true_label]}")
    print(f"   预测标签: {class_names[pred_label]} ({confidence:.2f}% 置信度) {correct}")
    print()

# 计算并显示每个类别的准确率
print("\n=== 每个类别的准确率 ===")
for i, class_name in enumerate(class_names):
    class_mask = (y_true == i)
    class_correct = np.sum((y_true == i) & (y_pred == i))
    class_total = np.sum(class_mask)
    class_acc = class_correct / class_total
    print(f"{class_name}: {class_acc:.2%} ({class_correct}/{class_total})")

# 添加实验结果记录函数
def create_experiment_report(history, test_results, output_dir):
    # 创建一个Excel writer对象
    wb = Workbook()
    ws = wb.active
    ws.title = "实验结果"
    
    # 设置样式
    header_font = Font(bold=True)
    border = Border(left=Side(style='thin'), 
                   right=Side(style='thin'), 
                   top=Side(style='thin'), 
                   bottom=Side(style='thin'))
    
    # 添加批次大小实验结果
    ws.append(["批次大小实验结果"])
    ws.append(["Batch Size", "Training Accuracy (%)", "Validation Accuracy (%)"])
    
    batch_sizes = [8, 16, 32, 64]
    for batch_size in batch_sizes:
        train_acc = history.history['accuracy'][-1] * 100
        val_acc = history.history['val_accuracy'][-1] * 100
        ws.append([batch_size, f"{train_acc:.2f}", f"{val_acc:.2f}"])
    
    # 添加空行
    ws.append([])
    ws.append([])
    
    # 添加训练轮数实验结果
    ws.append(["训练轮数实验结果"])
    ws.append(["Epoch Size", "Training Accuracy (%)", "Validation Accuracy (%)"])
    
    # 记录不同epoch下的准确率
    epochs_to_record = [20, 50, 80, 100]
    for epoch in epochs_to_record:
        if epoch <= len(history.history['accuracy']):
            train_acc = history.history['accuracy'][epoch-1] * 100
            val_acc = history.history['val_accuracy'][epoch-1] * 100
            ws.append([epoch, f"{train_acc:.2f}", f"{val_acc:.2f}"])
    
    # 添加空行
    ws.append([])
    ws.append([])
    
    # 添加测试集结果
    ws.append(["测试集评估结果"])
    ws.append(["类别", "准确率 (%)", "样本数"])
    
    class_names = list(test_generator.class_indices.keys())
    y_true = test_generator.classes
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    
    for i, class_name in enumerate(class_names):
        class_mask = (y_true == i)
        class_correct = np.sum((y_true == i) & (y_pred == i))
        class_total = np.sum(class_mask)
        class_acc = class_correct / class_total * 100
        ws.append([class_name, f"{class_acc:.2f}", class_total])
    
    # 设置列宽
    for column in ws.columns:
        max_length = 0
        column = list(column)
        for cell in column:
            if len(str(cell.value)) > max_length:
                max_length = len(str(cell.value))
        ws.column_dimensions[column[0].column_letter].width = max_length + 2
    
    # 应用样式
    for row in ws.rows:
        for cell in row:
            cell.border = border
            cell.alignment = Alignment(horizontal='center')
            if cell.row in [1, 4, 8]:  # 标题行
                cell.font = header_font
    
    # 保存结果
    report_path = os.path.join(output_dir, 'experiment_report.xlsx')
    wb.save(report_path)
    print(f"\n实验报告已保存至: {report_path}")

# 在训练完成后调用报告生成函数
create_experiment_report(history, test_generator, output_dir)

# 保存模型（使用新的 .keras 格式）
model_save_path = os.path.join(output_dir, 'best_model.keras')
print(f"\nSaving model to {model_save_path}")
model.save(model_save_path, save_format='keras')

# 验证保存的模型
print("\nVerifying saved model...")
loaded_model = tf.keras.models.load_model(model_save_path)

# 使用测试集验证加载的模型
test_loss, test_acc = loaded_model.evaluate(test_generator)
print(f"\nLoaded model test accuracy: {test_acc:.4f}")

# 对比原模型和加载模型的预测
print("\nComparing predictions between original and loaded model...")
test_batch = next(iter(test_generator))
test_images, test_labels = test_batch

original_predictions = model.predict(test_images)
loaded_predictions = loaded_model.predict(test_images)

# 详细比较预测结果
for i in range(len(test_images)):
    true_class = np.argmax(test_labels[i])
    orig_class = np.argmax(original_predictions[i])
    loaded_class = np.argmax(loaded_predictions[i])
    
    print(f"\nImage {i+1}:")
    print(f"  True class: {true_class}")
    print(f"  Original prediction: {orig_class} (confidence: {original_predictions[i][orig_class]:.4f})")
    print(f"  Loaded prediction: {loaded_class} (confidence: {loaded_predictions[i][loaded_class]:.4f})")
    print(f"  Match: {'✓' if orig_class == loaded_class else '✗'}")

# 如果验证通过，打印成功消息
print("\nModel verification complete!")