from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing import image
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 构建迁移学习模型（ResNet50）
def build_transfer_model(input_shape, num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # 冻结预训练层

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# 数据准备
train_datagen = ImageDataGenerator(
    rescale=1./255,  # 图像归一化
    validation_split=0.3,  # 将数据分为训练集和验证集
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    r'C:\Users\huimin.chen\projects\chm-vowel\vowelRecognition\output',  # 图像存储路径
    target_size=(128, 128),  # 调整图像尺寸以匹配模型输入
    batch_size=32,
    class_mode='categorical',
    subset='training'  # 使用训练集
)

validation_generator = train_datagen.flow_from_directory(
    r'C:\Users\huimin.chen\projects\chm-vowel\vowelRecognition\output',  # 同一图像路径
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # 使用验证集
)

# 获取类别标签
class_indices = train_generator.class_indices
class_labels = {v: k for k, v in class_indices.items()}  # 保持大写

print("训练集类别映射 (整数 -> 类别名称):", class_labels)
print("验证集类别映射 (整数 -> 类别名称):", {v: k for k, v in validation_generator.class_indices.items()})

# 检查测试集类别是否在 class_labels 中
test_classes = ['A', 'E', 'I', 'O', 'U']
for cls in test_classes:
    if cls not in class_labels.values():
        print(f"警告: 测试集类别 '{cls}' 不在训练集类别中。请检查类别名称是否一致。")

# 计算类别权重以处理数据不平衡
train_labels = train_generator.classes
class_weights_values = class_weight.compute_class_weight('balanced',
                                                          classes=np.unique(train_labels),
                                                          y=train_labels)
class_weights = {i: weight for i, weight in enumerate(class_weights_values)}
print("类别权重:", class_weights)

# 构建迁移学习模型
model = build_transfer_model(input_shape=(128, 128, 3), num_classes=len(class_indices))

# 定义回调函数
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)
lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=5, factor=0.5, min_lr=1e-6, verbose=1)

# 训练迁移学习模型
history = model.fit(
    train_generator,
    epochs=50,  # 增加训练轮数
    validation_data=validation_generator,
    class_weight=class_weights,
    callbacks=[early_stopping, lr_reduction]
)

# 可视化训练过程
def plot_history(history):
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
    
    plt.show()

plot_history(history)

# 使用模型进行预测
def predict_vowel(image_path, model, class_labels):
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # 增加一个批次维度
    img_array /= 255.0  # 归一化

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_label = class_labels.get(predicted_index, "Unknown")  # 使用 get 方法避免 KeyError

    return predicted_label

# 评估测试集
def evaluate_test_set(model, class_labels, test_dir):
    correct = 0
    total = 0
    per_class_correct = {label: 0 for label in class_labels.values()}
    per_class_total = {label: 0 for label in class_labels.values()}
    y_true = []
    y_pred = []

    for class_name in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_path):
            continue  # 忽略非文件夹
        for file in os.listdir(class_path):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):  # 支持多种图片格式
                image_path = os.path.join(class_path, file)
                predicted_class = predict_vowel(image_path, model, class_labels)
                true_class = class_name.upper()  # 确保类别名称为大写

                if true_class not in per_class_total:
                    print(f"警告: 测试集中发现未定义的类别 '{true_class}'。")
                    continue  # 跳过未定义的类别

                # 更新统计
                total += 1
                per_class_total[true_class] += 1
                if predicted_class == true_class:
                    correct += 1
                    per_class_correct[true_class] += 1

                y_true.append(true_class)
                y_pred.append(predicted_class)

    # 计算整体准确率
    overall_accuracy = correct / total * 100 if total > 0 else 0
    print(f"总体正确预测数: {correct} / {total}")
    print(f"总体准确率: {overall_accuracy:.2f}%\n")

    # 计算每个类别的准确率
    print("每个类别的分类准确率:")
    for label in per_class_total:
        if per_class_total[label] > 0:
            accuracy = per_class_correct[label] / per_class_total[label] * 100
            print(f"  类别 '{label}': {per_class_correct[label]} / {per_class_total[label]} = {accuracy:.2f}%")
        else:
            print(f"  类别 '{label}': 无测试样本")

    # 混淆矩阵和分类报告
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=sorted(class_labels.values())))

    cm = confusion_matrix(y_true, y_pred, labels=sorted(class_labels.values()))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(class_labels.values()),
                yticklabels=sorted(class_labels.values()))
    plt.ylabel('real label')
    plt.xlabel('prediction label')
    plt.title('matrix')
    plt.show()

# 定义测试集路径
test_dir = r'C:\Users\huimin.chen\projects\chm-vowel\vowelRecognition\test'

# 评估测试集
evaluate_test_set(model, class_labels, test_dir)
