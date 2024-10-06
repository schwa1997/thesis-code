import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 创建存储路径，如果不存在则创建
def create_output_dirs(base_dir, classes):
    for class_label in classes:
        dir_path = os.path.join(base_dir, class_label)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

# 将音频文件转换为梅尔频谱图并保存
def save_mel_spectrogram(audio_path, output_dir, label, file_name):
    y, sr = librosa.load(audio_path, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel-Spectrogram - {label}')
    plt.tight_layout()

    # 保存图像
    output_path = os.path.join(output_dir, label, f"{file_name}.png")
    plt.savefig(output_path)
    plt.close()

# 主要流程
def process_audio_files(audio_dir, output_dir, classes):
    # 创建类别目录
    create_output_dirs(output_dir, classes)
    
    # 遍历每个类别
    for label in classes:
        class_dir = os.path.join(audio_dir, label)
        for i, file in enumerate(os.listdir(class_dir)):
            if file.endswith(".wav"):  # 这里只处理 .wav 文件
                file_path = os.path.join(class_dir, file)
                save_mel_spectrogram(file_path, output_dir, label, f"{label}_{i+1}")

# 定义音频文件的类别和路径
audio_base_dir = r"C:\Users\huimin.chen\projects\chm-vowel\vowelRecognition\audio"  # 使用原始字符串避免转义
output_base_dir = r"C:\Users\huimin.chen\projects\chm-vowel\vowelRecognition\output"  # 使用原始字符串避免转义
vowel_classes = ["A", "E", "I", "O", "U"]  # 元音类别

# 执行音频文件转换
process_audio_files(audio_base_dir, output_base_dir, vowel_classes)
