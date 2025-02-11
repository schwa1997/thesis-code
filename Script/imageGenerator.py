import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import parselmouth
from parselmouth.praat import call

# ... (keep the create_output_dirs function as is) ...
def create_output_dirs(output_dir, classes):
    for label in classes:
        os.makedirs(os.path.join(output_dir, label), exist_ok=True)

# ... (rest of the existing code remains unchanged) ...

def extract_formants(audio_path, max_formant=5500, num_formants=5):
    """提取共振峰频率"""
    sound = parselmouth.Sound(audio_path)
    formant = call(sound, "To Formant (burg)", 0.0, num_formants, 
                  max_formant, 0.025, 50)
    
    # 获取中间点的共振峰
    t = sound.duration / 2
    formants = []
    for i in range(1, num_formants + 1):
        try:
            formant_freq = call(formant, "Get value at time", 
                              i, t, 'Hertz', 'Linear')
            formants.append(formant_freq)
        except:
            formants.append(0)
    return formants

def save_spectrogram_with_formants(audio_path, output_dir, label, file_name):
    # 加载音频
    y, sr = librosa.load(audio_path, sr=None)
    
    # 提取共振峰
    formants = extract_formants(audio_path)
    
    # 计算梅尔频谱图
    n_fft = 2048
    hop_length = 512
    mel_spect = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=128,
        fmin=20,
        fmax=8000
    )
    
    # 转换为分贝单位
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
    
    # 创建图像 - 调整大小为 55x240 像素
    plt.figure(figsize=(0.55, 2.4))  # 假设 DPI=100，则 0.55x2.4 英寸 = 55x240 像素
    
    # 绘制频谱图 - 使用对数频率刻度
    librosa.display.specshow(
        mel_spect_db,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='log',  # 改为对数频率刻度
        cmap='magma'
    )
    
    # 在频谱图上标记共振峰
   # times = np.linspace(0, librosa.get_duration(y=y, sr=sr), mel_spect.shape[1])
    #for formant_freq in formants:
    #    if formant_freq > 0:
     #       plt.axhline(y=formant_freq, color='cyan', alpha=0.3, linestyle='--')
    
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    
    # 保存图像
    output_path = os.path.join(output_dir, label, f"{file_name}.png")
    plt.savefig(output_path,
                dpi=100,
                bbox_inches='tight',
                pad_inches=0,
                format='png')
    plt.close()

    # 保存共振峰数据
    #formants_path = os.path.join(output_dir, label, f"{file_name}_formants.txt")
    #with open(formants_path, 'w') as f:
       # f.write(','.join(map(str, formants)))

def process_audio_files(audio_dir, output_dir, classes):
    # Create category directories
    create_output_dirs(output_dir, classes)
    
    # Iterate through each category
    for label in classes:
        class_dir = os.path.join(audio_dir, label)
        if not os.path.isdir(class_dir):
            print(f"Warning: Category directory '{class_dir}' does not exist, skipping processing.")
            continue
        for i, file in enumerate(os.listdir(class_dir)):
            if file.lower().endswith(".wav"):  # Use lower() to ignore case
                file_path = os.path.join(class_dir, file)
                save_spectrogram_with_formants(file_path, output_dir, label, f"{label}_{i+1}")
            else:
                print(f"Warning: File '{file}' is not in .wav format, skipped.")

# Define audio file categories and paths
audio_base_dir = r"C:\Users\huimin.chen\Downloads\code-20250129T112844Z-001\code\20250211-3\audio"
output_base_dir = r"C:\Users\huimin.chen\Downloads\code-20250129T112844Z-001\code\20250211-3\image"
vowel_classes = ["A", "E", "I", "O", "U"]  # Vowel categories

# Execute audio file conversion
process_audio_files(audio_base_dir, output_base_dir, vowel_classes)