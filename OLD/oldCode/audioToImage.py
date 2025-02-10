import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Create storage path, create if it doesn't exist
def create_output_dirs(base_dir, classes):
    for class_label in classes:
        dir_path = os.path.join(base_dir, class_label)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

# Convert audio file to mel spectrogram and save
def save_mel_spectrogram(audio_path, output_dir, label, file_name):
    y, sr = librosa.load(audio_path, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel-Spectrogram - {label}')
    plt.tight_layout()

    # Save image
    output_path = os.path.join(output_dir, label, f"{file_name}.png")
    plt.savefig(output_path)
    plt.close()

# Main process
def process_audio_files(audio_dir, output_dir, classes):
    # Create category directories
    create_output_dirs(output_dir, classes)
    
    # Iterate through each category
    for label in classes:
        class_dir = os.path.join(audio_dir, label)
        for i, file in enumerate(os.listdir(class_dir)):
            if file.endswith(".wav"):  # Only process .wav files here
                file_path = os.path.join(class_dir, file)
                save_mel_spectrogram(file_path, output_dir, label, f"{label}_{i+1}")

# Define audio file categories and paths
audio_base_dir = r"C:\Users\huimin.chen\projects\chm-vowel\vowelRecognition\audio"  # Use raw string to avoid escaping
output_base_dir = r"C:\Users\huimin.chen\projects\chm-vowel\vowelRecognition\output"  # Use raw string to avoid escaping
vowel_classes = ["A", "E", "I", "O", "U"]  # Vowel categories

# Execute audio file conversion
process_audio_files(audio_base_dir, output_base_dir, vowel_classes)
