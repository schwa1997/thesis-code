import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks  # Add this import

def create_output_dirs(output_dir, classes):
    for label in classes:
        os.makedirs(os.path.join(output_dir, label), exist_ok=True)

# Convert audio file to spectrogram and save
def save_spectrogram(audio_path, output_dir, label, file_name):
    y, sr = librosa.load(audio_path, sr=None)
    
    # Calculate short-time Fourier transform
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    
    # Create image
    plt.figure(figsize=(12, 8))
    
    # Plot spectrogram with log frequency scale
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(img, format='%+2.0f dB')
    plt.title('Spectrogram')
    
    plt.tight_layout()
    
    # Save image
    output_path = os.path.join(output_dir, label, f"{file_name}.png")
    plt.savefig(output_path, dpi=100)  # Adjust DPI to get desired image size
    plt.close()

    print(f"Saved spectrogram: {output_path}")
    print("Please manually crop the image to 240x55 pixels, focusing on the vowel transition from /a/ to /o/.")

# Main process
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
                file_name, _ = os.path.splitext(file)
                save_spectrogram(file_path, output_dir, label, f"{file_name}_{i+1}")
            else:
                print(f"Warning: File '{file}' is not in .wav format, skipped.")

# Define audio file categories and paths
audio_test_dir = r"C:\Users\huimin.chen\Downloads\thesis-code-20241023T072444Z-001\thesis-code\audio\test"
output_test_dir = r"C:\Users\huimin.chen\Downloads\thesis-code-20241023T072444Z-001\thesis-code\audio\output\test"

# Get category list
classes = [d for d in os.listdir(audio_test_dir) if os.path.isdir(os.path.join(audio_test_dir, d))]

# Execute audio file conversion
process_audio_files(audio_test_dir, output_test_dir, classes)
