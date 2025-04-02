import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import parselmouth
from parselmouth.praat import call

def create_output_dirs(output_dir, classes):
    for label in classes:
        os.makedirs(os.path.join(output_dir, label), exist_ok=True)


def extract_formants(audio_path, max_formant=5500, num_formants=5):
    sound = parselmouth.Sound(audio_path)
    formant = call(sound, "To Formant (burg)", 0.0, num_formants, 
                  max_formant, 0.025, 50)
    
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
    y, sr = librosa.load(audio_path, sr=None)
    
    formants = extract_formants(audio_path)
    
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
    

    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
    

    plt.figure(figsize=(0.55, 2.4))  

    librosa.display.specshow(
        mel_spect_db,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='log',  
        cmap='magma'
    )
    

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    

    output_path = os.path.join(output_dir, label, f"{file_name}.png")
    plt.savefig(output_path,
                dpi=100,
                bbox_inches='tight',
                pad_inches=0,
                format='png')
    plt.close()


def process_audio_files(audio_dir, output_dir, classes):

    create_output_dirs(output_dir, classes)
    

    for label in classes:
        class_dir = os.path.join(audio_dir, label)
        if not os.path.isdir(class_dir):
            print(f"Warning: Category directory '{class_dir}' does not exist, skipping processing.")
            continue
        for i, file in enumerate(os.listdir(class_dir)):
            if file.lower().endswith(".wav"):  
                file_path = os.path.join(class_dir, file)
                save_spectrogram_with_formants(file_path, output_dir, label, f"{label}_{i+1}")
            else:
                print(f"Warning: File '{file}' is not in .wav format, skipped.")


audio_base_dir = r"C:\Users\huimin.chen\Downloads\code-20250129T112844Z-001\code\20250211-3\audio"
output_base_dir = r"C:\Users\huimin.chen\Downloads\code-20250129T112844Z-001\code\20250211-3\image"
vowel_classes = ["A", "E", "I", "O", "U"]  # Vowel categories

process_audio_files(audio_base_dir, output_base_dir, vowel_classes)