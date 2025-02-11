import os
import numpy as np
from scipy.signal import lfilter
from pydub import AudioSegment
import pyttsx3
import soundfile as sf
from scipy.signal import butter, filtfilt

def get_italian_voice():
    """
    Find an Italian voice from available system voices.
    """
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    for voice in voices:
        if 'italian' in voice.name.lower() or 'it' in voice.id.lower():
            return voice.id
    return None

def generate_vowel(vowel, duration=1000, variation=0):
    """
    Generate an Italian vowel sound using text-to-speech synthesis with significant variations.
    :param vowel: The vowel character ('A', 'E', 'I', 'O', 'U')
    :param duration: Duration in milliseconds
    :param variation: Variation index for differentiation
    :return: AudioSegment object
    """
    engine = pyttsx3.init()
    italian_voice = get_italian_voice()
    if italian_voice:
        engine.setProperty('voice', italian_voice)
    else:
        print("Warning: No Italian voice found. Using default voice.")
    
    # Expanded variations without pitch (SAPI5 compatible)
    rates = [60, 80, 100, 120, 140, 160, 180, 200, 220, 240]  # More speed variations
    volumes = [0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]   # More volume variations
    
    # Use modulo to cycle through variations
    rate_idx = variation % len(rates)
    volume_idx = (variation // len(rates)) % len(volumes)
    
    engine.setProperty('rate', rates[rate_idx])
    engine.setProperty('volume', volumes[volume_idx])
    
    return engine, vowel

def add_noise_and_filter(audio_file, noise_level=0.005, cutoff=3000):
    """
    Add noise and apply a low-pass filter to an audio file.
    """
    data, samplerate = sf.read(audio_file)
    noise = np.random.normal(0, noise_level, data.shape)
    data_noisy = data + noise

    nyquist = 0.5 * samplerate
    normal_cutoff = cutoff / nyquist
    b, a = butter(6, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data_noisy)
    
    sf.write(audio_file, filtered_data, samplerate)

def change_pitch(audio_file, semitones):
    """
    Change pitch of an audio file by a given number of semitones.
    """
    sound = AudioSegment.from_file(audio_file, format="wav")
    new_sample_rate = int(sound.frame_rate * (2.0 ** (semitones / 12.0)))
    return sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate}).set_frame_rate(44100)

# Generate and export vowel variations
duration = 1000  # milliseconds
num_variations = 200  # Increased to 800 variations
vowel_folders = ['A', 'E', 'I', 'O', 'U']
pitch_variations = [-2, -1, 0, 1, 2]  # Semitones to shift

for vowel in vowel_folders:
    os.makedirs(vowel, exist_ok=True)
    for variation in range(num_variations):
        engine, vowel_sound = generate_vowel(vowel, duration, variation)
        base_filename = f"{vowel}/{vowel}_vars{variation}.wav"
        engine.save_to_file(vowel_sound, base_filename)
        engine.runAndWait()
        
        # Add noise and filter
       # add_noise_and_filter(base_filename)
        
        # Apply pitch variations
        for semitones in pitch_variations:
            if semitones != 0:  # Skip pitch change for original pitch
                pitched_audio = change_pitch(base_filename, semitones)
                pitch_filename = f"{vowel}/{vowel}_varr{variation}_pitch{semitones}.wav"
                pitched_audio.export(pitch_filename, format="wav")
        
        print(f"Generated {base_filename} with variations")