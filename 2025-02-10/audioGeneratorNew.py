import os
import numpy as np
from scipy.signal import lfilter
from pydub import AudioSegment
import pyttsx3

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

# Generate and export vowel variations
duration = 1000  # milliseconds
num_variations = 400  # Increased to 100 variations
vowel_folders = ['A', 'E', 'I', 'O', 'U']

for vowel in vowel_folders:
    os.makedirs(vowel, exist_ok=True)
    for variation in range(num_variations):
        engine, vowel_sound = generate_vowel(vowel, duration, variation)
        filename = f"{vowel}/{vowel}_var{variation}.wav"
        engine.save_to_file(vowel_sound, filename)
        engine.runAndWait()
        print(f"Generated {filename}")