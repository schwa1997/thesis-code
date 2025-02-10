import os
import numpy as np
import librosa
import soundfile as sf

def generate_italian_vowel(vowel, duration=1.0, sample_rate=44100, fundamental_freq=220):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Define formant frequencies for each vowel
    formants = {
        'a': [800, 1150, 2800, 3500, 4950],
        'e': [400, 1600, 2700, 3300, 4950],
        'i': [240, 1940, 2750, 3300, 4950],
        'o': [400, 750, 2400, 3300, 4950],
        'u': [250, 595, 2400, 3300, 4950]
    }
    
    # Generate the fundamental frequency
    signal = np.zeros_like(t)
    
    # Add harmonics with formant-based amplitudes
    for i in range(1, 20):  # Increase the number of harmonics
        freq = fundamental_freq * i
        amplitude = 1 / i  # Base amplitude decreases with harmonic number
        
        # Adjust amplitude based on proximity to formants
        for formant in formants[vowel]:
            amplitude *= 1 + 50 / (1 + ((freq - formant) / 50)**2)
        
        signal += amplitude * np.sin(2 * np.pi * freq * t)
    
    # Apply a more realistic envelope
    envelope = np.hanning(len(t)) ** 0.5  # Softer attack and decay
    signal *= envelope
    
    # Normalize
    signal *= 0.3 / np.max(np.abs(signal))
    
    return signal

def save_audio(signal, filename, sample_rate=44100):
    sf.write(filename, signal, sample_rate)

def create_variations(base_signal, sample_rate=44100):
    variations = []
    
    # Pitch variation
    variations.append(librosa.effects.pitch_shift(base_signal, sr=sample_rate, n_steps=2))
    variations.append(librosa.effects.pitch_shift(base_signal, sr=sample_rate, n_steps=-2))
    
    # Duration variation
    variations.append(librosa.effects.time_stretch(base_signal, rate=1.2))
    variations.append(librosa.effects.time_stretch(base_signal, rate=0.8))
    
    # Loudness variation
    variations.append(base_signal * 0.7)
    variations.append(np.clip(base_signal * 1.3, -1, 1))
    
    return variations

# List of Italian vowels
vowels = ['a', 'e', 'i', 'o', 'u']

for vowel in vowels:
    # Create the vowel folder if it doesn't exist
    os.makedirs(vowel.upper(), exist_ok=True)

    # Generate base Italian vowel
    base_vowel = generate_italian_vowel(vowel)

    # Save the base vowel
    save_audio(base_vowel, f'{vowel.upper()}/base_italian_{vowel}.wav')

    # Create and save variations
    variations = create_variations(base_vowel)
    for i, var in enumerate(variations):
        save_audio(var, f'{vowel.upper()}/italian_{vowel}_variation_{i+1}.wav')

    print(f"Italian '{vowel}' vowel and its variations have been generated and saved in the '{vowel.upper()}' folder.")
