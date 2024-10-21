import os
import numpy as np
import librosa
import soundfile as sf

def generate_italian_a_vowel(duration=1.0, sample_rate=44100, fundamental_freq=220):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Generate the fundamental frequency
    signal = np.sin(2 * np.pi * fundamental_freq * t)
    
    # Add harmonics (overtones) to create a more realistic vowel sound
    for i in range(2, 6):
        signal += 0.5 / i * np.sin(2 * np.pi * i * fundamental_freq * t)
    
    # Apply an envelope to smooth the start and end
    envelope = np.ones_like(signal)
    attack = int(0.01 * sample_rate)
    release = int(0.01 * sample_rate)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-release:] = np.linspace(1, 0, release)
    
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

# Create the 'A' folder if it doesn't exist
os.makedirs('A', exist_ok=True)

# Generate base Italian 'a' vowel
base_vowel = generate_italian_a_vowel()

# Save the base vowel
save_audio(base_vowel, 'A/base_italian_a.wav')

# Create and save variations
variations = create_variations(base_vowel)
for i, var in enumerate(variations):
    save_audio(var, f'A/italian_a_variation_{i+1}.wav')

print("Italian 'a' vowel and its variations have been generated and saved in the 'A' folder.")
