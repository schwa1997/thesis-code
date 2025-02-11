import os
import asyncio
import edge_tts
import numpy as np
import soundfile as sf
from pydub import AudioSegment

async def generate_vowel_edge(vowel, variation=0):
    """
    Generate an Italian vowel sound using Edge TTS with variations.
    :param vowel: The vowel character ('A', 'E', 'I', 'O', 'U')
    :param variation: Variation index for differentiation
    :return: Path to temporary audio file
    """
    # Only female Italian voices
    voices = [
        "it-IT-ElsaNeural",       # Female voice 1
        "it-IT-IsabellaNeural",   # Female voice 2
    ]
    
    # Variation parameters
    rates = ["-20%", "-10%", "0%", "+10%", "+20%"]
    pitches = ["-100Hz", "-50Hz", "0Hz", "+50Hz", "+100Hz"]
    volumes = ["-20%", "-10%", "0%", "+10%", "+20%"]
    
    # Use variation to cycle through parameters
    voice_idx = variation % len(voices)
    rate_idx = (variation // len(voices)) % len(rates)
    pitch_idx = (variation // (len(voices) * len(rates))) % len(pitches)
    volume_idx = (variation // (len(voices) * len(rates) * len(pitches))) % len(volumes)
    
    communicate = edge_tts.Communicate(
        vowel,
        voices[voice_idx],
        rate=rates[rate_idx],
        volume=volumes[volume_idx],
        pitch=pitches[pitch_idx]
    )
    
    # Create temporary file
    temp_file = f"temp_{vowel}_{variation}.wav"
    await communicate.save(temp_file)
    return temp_file

def change_pitch(audio_file, semitones):
    """
    Change pitch of an audio file by a given number of semitones.
    :param audio_file: Path to input audio file
    :param semitones: Number of semitones to shift (+/-)
    :return: AudioSegment with adjusted pitch
    """
    # Read audio data using soundfile first
    data, samplerate = sf.read(audio_file)
    
    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    
    # Save as temporary WAV file
    temp_wav = f"temp_pitch_{os.path.basename(audio_file)}"
    sf.write(temp_wav, data, samplerate, 'PCM_16')
    
    # Process with pydub
    sound = AudioSegment.from_wav(temp_wav)
    new_sample_rate = int(sound.frame_rate * (2.0 ** (semitones / 12.0)))
    pitched = sound._spawn(sound.raw_data, overrides={
        'frame_rate': new_sample_rate
    }).set_frame_rate(44100)
    
    # Clean up temporary file
    os.remove(temp_wav)
    
    return pitched

async def main():
    # Generate and export vowel variations
    num_variations = 200
    vowel_folders = ['A', 'E', 'I', 'O', 'U']
    pitch_variations = [-2, -1, 0, 1, 2]  # Semitones to shift

    for vowel in vowel_folders:
        os.makedirs(vowel, exist_ok=True)
        for variation in range(num_variations):
            try:
                # Generate base audio
                temp_file = await generate_vowel_edge(vowel, variation)
                base_filename = f"{vowel}/{vowel}_var{variation}.wav"
                
                # Read and write using soundfile to ensure correct WAV format
                data, samplerate = sf.read(temp_file)
                sf.write(base_filename, data, samplerate, 'PCM_16')
                
                # Apply pitch variations
                for semitones in pitch_variations:
                    if semitones != 0:
                        pitched_audio = change_pitch(base_filename, semitones)
                        pitch_filename = f"{vowel}/{vowel}_var{variation}_pitch{semitones}.wav"
                        pitched_audio.export(pitch_filename, format="wav")
                
                # Clean up temporary file
                os.remove(temp_file)
                print(f"Generated {base_filename} with variations")
                
            except Exception as e:
                print(f"Error processing {vowel} variation {variation}: {str(e)}")
                continue

if __name__ == "__main__":
    # Install required packages if not already installed
    try:
        import edge_tts
    except ImportError:
        import pip
        pip.main(['install', 'edge-tts'])
        import edge_tts

    # Run the async main function
    asyncio.run(main())
