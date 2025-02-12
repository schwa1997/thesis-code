from gtts import gTTS
import os
import random
import numpy as np

# Output folder
output_dir = "audio"
os.makedirs(output_dir, exist_ok=True)

# Define Italian vowels with more variations
vowel_variations = {
    "a": [
        # 标准发音
        "Ah", "Aah", "Ahh", "Ahhh", "Aaah",
        # 长音
        "Aaaah", "Aaaaah", "Aaaaaah", "Aaaaaaah",
        # 重音
        "AH", "AAH", "AAHH", "AAHHH",
        # 组合音
        "Aha", "Ahaa", "Aaha", "Aahaa",
        # 轻音
        "ah", "aah", "ahh", "ahhh",
        # 变调
        "Aah?", "Ahh!", "Aaah~", "Ahhh~"
    ],
    "e": [
        # 标准发音
        "Eh", "Eeh", "Ehh", "Ehhh", "Eeeh",
        # 长音
        "Eeeeh", "Eeeeeh", "Eeeeeeh", "Eeeeeeeh",
        # 重音
        "EH", "EEH", "EEHH", "EEHHH",
        # 组合音
        "Ehe", "Ehee", "Eehe", "Eehee",
        # 轻音
        "eh", "eeh", "ehh", "ehhh",
        # 变调
        "Eeh?", "Ehh!", "Eeeh~", "Ehhh~"
    ],
    "i": [
        # 标准发音
        "Ee", "Eee", "Ii", "Iii", "Iiii",
        # 长音
        "Eeee", "Eeeee", "Iiii", "Iiiii",
        # 重音
        "EE", "EEE", "II", "III",
        # 组合音
        "Eei", "Eeii", "Ieei", "Iiei",
        # 轻音
        "ee", "eee", "ii", "iii",
        # 变调
        "Ee?", "Ii!", "Eee~", "Iii~"
    ],
    "o": [
        # 标准发音
        "Oh", "Ooh", "Ohh", "Ohhh", "Oooh",
        # 长音
        "Ooooh", "Oooooh", "Ooooooh", "Oooooooh",
        # 重音
        "OH", "OOH", "OOHH", "OOHHH",
        # 组合音
        "Oho", "Ohoo", "Ooho", "Oohoo",
        # 轻音
        "oh", "ooh", "ohh", "ohhh",
        # 变调
        "Ooh?", "Ohh!", "Oooh~", "Ohhh~"
    ],
    "u": [
        # 标准发音
        "Oo", "Ooo", "Uu", "Uuu", "Uuuu",
        # 长音
        "Oooo", "Ooooo", "Uuuu", "Uuuuu",
        # 重音
        "OO", "OOO", "UU", "UUU",
        # 组合音
        "Oou", "Oouu", "Uoou", "Uuou",
        # 轻音
        "oo", "ooo", "uu", "uuu",
        # 变调
        "Oo?", "Uu!", "Ooo~", "Uuu~"
    ]
}

# TTS parameters for variation
speeds = [False, True]  # False for normal speed, True for slow
languages = ['it', 'it-IT']  # Different Italian language codes

def generate_variations(text, base_name, vowel_dir, index):
    """Generate multiple variations of the same text with different parameters"""
    variations = []
    
    # Add slight random variations to the text
    texts = [
        text,
        text.replace('h', 'hh', random.randint(0, 1)),
        text.lower(),
        text.upper(),
        text + text[0].lower()
    ]
    
    for i, variation_text in enumerate(texts):
        for speed in speeds:
            for lang in languages:
                try:
                    output_file = os.path.join(
                        vowel_dir, 
                        f"{base_name}_var_{index}_{i}_{speed}_{lang}.wav"
                    )
                    
                    # Add random pitch and rate variations
                    tts = gTTS(
                        text=variation_text,
                        lang=lang,
                        slow=speed
                    )
                    tts.save(output_file)
                    variations.append(output_file)
                    print(f"Saved: {output_file}")
                except Exception as e:
                    print(f"Error generating {output_file}: {e}")
                    continue
    
    return variations

# Generate sounds for each vowel
for vowel, variations in vowel_variations.items():
    # Create subdirectory for each vowel
    vowel_dir = os.path.join(output_dir, vowel)
    os.makedirs(vowel_dir, exist_ok=True)
    
    print(f"\nGenerating sounds for vowel '{vowel}'...")
    
    # Generate each variation
    all_variations = []
    for idx, text in enumerate(variations, 1):
        generated = generate_variations(text, vowel, vowel_dir, idx)
        all_variations.extend(generated)
    
    print(f"Generated {len(all_variations)} variations for vowel {vowel}")

print("\nAll vowel variations have been generated!")
