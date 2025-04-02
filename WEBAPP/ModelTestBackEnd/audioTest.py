import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import io
from PIL import Image
import time

app = Flask(__name__)
# CORS(app, resources={r"/api/*": {"origins": "*"}})
CORS(app)

def create_data_generator():
    test_datagen = ImageDataGenerator(
        rescale=1./255
    )
    
    test_generator = test_datagen.flow_from_directory(
        'test',
        target_size=(55, 240),
        batch_size=8,
        class_mode='categorical',
        shuffle=False,
        color_mode='grayscale'
    )
    
    return test_generator

def validate_model_thoroughly(model, test_generator, class_names):

    results_by_class = {cls: {'correct': 0, 'total': 0, 'confidences': []} 
                       for cls in class_names}
    
    all_predictions = []
    all_true_labels = []

    for i in range(len(test_generator)):
        images, labels = test_generator[i]
        predictions = model.predict(images, verbose=0)
        
        for img_idx in range(len(images)):
            true_class = np.argmax(labels[img_idx])
            pred_class = np.argmax(predictions[img_idx])
            confidence = predictions[img_idx][pred_class]
            
            cls_name = class_names[true_class]
            results_by_class[cls_name]['total'] += 1
            if true_class == pred_class:
                results_by_class[cls_name]['correct'] += 1
            results_by_class[cls_name]['confidences'].append(confidence)
            
            all_predictions.append(pred_class)
            all_true_labels.append(true_class)
    

    for cls in class_names:
        results = results_by_class[cls]
        accuracy = results['correct'] / results['total'] * 100
        avg_confidence = np.mean(results['confidences']) * 100
        min_confidence = np.min(results['confidences']) * 100
        max_confidence = np.max(results['confidences']) * 100
        
        print(f"\n类别 {cls}:")
        print(f"准确率: {accuracy:.2f}% ({results['correct']}/{results['total']})")
        print(f"平均置信度: {avg_confidence:.2f}%")
        print(f"最低置信度: {min_confidence:.2f}%")
        print(f"最高置信度: {max_confidence:.2f}%")
    
    # 2. 绘制混淆矩阵
    cm = tf.math.confusion_matrix(all_true_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # 在每个单元格中添加数值
    thresh = cm.numpy().max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j].numpy(), 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j].numpy() > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return results_by_class

def create_spectrogram_from_audio_bytes(audio_bytes, save_path='test_spectrograms'):
    """Convert audio bytes to spectrogram image using the same parameters as imageGenerator.py"""
    print("\n=== Starting Spectrogram Generation Process ===")
    
    # Create directory if it doesn't exist
    print(f"Creating directory: {save_path}")
    os.makedirs(save_path, exist_ok=True)
    
    # Save audio bytes to temporary file
    temp_audio_path = os.path.join(save_path, 'temp_audio.wav')
    print(f"Saving temporary audio file to: {temp_audio_path}")
    with open(temp_audio_path, 'wb') as f:
        f.write(audio_bytes)
    
    # Load audio using librosa
    print("Loading audio with librosa...")
    y, sr = librosa.load(temp_audio_path, sr=None)
    print(f"Audio loaded - Sample rate: {sr}Hz, Duration: {len(y)/sr:.2f}s")
    
    # Calculate mel spectrogram
    print("\nCalculating mel spectrogram...")
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
    print(f"Mel spectrogram shape: {mel_spect.shape}")
    
    # Convert to dB scale
    print("Converting to dB scale...")
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
    
    # Create figure
    print("\nCreating spectrogram plot...")
    plt.figure(figsize=(0.55, 2.4))
    
    # Plot spectrogram
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
    
    # Save spectrogram image
    timestamp = int(time.time())
    spec_filename = f'spectrogram_{timestamp}.png'
    spec_path = os.path.join(save_path, spec_filename)
    print(f"\nSaving original spectrogram to: {spec_path}")
    
    plt.savefig(spec_path,
                dpi=100,
                bbox_inches='tight',
                pad_inches=0,
                format='png')
    plt.close()
    
    # Load and preprocess the saved spectrogram
    print("\nPreprocessing spectrogram...")
    img = Image.open(spec_path)
    print(f"Original image size: {img.size}, mode: {img.mode}")
    
    img = img.convert('L')
    print("Converted to grayscale")
    
    # Convert to numpy array and preprocess
    img_array = np.array(img)
    print(f"Initial array shape: {img_array.shape}")
    
    # Rotate if needed
    if img_array.shape[0] != 240 or img_array.shape[1] != 55:
        print(f"Rotating image to correct orientation...")
        img_array = np.rot90(img_array, k=-1)
        print(f"After rotation shape: {img_array.shape}")
    
    # Apply contrast enhancement
    print("\nApplying contrast enhancement...")
    img_array = img_array.astype('float32') / 255.0
    mean = np.mean(img_array)
    adjusted = (img_array - mean) * 1.5 + mean
    img_array = np.clip(adjusted, 0, 1)
    print(f"Array value range: [{np.min(img_array):.3f}, {np.max(img_array):.3f}]")
    
    # Save the processed image
    processed_img = Image.fromarray((img_array * 255).astype(np.uint8))
    processed_path = os.path.join(save_path, f'processed_{spec_filename}')
    processed_img.save(processed_path)
    print(f"Saved processed image to: {processed_path}")
    
    # Add dimensions
    print("\nAdding channel and batch dimensions...")
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    print(f"Final array shape: {img_array.shape}")
    
    # Clean up
    try:
        print("\nCleaning up temporary audio file...")
        os.remove(temp_audio_path)
        print("Cleanup completed")
    except Exception as e:
        print(f"Warning: Could not remove temporary audio file: {str(e)}")
    
    print("\n=== Spectrogram Generation Complete ===")
    return img_array

@app.route('/api/test-model', methods=['POST'])
def test_model():
    print("\n=== Starting Model Test Process ===")
    try:
        # Get the base64 audio data from request
        print("Receiving request data...")
        data = request.json
        base64_audio = data['input']
        file_name = data.get('fileName', '')
        actual_vowel = file_name[0].lower() if file_name else 'unknown'
        print(f"Processing file: {file_name}, Expected vowel: {actual_vowel}")
        
        # Remove the data URL prefix if present
        if ',' in base64_audio:
            base64_audio = base64_audio.split(',')[1]
        
        # Decode base64 to bytes
        print("Decoding base64 audio data...")
        audio_bytes = base64.b64decode(base64_audio)
        print(f"Decoded audio size: {len(audio_bytes)} bytes")
        
        # Create test directory
        timestamp = int(time.time())
        test_dir = os.path.join('test_spectrograms', f'test_{timestamp}')
        print(f"\nCreating test directory: {test_dir}")
        
        # Generate spectrogram
        print("\nGenerating spectrogram...")
        img_array = create_spectrogram_from_audio_bytes(audio_bytes, save_path=test_dir)
        
        # Load model
        print("\nLoading model...")
        model_path = 'analysis/best_model.keras'
        if not os.path.exists(model_path):
            model_path = 'analysis/best_model.h5'
        print(f"Using model from: {model_path}")
        
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        print("Model loaded and compiled successfully")
        
        # Make prediction
        print("\nMaking prediction...")
        predictions = model.predict(img_array, verbose=0)
        pred_class = np.argmax(predictions[0])
        confidence = float(predictions[0][pred_class])
        
        # Process results
        vowel_mapping = {0: 'a', 1: 'e', 2: 'i', 3: 'o', 4: 'u'}
        predicted_vowel = vowel_mapping.get(pred_class, 'unknown')
        print(f"\nPrediction Results:")
        print(f"Predicted vowel: {predicted_vowel}")
        print(f"Confidence: {confidence*100:.2f}%")
        print(f"Actual vowel: {actual_vowel}")
        
        # Create response
        all_probabilities = {
            vowel_mapping[i]: float(prob) 
            for i, prob in enumerate(predictions[0])
        }
        
        print("\nProbabilities for each vowel:")
        for vowel, prob in all_probabilities.items():
            print(f"{vowel}: {prob*100:.2f}%")
        
        response = {
            'success': True,
            'prediction': int(pred_class),
            'predictedVowel': predicted_vowel,
            'actualVowel': actual_vowel,
            'isCorrect': predicted_vowel == actual_vowel,
            'confidence': confidence,
            'allProbabilities': all_probabilities
        }
        
        print("\n=== Model Test Complete ===")
        return jsonify(response)
        
    except Exception as e:
        print("\n=== Error in Model Test ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

def main():
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == "__main__":
    main() 