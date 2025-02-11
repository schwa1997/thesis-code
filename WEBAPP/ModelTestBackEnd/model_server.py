from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import io
import base64
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)
# 修改 CORS 设置，明确允许所有源
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Mapping from class index to vowel
VOWEL_MAPPING = {
    0: 'a',
    1: 'e',
    2: 'i',
    3: 'o',
    4: 'u'
}

def preprocess_image(image):
    """
    确保预处理步骤与训练时完全一致
    """
    print(f"Preprocessing image...")
    print(f"Initial image mode: {image.mode}")
    print(f"Initial image size: {image.size}")
    
    # 1. 转换为灰度图
    if image.mode != 'L':
        image = image.convert('L')
        print(f"After grayscale conversion: {image.mode}")
    
    # 2. 转换为numpy数组并调整方向
    image_array = np.array(image)
    # 如果图像方向不正确，进行旋转
    if image_array.shape[0] != 240 or image_array.shape[1] != 55:
        image_array = np.rot90(image_array, k=-1)  # 顺时针旋转90度
    print(f"After rotation shape: {image_array.shape}")
    
    # 3. 确保数据类型和归一化
    image_array = image_array.astype('float32') / 255.0
    
    # 4. 应用对比度增强（与CNN.py中的预处理一致）
    mean = np.mean(image_array)
    adjusted = (image_array - mean) * 1.5 + mean
    image_array = np.clip(adjusted, 0, 1)
    print(f"Applied contrast enhancement")
    
    # 5. 添加通道维度和批次维度
    image_array = np.expand_dims(image_array, axis=-1)  # 添加通道维度
    image_array = np.expand_dims(image_array, axis=0)   # 添加批次维度
    print(f"Final array shape: {image_array.shape}")
    
    # 保存处理后的图像用于调试
    debug_image = Image.fromarray((image_array[0, :, :, 0] * 255).astype(np.uint8))
    debug_image.save('debug_processed_image.png')
    print(f"Saved debug image to debug_processed_image.png")
    
    # 验证最终形状
    if image_array.shape != (1, 240, 55, 1):
        raise ValueError(f"Incorrect final shape: {image_array.shape}, expected (1, 240, 55, 1)")
    
    return image_array

# Load model
try:
    model = tf.keras.models.load_model('analysis/best_model.keras')
    print("Model loaded successfully!")
    
    # 打印模型结构以验证
    print("\nModel Summary:")
    model.summary()
    
    # 打印输入输出形状
    print("\nModel input shape:", model.input_shape)
    print("Model output shape:", model.output_shape)
    
except Exception as e:
    print(f"Failed to load model: {str(e)}")
    model = None

@app.route('/api/test-model', methods=['POST'])
def test_model():
    print("Received request")  # 调试日志
    # 添加预请求处理
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded correctly'
        })

    try:
        # Get image data from frontend
        data = request.json
        if not data or 'input' not in data:
            return jsonify({
                'success': False,
                'error': 'No image data received'
            })

        # Get actual vowel from filename
        file_name = data.get('fileName', '')
        actual_vowel = file_name[0].lower() if file_name else 'unknown'

        # Decode and load image
        image_data = base64.b64decode(data['input'].split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        
        # 打印原始图像信息
        print(f"\nOriginal image mode: {image.mode}")
        print(f"Original image size: {image.size}")
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        predicted_vowel = VOWEL_MAPPING.get(predicted_class, 'unknown')
        
        # Print detailed debug information
        print("\nPrediction details:")
        print(f"Predicted class: {predicted_class} ({predicted_vowel})")
        print(f"Actual vowel: {actual_vowel}")
        print("\nProbabilities for each class:")
        for i, prob in enumerate(predictions[0]):
            print(f"{VOWEL_MAPPING[i]}: {prob:.6f}")
        
        return jsonify({
            'success': True,
            'prediction': int(predicted_class),
            'predictedVowel': predicted_vowel,
            'actualVowel': actual_vowel,
            'isCorrect': predicted_vowel == actual_vowel,
            'confidence': confidence,
            'allProbabilities': {VOWEL_MAPPING[i]: float(prob) for i, prob in enumerate(predictions[0])}
        })
    
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(port=5000, debug=True, host='0.0.0.0') 