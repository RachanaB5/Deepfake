# backend/flask_api.py
from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
from PIL import Image
import io
import tempfile
import os
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/detect', methods=['POST'])
def detect_deepfake():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
            
        file = request.files['file']
        
        # Validate file type
        if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            return jsonify({"error": "Only JPG/PNG images supported"}), 400

        # Read and verify image
        try:
            img = Image.open(file.stream)
            img.verify()
            img = Image.open(file.stream)  # Reopen after verify
            img_array = np.array(img)
        except Exception as e:
            return jsonify({"error": f"Invalid image: {str(e)}"}), 400

        # Generate heatmap
        try:
            heatmap = generate_heatmap(img_array)
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                Image.fromarray(heatmap).save(tmp.name)
                return jsonify({
                    "success": True,
                    "heatmap_path": tmp.name,
                    "message": "Analysis complete"
                })
        except ValueError as e:
            return jsonify({"error": str(e)}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500
        
@app.route('/heatmap/<filename>', methods=['GET'])
def generate_heatmap(img_array):
    """Robust heatmap generation with proper error handling"""
    try:
        # 1. Ensure proper image format
        if len(img_array.shape) == 2:  # Grayscale image
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:  # RGBA image
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        
        # 2. Convert to grayscale and detect edges
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # 3. Create heatmap and ensure matching dimensions
        heatmap = cv2.applyColorMap(edges, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
        
        # 4. Blend images with type conversion
        blended = cv2.addWeighted(
            img_array.astype('float32'),
            0.7,
            heatmap.astype('float32'),
            0.3,
            0
        )
        
        return blended.astype('uint8')
        
    except Exception as e:
        logger.error(f"Heatmap generation failed: {str(e)}")
        raise ValueError(f"Heatmap generation error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3200, debug=True)