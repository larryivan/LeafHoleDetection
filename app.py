from flask import Flask, request, render_template, jsonify, send_file, session
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import base64
import io
from leaf_hole_detection import LeafHoleDetector
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = 'your-secret-key-here'  # Change this in production

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image):
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.png', image)
    img_str = base64.b64encode(buffer).decode()
    return f"data:image/png;base64,{img_str}"

# Store processed images temporarily (in production, use Redis or database)
processed_images = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/manual_edit', methods=['POST'])
def manual_edit():
    try:
        data = request.json
        session_id = session.get('session_id')
        
        if not session_id or session_id not in processed_images:
            return jsonify({'error': 'No image data found. Please upload an image first.'}), 400
        
        image_data = processed_images[session_id]
        enhanced_image = image_data['enhanced']
        original_shape = enhanced_image.shape[:2]
        
        # Process manual selections
        leaf_mask = create_mask_from_selections(
            data['leaf'], 
            data['canvasWidth'], 
            data['canvasHeight'], 
            original_shape
        )
        
        holes_mask = create_mask_from_selections(
            data['holes'], 
            data['canvasWidth'], 
            data['canvasHeight'], 
            original_shape
        )
        
        # Calculate new statistics
        if leaf_mask is not None and np.any(leaf_mask):
            leaf_area = cv2.countNonZero(leaf_mask)
            # Ensure holes are within leaf area
            if holes_mask is not None:
                holes_mask = cv2.bitwise_and(holes_mask, leaf_mask)
            hole_area = cv2.countNonZero(holes_mask) if holes_mask is not None else 0
        else:
            return jsonify({'error': 'Please select a valid leaf area'}), 400
        
        # Count hole regions
        if holes_mask is not None and np.any(holes_mask):
            contours, _ = cv2.findContours(holes_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hole_count = len(contours)
        else:
            hole_count = 0
        
        hole_ratio = hole_area / leaf_area if leaf_area > 0 else 0
        
        # Update stored data
        image_data['manual_leaf_mask'] = leaf_mask
        image_data['manual_holes_mask'] = holes_mask
        
        statistics = {
            'hole_ratio': f"{hole_ratio:.2%}",
            'leaf_area': leaf_area,
            'hole_area': hole_area,
            'hole_count': hole_count
        }
        
        return jsonify({
            'success': True,
            'statistics': statistics
        })
        
    except Exception as e:
        return jsonify({'error': f'Manual edit failed: {str(e)}'}), 500

def create_mask_from_selections(selections, canvas_width, canvas_height, target_shape):
    """Create binary mask from manual selections"""
    if not selections:
        return None
    
    mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    
    for path in selections:
        if len(path) > 2:
            # Convert path to numpy array
            points = np.array([[int(p['x']), int(p['y'])] for p in path], dtype=np.int32)
            # Fill the polygon
            cv2.fillPoly(mask, [points], 255)
    
    # Resize mask to match target image dimensions
    if mask.shape != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]))
    
    return mask

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Generate session ID for this processing session
            session_id = str(uuid.uuid4())
            session['session_id'] = session_id
            
            # Get language preference from request headers
            lang = request.headers.get('Accept-Language', 'en')
            if 'zh' in lang:
                session['language'] = 'zh'
            else:
                session['language'] = 'en'
            
            # Process the image
            detector = LeafHoleDetector()
            detector.debug = False  # Disable matplotlib display
            
            # Read and process image
            image = cv2.imread(filepath)
            if image is None:
                return jsonify({'error': 'Invalid image file'}), 400
            
            # Process image
            enhanced = detector.enhance_image_scan_effect(image)
            leaf_mask, leaf_contour = detector.segment_leaf(enhanced)
            
            if leaf_mask is None:
                return jsonify({'error': 'Could not detect leaf in image'}), 400
            
            hole_contours, holes_binary = detector.detect_holes(enhanced, leaf_mask)
            hole_ratio = detector.calculate_hole_ratio(leaf_mask, holes_binary)
            
            # Store processed images for manual editing
            processed_images[session_id] = {
                'original': image,
                'enhanced': enhanced,
                'leaf_mask': leaf_mask,
                'holes_binary': holes_binary,
                'hole_contours': hole_contours
            }
            
            # Create result visualization
            result_image = enhanced.copy()
            cv2.drawContours(result_image, hole_contours, -1, (0, 0, 255), 2)
            
            # Convert images to base64 for web display
            original_b64 = image_to_base64(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            enhanced_b64 = image_to_base64(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
            leaf_mask_b64 = image_to_base64(leaf_mask)
            holes_b64 = image_to_base64(holes_binary)
            result_b64 = image_to_base64(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            
            # Calculate statistics
            leaf_area = cv2.countNonZero(leaf_mask)
            hole_area = cv2.countNonZero(holes_binary)
            hole_count = len(hole_contours)
            
            results = {
                'success': True,
                'hole_ratio': f"{hole_ratio:.2%}",
                'statistics': {
                    'leaf_area': leaf_area,
                    'hole_area': hole_area,
                    'hole_count': hole_count,
                    'ratio_decimal': hole_ratio
                },
                'images': {
                    'original': original_b64,
                    'enhanced': enhanced_b64,
                    'leaf_mask': leaf_mask_b64,
                    'holes': holes_b64,
                    'result': result_b64
                }
            }
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify(results)
            
        except Exception as e:
            # Clean up uploaded file on error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5088)
