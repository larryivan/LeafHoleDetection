from flask import Flask, request, render_template, jsonify, send_file, session, make_response
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
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
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
            leaf_area_pixels = cv2.countNonZero(leaf_mask)
            # Ensure holes are within leaf area
            if holes_mask is not None:
                holes_mask = cv2.bitwise_and(holes_mask, leaf_mask)
            hole_area_pixels = cv2.countNonZero(holes_mask) if holes_mask is not None else 0
        else:
            return jsonify({'error': 'Please select a valid leaf area'}), 400
        
        # Count hole regions
        if holes_mask is not None and np.any(holes_mask):
            contours, _ = cv2.findContours(holes_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hole_count = len(contours)
        else:
            hole_count = 0
        
        hole_ratio = hole_area_pixels / leaf_area_pixels if leaf_area_pixels > 0 else 0
        
        # Calculate leaf dimensions from manual leaf mask
        leaf_length_pixels = 0
        leaf_width_pixels = 0
        if leaf_mask is not None and np.any(leaf_mask):
            # Find leaf contour from mask
            leaf_contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if leaf_contours:
                # Use the largest contour as the leaf
                leaf_contour = max(leaf_contours, key=cv2.contourArea)
                
                # Calculate leaf dimensions using the detector
                from leaf_hole_detection import LeafHoleDetector
                detector = LeafHoleDetector()
                leaf_length_info = detector.calculate_leaf_length(leaf_contour)
                leaf_width_info = detector.calculate_leaf_width(leaf_contour)
                
                leaf_length_pixels = leaf_length_info[0]
                leaf_width_pixels = leaf_width_info[0]
        
        # Update stored data
        image_data['manual_leaf_mask'] = leaf_mask
        image_data['manual_holes_mask'] = holes_mask
        
        # Get pixels per cm ratio from stored data
        pixels_per_cm = image_data.get('pixels_per_cm')
        
        statistics = {
            'hole_ratio': f"{hole_ratio:.2%}",
            'leaf_area_pixels': leaf_area_pixels,
            'hole_area_pixels': hole_area_pixels,
            'hole_count': hole_count,
            'has_reference': pixels_per_cm is not None,
            'pixels_per_cm': pixels_per_cm,
            'leaf_length_pixels': leaf_length_pixels,
            'leaf_width_pixels': leaf_width_pixels
        }
        
        # Add absolute areas if reference is available
        if pixels_per_cm is not None:
            from leaf_hole_detection import LeafHoleDetector
            detector = LeafHoleDetector()
            leaf_area_cm2 = detector.convert_to_absolute_area(leaf_area_pixels, pixels_per_cm)
            hole_area_cm2 = detector.convert_to_absolute_area(hole_area_pixels, pixels_per_cm)
            leaf_length_cm = leaf_length_pixels / pixels_per_cm if leaf_length_pixels > 0 else 0
            leaf_width_cm = leaf_width_pixels / pixels_per_cm if leaf_width_pixels > 0 else 0
            
            statistics.update({
                'leaf_area_cm2': leaf_area_cm2,
                'hole_area_cm2': hole_area_cm2,
                'leaf_length_cm': leaf_length_cm,
                'leaf_width_cm': leaf_width_cm,
                'reference_detected': True
            })
        else:
            statistics.update({
                'leaf_area_cm2': None,
                'hole_area_cm2': None,
                'leaf_length_cm': None,
                'leaf_width_cm': None,
                'reference_detected': False
            })
        
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
            
            # Process image with new comprehensive method
            processing_results = detector.process_image(filepath)
            
            if 'error' in processing_results:
                return jsonify({'error': processing_results['error']}), 400
            
            # Validate that we have the required leaf dimension data
            if 'leaf_length_pixels' not in processing_results:
                processing_results['leaf_length_pixels'] = 0
            if 'leaf_width_pixels' not in processing_results:
                processing_results['leaf_width_pixels'] = 0
            
            # Extract results
            hole_ratio = processing_results['hole_ratio']
            leaf_area_pixels = processing_results['leaf_area_pixels']
            hole_area_pixels = processing_results['hole_area_pixels']
            hole_count = processing_results['hole_count']
            pixels_per_cm = processing_results['pixels_per_cm']
            has_reference = processing_results['has_reference']
            
            # Get image data
            original_image = processing_results['original_image']
            enhanced = processing_results['enhanced_image']
            leaf_mask = processing_results['leaf_mask']
            holes_binary = processing_results['holes_binary']
            hole_contours = processing_results['hole_contours']
            
            # Store processed images for manual editing
            processed_images[session_id] = {
                'original': original_image,
                'enhanced': enhanced,
                'leaf_mask': leaf_mask,
                'holes_binary': holes_binary,
                'hole_contours': hole_contours,
                'pixels_per_cm': pixels_per_cm  # Store for manual editing
            }
            
            # Create result visualization
            result_image = enhanced.copy()
            cv2.drawContours(result_image, hole_contours, -1, (0, 0, 255), 2)
            
            # Draw reference square if detected
            if has_reference:
                ref_result = detector.detect_reference_square(original_image)
                if ref_result is not None:
                    ref_contour, _ = ref_result
                    # Draw reference square with thick blue outline
                    cv2.drawContours(result_image, [ref_contour], -1, (255, 0, 0), 3)
                    # Add text label
                    rect = cv2.boundingRect(ref_contour)
                    cv2.putText(result_image, '1cm', (rect[0], rect[1]-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # Convert images to base64 for web display
            original_b64 = image_to_base64(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            enhanced_b64 = image_to_base64(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
            leaf_mask_b64 = image_to_base64(leaf_mask)
            holes_b64 = image_to_base64(holes_binary)
            result_b64 = image_to_base64(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            
            # Get leaf dimensions
            leaf_length_pixels = processing_results.get('leaf_length_pixels', 0)
            leaf_width_pixels = processing_results.get('leaf_width_pixels', 0)
            
            # Prepare statistics
            statistics = {
                'leaf_area_pixels': leaf_area_pixels,
                'hole_area_pixels': hole_area_pixels,
                'hole_count': hole_count,
                'ratio_decimal': hole_ratio,
                'has_reference': has_reference,
                'pixels_per_cm': pixels_per_cm,
                'leaf_length_pixels': leaf_length_pixels,
                'leaf_width_pixels': leaf_width_pixels
            }
            
            # Add absolute areas if reference is detected
            if has_reference:
                statistics.update({
                    'leaf_area_cm2': processing_results['leaf_area_cm2'],
                    'hole_area_cm2': processing_results['hole_area_cm2'],
                    'leaf_length_cm': processing_results.get('leaf_length_cm'),
                    'leaf_width_cm': processing_results.get('leaf_width_cm'),
                    'reference_detected': True
                })
            else:
                statistics.update({
                    'leaf_area_cm2': None,
                    'hole_area_cm2': None,
                    'leaf_length_cm': None,
                    'leaf_width_cm': None,
                    'reference_detected': False
                })
            
            results = {
                'success': True,
                'hole_ratio': f"{hole_ratio:.2%}",
                'statistics': statistics,
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

@app.route('/batch_upload', methods=['POST'])
def batch_upload():
    if 'files' not in request.files:
        return jsonify({'error': 'No files selected'}), 400
    
    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        return jsonify({'error': 'No files selected'}), 400
    
    # Validate file types
    valid_files = []
    for file in files:
        if file and allowed_file(file.filename):
            valid_files.append(file)
    
    if not valid_files:
        return jsonify({'error': 'No valid image files found'}), 400
    
    # Process each file
    results = []
    detector = LeafHoleDetector()
    detector.debug = False  # Disable matplotlib display for batch processing
    
    for i, file in enumerate(valid_files):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"batch_{i}_{filename}")
            file.save(filepath)
            
            # Process the image
            processing_results = detector.process_image(filepath)
            
            if 'error' in processing_results:
                results.append({
                    'filename': filename,
                    'success': False,
                    'error': processing_results['error']
                })
            else:
                # Validate that we have the required leaf dimension data
                if 'leaf_length_pixels' not in processing_results:
                    processing_results['leaf_length_pixels'] = 0
                if 'leaf_width_pixels' not in processing_results:
                    processing_results['leaf_width_pixels'] = 0
                # Extract key results for batch processing
                hole_ratio = processing_results['hole_ratio']
                leaf_area_pixels = processing_results['leaf_area_pixels']
                hole_area_pixels = processing_results['hole_area_pixels']
                hole_count = processing_results['hole_count']
                pixels_per_cm = processing_results['pixels_per_cm']
                has_reference = processing_results['has_reference']
                leaf_length_pixels = processing_results.get('leaf_length_pixels', 0)
                leaf_width_pixels = processing_results.get('leaf_width_pixels', 0)
                
                # Get image data
                original_image = processing_results['original_image']
                enhanced = processing_results['enhanced_image']
                leaf_mask = processing_results['leaf_mask']
                holes_binary = processing_results['holes_binary']
                hole_contours = processing_results['hole_contours']
                
                # Create result visualization
                result_image = enhanced.copy()
                cv2.drawContours(result_image, hole_contours, -1, (0, 0, 255), 2)
                
                # Draw reference square if detected
                if has_reference:
                    ref_result = detector.detect_reference_square(original_image)
                    if ref_result is not None:
                        ref_contour, _ = ref_result
                        cv2.drawContours(result_image, [ref_contour], -1, (255, 0, 0), 3)
                        rect = cv2.boundingRect(ref_contour)
                        cv2.putText(result_image, '1cm', (rect[0], rect[1]-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
                # Convert images to base64 for web display
                images = {
                    'original': image_to_base64(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)),
                    'enhanced': image_to_base64(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)),
                    'leaf_mask': image_to_base64(leaf_mask),
                    'holes': image_to_base64(holes_binary),
                    'result': image_to_base64(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
                }
                
                result_item = {
                    'filename': filename,
                    'success': True,
                    'hole_ratio': f"{hole_ratio:.2%}",
                    'statistics': {
                        'leaf_area_pixels': leaf_area_pixels,
                        'hole_area_pixels': hole_area_pixels,
                        'hole_count': hole_count,
                        'ratio_decimal': hole_ratio,
                        'has_reference': has_reference,
                        'pixels_per_cm': pixels_per_cm,
                        'leaf_length_pixels': leaf_length_pixels,
                        'leaf_width_pixels': leaf_width_pixels
                    },
                    'images': images  # Include images for detailed display
                }
                
                # Add absolute areas if reference is detected
                if has_reference:
                    result_item['statistics'].update({
                        'leaf_area_cm2': processing_results['leaf_area_cm2'],
                        'hole_area_cm2': processing_results['hole_area_cm2'],
                        'leaf_length_cm': processing_results.get('leaf_length_cm'),
                        'leaf_width_cm': processing_results.get('leaf_width_cm'),
                        'reference_detected': True
                    })
                else:
                    result_item['statistics'].update({
                        'leaf_area_cm2': None,
                        'hole_area_cm2': None,
                        'leaf_length_cm': None,
                        'leaf_width_cm': None,
                        'reference_detected': False
                    })
                
                results.append(result_item)
            
            # Clean up uploaded file
            os.remove(filepath)
            
        except Exception as e:
            results.append({
                'filename': file.filename,
                'success': False,
                'error': f'Processing failed: {str(e)}'
            })
            # Clean up uploaded file on error
            if os.path.exists(filepath):
                os.remove(filepath)
    
    # Calculate summary statistics
    successful_results = [r for r in results if r['success']]
    total_files = len(results)
    successful_files = len(successful_results)
    
    summary = {
        'total_files': total_files,
        'successful_files': successful_files,
        'failed_files': total_files - successful_files
    }
    
    if successful_results:
        # Calculate average hole ratio
        hole_ratios = [r['statistics']['ratio_decimal'] for r in successful_results]
        avg_hole_ratio = sum(hole_ratios) / len(hole_ratios)
        
        # Calculate totals if any files have reference squares
        files_with_reference = [r for r in successful_results if r['statistics']['has_reference']]
        if files_with_reference:
            total_leaf_area_cm2 = sum(r['statistics']['leaf_area_cm2'] for r in files_with_reference)
            total_hole_area_cm2 = sum(r['statistics']['hole_area_cm2'] for r in files_with_reference)
            
            summary.update({
                'avg_hole_ratio': f"{avg_hole_ratio:.2%}",
                'files_with_reference': len(files_with_reference),
                'total_leaf_area_cm2': total_leaf_area_cm2,
                'total_hole_area_cm2': total_hole_area_cm2,
                'overall_hole_ratio_cm2': f"{(total_hole_area_cm2 / total_leaf_area_cm2 * 100):.2f}%" if total_leaf_area_cm2 > 0 else "0%"
            })
        else:
            summary.update({
                'avg_hole_ratio': f"{avg_hole_ratio:.2%}",
                'files_with_reference': 0,
                'total_leaf_area_cm2': None,
                'total_hole_area_cm2': None,
                'overall_hole_ratio_cm2': None
            })
    
    return jsonify({
        'success': True,
        'summary': summary,
        'results': results
    })

@app.route('/export_batch_results', methods=['POST'])
def export_batch_results():
    """Export batch processing results to CSV"""
    try:
        data = request.json
        if not data or 'results' not in data:
            return jsonify({'error': 'No results data provided'}), 400
        
        results = data['results']
        
        # Create CSV content
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        header = ['Filename', 'Success', 'Hole Ratio', 'Leaf Area (pixels)', 'Hole Area (pixels)', 'Hole Count', 'Leaf Length (pixels)', 'Leaf Width (pixels)']
        if any(r.get('statistics', {}).get('has_reference', False) for r in results if r.get('success')):
            header.extend(['Leaf Area (cm²)', 'Hole Area (cm²)', 'Leaf Length (cm)', 'Leaf Width (cm)', 'Reference Detected', 'Pixels per cm'])
        writer.writerow(header)
        
        # Write data rows
        for result in results:
            if result.get('success'):
                stats = result.get('statistics', {})
                row = [
                    result.get('filename', ''),
                    'Yes',
                    result.get('hole_ratio', ''),
                    stats.get('leaf_area_pixels', ''),
                    stats.get('hole_area_pixels', ''),
                    stats.get('hole_count', ''),
                    f"{stats.get('leaf_length_pixels', 0):.1f}",
                    f"{stats.get('leaf_width_pixels', 0):.1f}"
                ]
                
                if stats.get('has_reference'):
                    row.extend([
                        f"{stats.get('leaf_area_cm2', 0):.2f}",
                        f"{stats.get('hole_area_cm2', 0):.2f}",
                        f"{stats.get('leaf_length_cm', 0):.2f}",
                        f"{stats.get('leaf_width_cm', 0):.2f}",
                        'Yes',
                        f"{stats.get('pixels_per_cm', 0):.2f}"
                    ])
                else:
                    row.extend(['', '', '', '', 'No', ''])
                    
                writer.writerow(row)
            else:
                row = [result.get('filename', ''), 'No', '', '', '', '', '', '']
                if any(r.get('statistics', {}).get('has_reference', False) for r in results if r.get('success')):
                    row.extend(['', '', '', '', '', ''])
                writer.writerow(row)
        
        # Get CSV content
        csv_content = output.getvalue()
        output.close()
        
        # Create response
        response = make_response(csv_content)
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = 'attachment; filename=leaf_hole_detection_results.csv'
        
        return response
        
    except Exception as e:
        return jsonify({'error': f'Export failed: {str(e)}'}), 500

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=1102)
