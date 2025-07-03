# Leaf Hole Detection System

An AI-powered web application for detecting and measuring holes in leaf images using computer vision techniques. The system provides both automatic detection and manual editing capabilities for accurate analysis.

## Features

### Core Functionality
- **Automatic Detection**: AI-powered leaf segmentation and hole detection
- **Image Enhancement**: Scanner-like effect to improve image quality
- **Manual Editing**: Interactive canvas for manual leaf area and hole selection
- **Accurate Measurements**: Calculates hole area ratio and statistics
- **Multi-language Support**: English and Chinese interface

### Technical Capabilities
- **Advanced Image Processing**: 
  - Image denoising and enhancement
  - Auto white balance and contrast optimization
  - Edge detection and morphological operations
- **Smart Hole Detection**: 
  - Detects white holes matching background color
  - Handles holes at leaf edges
  - Filters false positives using shape analysis
- **Web Interface**: 
  - Drag-and-drop file upload
  - Real-time processing visualization
  - Responsive design for all devices

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd Edge-detection
```

2. Install dependencies:
```bash
pip install -r requirements_web.txt
```

3. Create necessary directories:
```bash
mkdir uploads results static/js
```

## Usage

### Running the Application
1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

### Using the System

#### Automatic Detection
1. **Upload Image**: Drag and drop or select a leaf image
2. **Wait for Processing**: AI analyzes the image automatically
3. **View Results**: See detection statistics and visual results
4. **Download Results**: Save processed images if needed

#### Manual Editing (Optional)
1. Click **"Manual Edit"** if automatic results need adjustment
2. **Select Tools**:
   - Green tool: Draw leaf area boundaries
   - Red tool: Mark hole locations
3. **Draw Selections**: Click and drag to create selections
4. **Recalculate**: Click to update statistics based on manual input

### Supported File Formats
- PNG, JPG, JPEG, GIF, BMP
- Maximum file size: 16MB
- Recommended: High-resolution images with good lighting

## Project Structure

```
Edge-detection/
├── app.py                      # Flask web application
├── leaf_hole_detection.py      # Core detection algorithms
├── requirements_web.txt        # Python dependencies
├── templates/
│   └── index.html              # Main web interface
├── static/
│   ├── style.css              # Additional styling
│   └── js/
│       └── i18n.js            # Internationalization
├── uploads/                    # Temporary upload storage
├── results/                    # Processed results storage
└── README.md                   # This file
```


## Algorithm Overview

### Image Enhancement Pipeline
1. **Denoising**: Bilateral filtering to reduce noise
2. **Color Space Conversion**: LAB color space for better processing
3. **CLAHE**: Contrast Limited Adaptive Histogram Equalization
4. **White Balance**: Automatic color correction
5. **Sharpening**: Enhance image details

### Detection Pipeline
1. **Leaf Segmentation**: Otsu thresholding with morphological operations
2. **Hole Detection**: Multi-method approach:
   - Bright region detection for interior holes
   - Boundary topology analysis for edge holes
   - Shape and size filtering
3. **Validation**: Circularity and area constraints

### Manual Editing
- Canvas-based drawing interface
- Real-time visual feedback

## Configuration

### Application Settings
- Upload folder: `uploads/`
- Results folder: `results/`
- Max file size: 16MB
- Session timeout: 24 hours

## Deployment
```bash
python app.py
```

## Author

**Larry Ivan Han**
- Email: larryivanhan@gmail.com
