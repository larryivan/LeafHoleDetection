import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

class LeafHoleDetector:
    def __init__(self):
        self.debug = True
    
    def enhance_image_scan_effect(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image to simulate scanner effect, making white background whiter
        """
        # 1. Denoising
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # 2. Convert to LAB color space for white balance
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge back to LAB
        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # 3. Auto white balance
        enhanced = self.auto_white_balance(enhanced)
        
        # 4. Enhance contrast and brightness
        enhanced = self.enhance_contrast_brightness(enhanced)
        
        # 5. Sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    
    def auto_white_balance(self, image: np.ndarray) -> np.ndarray:
        """
        Auto white balance
        """
        result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        return result
    
    def enhance_contrast_brightness(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast and brightness to make white background whiter
        """
        # Calculate histogram
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Find brightest pixel value as white reference
        white_threshold = np.argmax(hist[200:]) + 200
        
        # Adaptive brightness and contrast adjustment
        alpha = 255.0 / white_threshold  # contrast
        beta = -alpha * 50  # brightness offset
        
        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        # Further whiten background
        enhanced = self.whiten_background(enhanced)
        
        return enhanced
    
    def whiten_background(self, image: np.ndarray) -> np.ndarray:
        """
        Further whiten background
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find background regions (brightest areas)
        _, bg_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to ensure complete background regions
        kernel = np.ones((5,5), np.uint8)
        bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Set background to pure white
        result = image.copy()
        result[bg_mask == 255] = [255, 255, 255]
        
        return result
    
    def segment_leaf(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment leaf from white background
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use Otsu threshold segmentation
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to remove noise
        kernel = np.ones((5,5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find largest contour (leaf)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
        
        # Select contour with largest area as leaf
        leaf_contour = max(contours, key=cv2.contourArea)
        
        # Create leaf mask
        leaf_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillPoly(leaf_mask, [leaf_contour], 255)
        
        return leaf_mask, leaf_contour
    
    def detect_holes(self, image: np.ndarray, leaf_mask: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Detect holes on leaf - holes are white like background, including edge holes
        """
        # Get original image within leaf region
        masked_image = cv2.bitwise_and(image, image, mask=leaf_mask)
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Direct bright region detection
        _, bright_regions = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        bright_regions = cv2.bitwise_and(bright_regions, leaf_mask)
        
        # Method 2: Detect concavities in leaf boundary (edge holes)
        edge_holes = self.detect_edge_holes(leaf_mask)
        
        # Combine both methods
        combined_holes = cv2.bitwise_or(bright_regions, edge_holes)
        
        # Clean up with morphological operations
        kernel_small = np.ones((2,2), np.uint8)
        combined_holes = cv2.morphologyEx(combined_holes, cv2.MORPH_OPEN, kernel_small)
        combined_holes = cv2.morphologyEx(combined_holes, cv2.MORPH_CLOSE, kernel_small)
        
        # Find hole contours
        contours, _ = cv2.findContours(combined_holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter holes based on size
        min_hole_area = 15
        max_hole_area = cv2.countNonZero(leaf_mask) * 0.2
        
        hole_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_hole_area < area < max_hole_area:
                hole_contours.append(cnt)
        
        # Create final holes binary mask
        final_holes = np.zeros_like(combined_holes)
        cv2.fillPoly(final_holes, hole_contours, 255)
        
        return hole_contours, final_holes
    
    def detect_edge_holes(self, leaf_mask: np.ndarray) -> np.ndarray:
        """
        Detect holes at leaf edges by analyzing boundary topology
        """
        # Find leaf contour
        contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros_like(leaf_mask)
        
        main_contour = max(contours, key=cv2.contourArea)
        
        # Create a slightly dilated version of the leaf
        kernel = np.ones((10,10), np.uint8)
        dilated_leaf = cv2.dilate(leaf_mask, kernel, iterations=1)
        
        # Find the difference - this captures concave areas
        boundary_region = cv2.subtract(dilated_leaf, leaf_mask)
        
        # Apply morphological operations to clean up
        kernel_clean = np.ones((3,3), np.uint8)
        boundary_region = cv2.morphologyEx(boundary_region, cv2.MORPH_CLOSE, kernel_clean)
        
        # Find connected components in boundary region
        boundary_contours, _ = cv2.findContours(boundary_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for potential holes based on size and shape
        edge_holes_mask = np.zeros_like(leaf_mask)
        
        for cnt in boundary_contours:
            area = cv2.contourArea(cnt)
            if 20 < area < 2000:  # Reasonable hole size range
                # Check if it's a deep concavity (potential hole)
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                
                if hull_area > 0:
                    solidity = area / hull_area
                    if solidity < 0.8:  # High concavity indicates potential hole
                        cv2.fillPoly(edge_holes_mask, [cnt], 255)
        
        return edge_holes_mask

    def is_edge_hole(self, contour: np.ndarray, leaf_mask: np.ndarray) -> bool:
        """
        Check if a hole contour is at the edge of the leaf
        """
        # Create a more aggressively eroded leaf mask
        kernel = np.ones((15,15), np.uint8)
        eroded_leaf = cv2.erode(leaf_mask, kernel, iterations=1)
        
        # Create a mask for this contour
        contour_mask = np.zeros_like(leaf_mask)
        cv2.fillPoly(contour_mask, [contour], 255)
        
        # Check overlap with eroded interior
        overlap_with_interior = cv2.bitwise_and(contour_mask, eroded_leaf)
        interior_pixels = cv2.countNonZero(overlap_with_interior)
        total_pixels = cv2.countNonZero(contour_mask)
        
        # If less than 50% of hole is in deep interior, consider it an edge hole
        if total_pixels > 0:
            interior_ratio = interior_pixels / total_pixels
            return interior_ratio < 0.5
        
        return False
    
    def calculate_hole_ratio(self, leaf_mask: np.ndarray, holes_binary: np.ndarray) -> float:
        """
        Calculate ratio of hole area to leaf area
        """
        leaf_area = cv2.countNonZero(leaf_mask)
        hole_area = cv2.countNonZero(holes_binary)
        
        if leaf_area == 0:
            return 0.0
        
        ratio = hole_area / leaf_area
        return ratio
    
    def visualize_results(self, original: np.ndarray, enhanced: np.ndarray, 
                         leaf_mask: np.ndarray, holes_binary: np.ndarray, 
                         hole_contours: List[np.ndarray], hole_ratio: float):
        """
        Visualize detection results
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0,0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0,0].set_title('Original Image')
        axes[0,0].axis('off')
        
        # Enhanced image
        axes[0,1].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        axes[0,1].set_title('Enhanced Image')
        axes[0,1].axis('off')
        
        # Leaf segmentation
        axes[0,2].imshow(leaf_mask, cmap='gray')
        axes[0,2].set_title('Leaf Segmentation')
        axes[0,2].axis('off')
        
        # Hole detection
        axes[1,0].imshow(holes_binary, cmap='gray')
        axes[1,0].set_title('Detected Holes')
        axes[1,0].axis('off')
        
        # Result overlay
        result = enhanced.copy()
        cv2.drawContours(result, hole_contours, -1, (0, 0, 255), 2)
        axes[1,1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[1,1].set_title(f'Final Result\nHole Ratio: {hole_ratio:.2%}')
        axes[1,1].axis('off')
        
        # Statistics
        axes[1,2].text(0.1, 0.8, f'Leaf area: {cv2.countNonZero(leaf_mask)} pixels', transform=axes[1,2].transAxes)
        axes[1,2].text(0.1, 0.6, f'Hole area: {cv2.countNonZero(holes_binary)} pixels', transform=axes[1,2].transAxes)
        axes[1,2].text(0.1, 0.4, f'Hole count: {len(hole_contours)}', transform=axes[1,2].transAxes)
        axes[1,2].text(0.1, 0.2, f'Area ratio: {hole_ratio:.2%}', transform=axes[1,2].transAxes, fontweight='bold')
        axes[1,2].set_title('Statistics')
        axes[1,2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def process_image(self, image_path: str) -> float:
        """
        Process single image and return hole area ratio
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        print("Starting image processing...")
        
        # 1. Image enhancement
        print("1. Enhancing image...")
        enhanced = self.enhance_image_scan_effect(image)
        
        # 2. Leaf segmentation
        print("2. Segmenting leaf...")
        leaf_mask, leaf_contour = self.segment_leaf(enhanced)
        
        if leaf_mask is None:
            print("Error: Could not detect leaf")
            return 0.0
        
        # 3. Hole detection
        print("3. Detecting holes...")
        hole_contours, holes_binary = self.detect_holes(enhanced, leaf_mask)
        
        # 4. Calculate area ratio
        hole_ratio = self.calculate_hole_ratio(leaf_mask, holes_binary)
        
        print(f"Detection complete! Hole area ratio: {hole_ratio:.2%}")
        
        # 5. Visualize results
        if self.debug:
            self.visualize_results(image, enhanced, leaf_mask, holes_binary, hole_contours, hole_ratio)
        
        return hole_ratio

def main():
    """
    Main function example
    """
    detector = LeafHoleDetector()
    
    # Process image (replace with your image path)
    image_path = "leaf_sample.png"  # Replace with actual image path
    
    try:
        hole_ratio = detector.process_image(image_path)
        print(f"\nFinal result: Hole area occupies {hole_ratio:.2%} of leaf area")
    except Exception as e:
        print(f"Processing failed: {e}")

if __name__ == "__main__":
    main()
