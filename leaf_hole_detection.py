import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict

class LeafHoleDetector:
    def __init__(self):
        self.debug = True
        self.pixel_to_cm_ratio = None  # 像素到厘米的转换比例
        self.reference_square_size = 1.0  # 参照方块的实际尺寸（厘米）
    
    def detect_reference_square(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        """
        检测图像中的1cm×1cm参照方块（支持黑色和白色方块）
        返回: (方块轮廓, 像素边长)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 方法1：检测黑色方块（暗色区域）
        best_square, best_side_length = self._detect_dark_square(gray)
        if best_square is not None:
            if self.debug:
                print("Detected dark reference square")
            return best_square, best_side_length
        
        # 方法2：检测白色方块（亮色区域）
        best_square, best_side_length = self._detect_light_square(gray)
        if best_square is not None:
            if self.debug:
                print("Detected light reference square")
            return best_square, best_side_length
        
        # 方法3：边缘检测方法
        best_square, best_side_length = self._detect_edge_square(gray)
        if best_square is not None:
            if self.debug:
                print("Detected reference square using edge detection")
            return best_square, best_side_length
        
        return None
    
    def _detect_dark_square(self, gray: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """检测黑色/暗色参照方块"""
        # 使用Otsu阈值检测暗色区域
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 形态学操作清理噪声
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return self._find_square_contour(binary)
    
    def _detect_light_square(self, gray: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """检测白色/亮色参照方块"""
        # 使用自适应阈值检测亮色区域
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        return self._find_square_contour(binary)
    
    def _detect_edge_square(self, gray: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """使用边缘检测方法检测参照方块"""
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny边缘检测
        edges = cv2.Canny(blurred, 50, 150)
        
        # 形态学操作连接边缘
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return self._find_square_contour(edges)
    
    def _find_square_contour(self, binary: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """在二值图像中查找正方形轮廓"""
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_square = None
        best_side_length = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 过滤太小或太大的轮廓（调整范围以适应手绘方块）
            if area < 50 or area > 100000:
                continue
            
            # 计算轮廓的多边形近似
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 检查是否为四边形
            if len(approx) == 4:
                # 计算边长
                side_lengths = []
                for i in range(4):
                    p1 = approx[i][0]
                    p2 = approx[(i + 1) % 4][0]
                    length = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                    side_lengths.append(length)
                
                avg_side = np.mean(side_lengths)
                if avg_side < 10:  # 太小的不考虑
                    continue
                
                # 检查是否为正方形（放宽条件以适应手绘）
                side_variance = np.var(side_lengths) / (avg_side ** 2)
                if side_variance < 0.05:  # 边长变异系数小于5%
                    # 计算轮廓的紧密度
                    rect = cv2.boundingRect(contour)
                    rect_area = rect[2] * rect[3]
                    compactness = area / rect_area if rect_area > 0 else 0
                    
                    # 计算长宽比
                    aspect_ratio = max(rect[2], rect[3]) / min(rect[2], rect[3]) if min(rect[2], rect[3]) > 0 else float('inf')
                    
                    # 综合评分（考虑紧密度、长宽比和面积）
                    score = compactness * (1.0 / aspect_ratio) * np.sqrt(area)
                    
                    if compactness > 0.6 and aspect_ratio < 1.3 and score > best_score:
                        best_square = contour
                        best_side_length = avg_side
                        best_score = score
        
        return best_square, best_side_length
    
    def calculate_pixel_to_cm_ratio(self, reference_square_result: Tuple[np.ndarray, float]) -> float:
        """
        计算像素到厘米的转换比例
        """
        if reference_square_result is None:
            return None
        
        _, pixel_side_length = reference_square_result
        # 1cm对应的像素长度
        pixels_per_cm = pixel_side_length / self.reference_square_size
        return pixels_per_cm
    
    def convert_to_absolute_area(self, pixel_area: int, pixels_per_cm: float) -> float:
        """
        将像素面积转换为绝对面积（平方厘米）
        """
        if pixels_per_cm is None or pixels_per_cm <= 0:
            return 0.0
        
        # 像素面积转换为平方厘米
        cm_area = pixel_area / (pixels_per_cm ** 2)
        return cm_area
    
    def detect_reference_and_calculate_ratio(self, image: np.ndarray) -> Optional[float]:
        """
        检测参照方块并计算像素到厘米的转换比例
        """
        if self.debug:
            print("Searching for reference square...")
        
        reference_result = self.detect_reference_square(image)
        if reference_result is None:
            if self.debug:
                print("Warning: Could not detect reference square")
                print("Make sure your image contains a clear 1cm×1cm square (black or white)")
                print("The square should be well-defined and not overlapping with the leaf")
            return None
        
        pixels_per_cm = self.calculate_pixel_to_cm_ratio(reference_result)
        self.pixel_to_cm_ratio = pixels_per_cm
        
        if self.debug:
            contour, side_length = reference_result
            area = cv2.contourArea(contour)
            print(f"Reference square detected successfully!")
            print(f"  - Square area: {area:.0f} pixels")
            print(f"  - Side length: {side_length:.1f} pixels")
            print(f"  - Conversion ratio: {pixels_per_cm:.2f} pixels per cm")
        
        return pixels_per_cm
    
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
        
        # 新增：应用中值滤波来抑制细长的叶脉
        # 内核大小可以根据叶脉的粗细进行调整，5x5是一个比较通用的选择
        gray = cv2.medianBlur(gray, 5)
        
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
                # 新增：增加形状判断，过滤掉像叶脉一样的细长轮廓
                if self.is_likely_hole(cnt):
                    hole_contours.append(cnt)
        
        # Create final holes binary mask
        final_holes = np.zeros_like(combined_holes)
        cv2.fillPoly(final_holes, hole_contours, 255)
        
        return hole_contours, final_holes
    
    def is_likely_hole(self, contour: np.ndarray, min_circularity: float = 0.2) -> bool:
        """
        判断一个轮廓是否可能是一个洞，而不是细长的叶脉。
        洞通常形状较为紧凑、接近圆形，而叶脉则细长。
        我们使用"圆度"和"长宽比"作为判断依据。
        """
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return False

        area = cv2.contourArea(contour)
        if area == 0:
            return False
            
        # 圆度计算公式: (4 * pi * area) / (perimeter^2)
        # 圆的圆度为1，细长形状的圆度接近0
        circularity = (4 * np.pi * area) / (perimeter * perimeter)
        
        # 增加一个对细长形状的额外判断
        aspect_ratio = 0
        try:
            _, (width, height), _ = cv2.minAreaRect(contour)
            if min(width, height) > 0:
                aspect_ratio = max(width, height) / min(width, height)
        except:
            # 在某些退化的情况下，minAreaRect可能会失败
            pass
            
        # 如果形状非常细长（高长宽比），也认为是叶脉
        if aspect_ratio > 5.0:
            if self.debug:
                print(f"Filtered by aspect ratio: {aspect_ratio:.2f}")
            return False

        # 圆度阈值可以根据实际情况调整，这里设置为0.2
        if circularity < min_circularity:
            if self.debug:
                print(f"Filtered by circularity: {circularity:.2f}")
            return False
            
        return True
    
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
    
    def calculate_leaf_length(self, leaf_contour: np.ndarray) -> Tuple[float, Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        计算叶片长度，使用主轴长度法
        返回: (叶长像素值, ((起点x, 起点y), (终点x, 终点y)))
        """
        if leaf_contour is None:
            return 0.0, ((0, 0), (0, 0))
        
        # 确保轮廓有足够的点数
        if len(leaf_contour) < 5:
            if self.debug:
                print(f"Leaf contour has only {len(leaf_contour)} points, using fallback method")
            # 直接使用最大距离方法
            return self._calculate_max_distance(leaf_contour)
        
        # 方法1：使用最小外接椭圆的长轴
        try:
            # 计算最小外接椭圆
            ellipse = cv2.fitEllipse(leaf_contour)
            center, axes, angle = ellipse
            
            # 检查椭圆参数是否有效
            if any(np.isnan([center[0], center[1], axes[0], axes[1], angle])) or any(np.isinf([center[0], center[1], axes[0], axes[1], angle])):
                if self.debug:
                    print("Invalid ellipse parameters, using fallback method")
                return self._calculate_max_distance(leaf_contour)
            
            # 获取长轴长度（椭圆的长轴就是叶片的主轴）
            major_axis_length = max(axes)
            
            # 计算长轴的起点和终点坐标
            angle_rad = np.radians(angle)
            half_major = major_axis_length / 2
            
            # 如果长轴是水平方向更长，使用对应的轴
            if axes[0] > axes[1]:
                # 长轴沿着椭圆的主轴方向
                dx = half_major * np.cos(angle_rad)
                dy = half_major * np.sin(angle_rad)
            else:
                # 长轴沿着椭圆的次轴方向
                dx = half_major * np.cos(angle_rad + np.pi/2)
                dy = half_major * np.sin(angle_rad + np.pi/2)
            
            start_point = (int(center[0] - dx), int(center[1] - dy))
            end_point = (int(center[0] + dx), int(center[1] + dy))
            
            return major_axis_length, (start_point, end_point)
            
        except Exception as e:
            # 如果椭圆拟合失败，使用备用方法
            if self.debug:
                print(f"Ellipse fitting failed: {e}, using fallback method")
            return self._calculate_max_distance(leaf_contour)
    
    def _calculate_max_distance(self, leaf_contour: np.ndarray) -> Tuple[float, Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        计算轮廓上任意两点间的最大距离（备用方法）
        """
        if leaf_contour is None or len(leaf_contour) < 2:
            return 0.0, ((0, 0), (0, 0))
        
        max_distance = 0
        best_points = ((0, 0), (0, 0))
        
        # 为了提高效率，我们只检查轮廓上的部分点
        contour_points = leaf_contour.reshape(-1, 2)
        step = max(1, len(contour_points) // 50)  # 最多检查50个点
        
        for i in range(0, len(contour_points), step):
            for j in range(i + step, len(contour_points), step):
                p1 = contour_points[i]
                p2 = contour_points[j]
                distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                
                if distance > max_distance:
                    max_distance = distance
                    best_points = (tuple(p1), tuple(p2))
        
        return max_distance, best_points

    def calculate_leaf_width(self, leaf_contour: np.ndarray) -> Tuple[float, Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        计算叶片宽度，使用主轴垂直方向的最大距离
        返回: (叶宽像素值, ((起点x, 起点y), (终点x, 终点y)))
        """
        if leaf_contour is None:
            return 0.0, ((0, 0), (0, 0))
        
        # 确保轮廓有足够的点数
        if len(leaf_contour) < 5:
            if self.debug:
                print(f"Leaf contour has only {len(leaf_contour)} points, using fallback for width")
            # 对于点数不足的情况，返回一个估算的宽度（长度的一半）
            length, _ = self._calculate_max_distance(leaf_contour)
            return length * 0.5, ((0, 0), (0, 0))
        
        try:
            # 计算最小外接椭圆
            ellipse = cv2.fitEllipse(leaf_contour)
            center, axes, angle = ellipse
            
            # 检查椭圆参数是否有效
            if any(np.isnan([center[0], center[1], axes[0], axes[1], angle])) or any(np.isinf([center[0], center[1], axes[0], axes[1], angle])):
                if self.debug:
                    print("Invalid ellipse parameters for width, using fallback method")
                length, _ = self._calculate_max_distance(leaf_contour)
                return length * 0.5, ((0, 0), (0, 0))
            
            # 获取短轴长度（椭圆的短轴就是叶片的宽度）
            minor_axis_length = min(axes)
            
            # 计算短轴的起点和终点坐标
            angle_rad = np.radians(angle)
            half_minor = minor_axis_length / 2
            
            # 短轴垂直于长轴
            if axes[0] > axes[1]:
                # 短轴沿着椭圆的次轴方向
                dx = half_minor * np.cos(angle_rad + np.pi/2)
                dy = half_minor * np.sin(angle_rad + np.pi/2)
            else:
                # 短轴沿着椭圆的主轴方向
                dx = half_minor * np.cos(angle_rad)
                dy = half_minor * np.sin(angle_rad)
            
            start_point = (int(center[0] - dx), int(center[1] - dy))
            end_point = (int(center[0] + dx), int(center[1] + dy))
            
            return minor_axis_length, (start_point, end_point)
            
        except Exception as e:
            # 如果椭圆拟合失败，使用估算方法
            if self.debug:
                print(f"Ellipse fitting failed for width: {e}, using fallback method")
            length, _ = self._calculate_max_distance(leaf_contour)
            return length * 0.5, ((0, 0), (0, 0))
    
    def visualize_results(self, original: np.ndarray, enhanced: np.ndarray, 
                         leaf_mask: np.ndarray, holes_binary: np.ndarray, 
                         hole_contours: List[np.ndarray], hole_ratio: float,
                         pixels_per_cm: Optional[float] = None,
                         leaf_length_info: Optional[Tuple] = None,
                         leaf_width_info: Optional[Tuple] = None):
        """
        Visualize detection results with absolute measurements including leaf length
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image with reference square detection
        display_img = original.copy()
        if pixels_per_cm is not None:
            # Draw reference square if detected
            ref_result = self.detect_reference_square(original)
            if ref_result is not None:
                ref_contour, _ = ref_result
                # Draw reference square with thick blue outline
                cv2.drawContours(display_img, [ref_contour], -1, (255, 0, 0), 3)
                # Add text label
                rect = cv2.boundingRect(ref_contour)
                cv2.putText(display_img, '1cm', (rect[0], rect[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        axes[0,0].imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
        axes[0,0].set_title('Original Image' + (' (Reference Detected)' if pixels_per_cm else ''))
        axes[0,0].axis('off')
        
        # Enhanced image
        axes[0,1].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        axes[0,1].set_title('Enhanced Image')
        axes[0,1].axis('off')
        
        # Leaf segmentation with length and width lines
        leaf_display = cv2.cvtColor(leaf_mask, cv2.COLOR_GRAY2BGR)
        
        # Draw leaf length line
        if leaf_length_info:
            length_pixels, (start_pt, end_pt) = leaf_length_info
            cv2.line(leaf_display, start_pt, end_pt, (0, 255, 0), 2)  # Green line for length
            # Add length label
            mid_x = (start_pt[0] + end_pt[0]) // 2
            mid_y = (start_pt[1] + end_pt[1]) // 2
            cv2.putText(leaf_display, f'L:{length_pixels:.0f}px', (mid_x, mid_y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw leaf width line
        if leaf_width_info:
            width_pixels, (start_pt, end_pt) = leaf_width_info
            cv2.line(leaf_display, start_pt, end_pt, (255, 0, 0), 2)  # Blue line for width
            # Add width label
            mid_x = (start_pt[0] + end_pt[0]) // 2
            mid_y = (start_pt[1] + end_pt[1]) // 2
            cv2.putText(leaf_display, f'W:{width_pixels:.0f}px', (mid_x, mid_y+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        axes[0,2].imshow(leaf_display)
        axes[0,2].set_title('Leaf Segmentation with Measurements')
        axes[0,2].axis('off')
        
        # Hole detection
        axes[1,0].imshow(holes_binary, cmap='gray')
        axes[1,0].set_title('Detected Holes')
        axes[1,0].axis('off')
        
        # Result overlay
        result = enhanced.copy()
        cv2.drawContours(result, hole_contours, -1, (0, 0, 255), 2)
        
        # Also draw length and width lines on result
        if leaf_length_info:
            length_pixels, (start_pt, end_pt) = leaf_length_info
            cv2.line(result, start_pt, end_pt, (0, 255, 0), 2)
        if leaf_width_info:
            width_pixels, (start_pt, end_pt) = leaf_width_info
            cv2.line(result, start_pt, end_pt, (255, 0, 0), 2)
        
        axes[1,1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[1,1].set_title(f'Final Result\nHole Ratio: {hole_ratio:.2%}')
        axes[1,1].axis('off')
        
        # Statistics
        leaf_area_pixels = cv2.countNonZero(leaf_mask)
        hole_area_pixels = cv2.countNonZero(holes_binary)
        
        y_pos = 0.9
        axes[1,2].text(0.1, y_pos, f'Leaf area: {leaf_area_pixels} pixels', transform=axes[1,2].transAxes)
        y_pos -= 0.08
        axes[1,2].text(0.1, y_pos, f'Hole area: {hole_area_pixels} pixels', transform=axes[1,2].transAxes)
        y_pos -= 0.08
        axes[1,2].text(0.1, y_pos, f'Hole count: {len(hole_contours)}', transform=axes[1,2].transAxes)
        y_pos -= 0.08
        axes[1,2].text(0.1, y_pos, f'Area ratio: {hole_ratio:.2%}', transform=axes[1,2].transAxes, fontweight='bold')
        
        # Add leaf length and width info
        if leaf_length_info:
            length_pixels, _ = leaf_length_info
            y_pos -= 0.08
            axes[1,2].text(0.1, y_pos, f'Leaf length: {length_pixels:.1f} px', transform=axes[1,2].transAxes, color='green')
        
        if leaf_width_info:
            width_pixels, _ = leaf_width_info
            y_pos -= 0.08
            axes[1,2].text(0.1, y_pos, f'Leaf width: {width_pixels:.1f} px', transform=axes[1,2].transAxes, color='blue')
        
        if pixels_per_cm is not None:
            y_pos -= 0.1
            leaf_area_cm2 = self.convert_to_absolute_area(leaf_area_pixels, pixels_per_cm)
            hole_area_cm2 = self.convert_to_absolute_area(hole_area_pixels, pixels_per_cm)
            axes[1,2].text(0.1, y_pos, f'Leaf area: {leaf_area_cm2:.2f} cm²', transform=axes[1,2].transAxes, color='green', fontweight='bold')
            y_pos -= 0.08
            axes[1,2].text(0.1, y_pos, f'Hole area: {hole_area_cm2:.2f} cm²', transform=axes[1,2].transAxes, color='red', fontweight='bold')
            
            # Add absolute length and width measurements
            if leaf_length_info:
                length_pixels, _ = leaf_length_info
                length_cm = length_pixels / pixels_per_cm
                y_pos -= 0.08
                axes[1,2].text(0.1, y_pos, f'Leaf length: {length_cm:.2f} cm', transform=axes[1,2].transAxes, color='green', fontweight='bold')
            
            if leaf_width_info:
                width_pixels, _ = leaf_width_info
                width_cm = width_pixels / pixels_per_cm
                y_pos -= 0.08
                axes[1,2].text(0.1, y_pos, f'Leaf width: {width_cm:.2f} cm', transform=axes[1,2].transAxes, color='blue', fontweight='bold')
            
            y_pos -= 0.08
            axes[1,2].text(0.1, y_pos, f'Scale: {pixels_per_cm:.1f} px/cm', transform=axes[1,2].transAxes, color='black')
        
        axes[1,2].set_title('Statistics')
        axes[1,2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def process_image(self, image_path: str) -> Dict:
        """
        Process single image and return comprehensive results including absolute areas
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        print("Starting image processing...")
        
        # 0. Detect reference square first
        print("0. Detecting reference square...")
        pixels_per_cm = self.detect_reference_and_calculate_ratio(image)
        
        # 1. Image enhancement
        print("1. Enhancing image...")
        enhanced = self.enhance_image_scan_effect(image)
        
        # 2. Leaf segmentation
        print("2. Segmenting leaf...")
        leaf_mask, leaf_contour = self.segment_leaf(enhanced)
        
        if leaf_mask is None:
            print("Error: Could not detect leaf")
            return {'error': 'Could not detect leaf'}
        
        # 3. Hole detection
        print("3. Detecting holes...")
        hole_contours, holes_binary = self.detect_holes(enhanced, leaf_mask)
        
        # 4. Calculate leaf dimensions
        print("4. Calculating leaf dimensions...")
        try:
            leaf_length_info = self.calculate_leaf_length(leaf_contour)
            leaf_width_info = self.calculate_leaf_width(leaf_contour)
        except Exception as e:
            print(f"Error calculating leaf dimensions: {e}")
            # 提供默认值
            leaf_length_info = (0.0, ((0, 0), (0, 0)))
            leaf_width_info = (0.0, ((0, 0), (0, 0)))
        
        # 5. Calculate areas
        leaf_area_pixels = cv2.countNonZero(leaf_mask)
        hole_area_pixels = cv2.countNonZero(holes_binary)
        hole_ratio = self.calculate_hole_ratio(leaf_mask, holes_binary)
        
        # 6. Calculate absolute areas if reference square is detected
        # 确保叶长和叶宽数据总是有效的
        leaf_length_pixels = leaf_length_info[0] if leaf_length_info and len(leaf_length_info) > 0 else 0.0
        leaf_width_pixels = leaf_width_info[0] if leaf_width_info and len(leaf_width_info) > 0 else 0.0
        
        results = {
            'hole_ratio': hole_ratio,
            'leaf_area_pixels': leaf_area_pixels,
            'hole_area_pixels': hole_area_pixels,
            'hole_count': len(hole_contours),
            'pixels_per_cm': pixels_per_cm,
            'has_reference': pixels_per_cm is not None,
            'leaf_length_pixels': leaf_length_pixels,
            'leaf_width_pixels': leaf_width_pixels
        }
        
        if pixels_per_cm is not None:
            leaf_area_cm2 = self.convert_to_absolute_area(leaf_area_pixels, pixels_per_cm)
            hole_area_cm2 = self.convert_to_absolute_area(hole_area_pixels, pixels_per_cm)
            leaf_length_cm = leaf_length_pixels / pixels_per_cm if leaf_length_pixels > 0 else 0.0
            leaf_width_cm = leaf_width_pixels / pixels_per_cm if leaf_width_pixels > 0 else 0.0
            
            results.update({
                'leaf_area_cm2': leaf_area_cm2,
                'hole_area_cm2': hole_area_cm2,
                'leaf_length_cm': leaf_length_cm,
                'leaf_width_cm': leaf_width_cm,
                'reference_detected': True
            })
            
            print(f"Leaf area: {leaf_area_cm2:.2f} cm²")
            print(f"Hole area: {hole_area_cm2:.2f} cm²")
            print(f"Leaf length: {leaf_length_cm:.2f} cm")
            print(f"Leaf width: {leaf_width_cm:.2f} cm")
        else:
            results.update({
                'leaf_area_cm2': None,
                'hole_area_cm2': None,
                'leaf_length_cm': None,
                'leaf_width_cm': None,
                'reference_detected': False
            })
            print("No reference square detected - only pixel measurements available")
            print(f"Leaf length: {leaf_length_pixels:.1f} pixels")
            print(f"Leaf width: {leaf_width_pixels:.1f} pixels")
        
        print(f"Detection complete! Hole area ratio: {hole_ratio:.2%}")
        
        # 7. Store processed data for visualization
        results.update({
            'original_image': image,
            'enhanced_image': enhanced,
            'leaf_mask': leaf_mask,
            'holes_binary': holes_binary,
            'hole_contours': hole_contours,
            'leaf_length_info': leaf_length_info,
            'leaf_width_info': leaf_width_info
        })
        
        # 8. Visualize results
        if self.debug:
            self.visualize_results(image, enhanced, leaf_mask, holes_binary, hole_contours, 
                                 hole_ratio, pixels_per_cm, leaf_length_info, leaf_width_info)
        
        return results

    def debug_reference_detection(self, image: np.ndarray):
        """
        调试参照方块检测过程，显示各个步骤的结果
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 原始灰度图
        axes[0,0].imshow(gray, cmap='gray')
        axes[0,0].set_title('Original Grayscale')
        axes[0,0].axis('off')
        
        # 暗色检测
        _, dark_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        dark_binary = cv2.morphologyEx(dark_binary, cv2.MORPH_CLOSE, kernel)
        dark_binary = cv2.morphologyEx(dark_binary, cv2.MORPH_OPEN, kernel)
        
        axes[0,1].imshow(dark_binary, cmap='gray')
        axes[0,1].set_title('Dark Square Detection')
        axes[0,1].axis('off')
        
        # 亮色检测
        light_binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)
        
        axes[0,2].imshow(light_binary, cmap='gray')
        axes[0,2].set_title('Light Square Detection')
        axes[0,2].axis('off')
        
        # 边缘检测
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        axes[1,0].imshow(edges, cmap='gray')
        axes[1,0].set_title('Edge Detection')
        axes[1,0].axis('off')
        
        # 检测到的轮廓
        all_contours_img = image.copy()
        
        # 暗色轮廓
        dark_contours, _ = cv2.findContours(dark_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(all_contours_img, dark_contours, -1, (0, 255, 0), 2)  # 绿色
        
        # 亮色轮廓
        light_contours, _ = cv2.findContours(light_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(all_contours_img, light_contours, -1, (255, 0, 0), 2)  # 红色
        
        # 边缘轮廓
        edge_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(all_contours_img, edge_contours, -1, (0, 0, 255), 2)  # 蓝色
        
        axes[1,1].imshow(cv2.cvtColor(all_contours_img, cv2.COLOR_BGR2RGB))
        axes[1,1].set_title('All Detected Contours\n(Green=Dark, Red=Light, Blue=Edge)')
        axes[1,1].axis('off')
        
        # 最终检测结果
        final_result = image.copy()
        ref_result = self.detect_reference_square(image)
        if ref_result is not None:
            ref_contour, side_length = ref_result
            cv2.drawContours(final_result, [ref_contour], -1, (255, 0, 0), 3)
            rect = cv2.boundingRect(ref_contour)
            cv2.putText(final_result, f'1cm ({side_length:.1f}px)', (rect[0], rect[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            axes[1,2].set_title('Reference Square Detected!')
        else:
            axes[1,2].set_title('No Reference Square Found')
        
        axes[1,2].imshow(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB))
        axes[1,2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # 打印详细信息
        print("\n=== Reference Square Detection Debug Info ===")
        print(f"Image size: {image.shape[1]}x{image.shape[0]} pixels")
        print(f"Dark contours found: {len(dark_contours)}")
        print(f"Light contours found: {len(light_contours)}")
        print(f"Edge contours found: {len(edge_contours)}")
        
        if ref_result is not None:
            ref_contour, side_length = ref_result
            area = cv2.contourArea(ref_contour)
            rect = cv2.boundingRect(ref_contour)
            aspect_ratio = max(rect[2], rect[3]) / min(rect[2], rect[3])
            print(f"\nDetected square:")
            print(f"  - Area: {area:.0f} pixels")
            print(f"  - Side length: {side_length:.1f} pixels") 
            print(f"  - Bounding box: {rect[2]}x{rect[3]} pixels")
            print(f"  - Aspect ratio: {aspect_ratio:.2f}")
            print(f"  - Estimated scale: {side_length:.1f} pixels per cm")
        else:
            print("\nNo reference square detected!")
            print("Tips for better detection:")
            print("1. Make sure the square is clearly visible and not overlapping with the leaf")
            print("2. The square should have good contrast with the background")
            print("3. Try to make the square edges as straight as possible")
            print("4. Ensure the square is not too small (at least 50 pixels) or too large")

def main():
    """
    Main function example
    """
    import sys
    
    detector = LeafHoleDetector()
    
    # Process image (replace with your image path)
    image_path = "leaf_sample.png"  # Replace with actual image path
    
    # Check if debug mode is requested
    debug_reference = "--debug-reference" in sys.argv
    
    try:
        if debug_reference:
            # Debug reference square detection
            import cv2
            image = cv2.imread(image_path)
            if image is not None:
                print("Running reference square detection debug...")
                detector.debug_reference_detection(image)
            else:
                print(f"Could not load image: {image_path}")
        else:
            # Normal processing
            results = detector.process_image(image_path)
            if 'error' in results:
                print(f"Processing failed: {results['error']}")
            else:
                print(f"\nFinal result: Hole area occupies {results['hole_ratio']:.2%} of leaf area")
                if results['has_reference']:
                    print(f"Leaf area: {results['leaf_area_cm2']:.2f} cm²")
                    print(f"Hole area: {results['hole_area_cm2']:.2f} cm²")
                else:
                    print("\nTo get absolute area measurements, include a 1cm×1cm reference square in your image")
                    print("Run with --debug-reference to see the detection process")
    except Exception as e:
        print(f"Processing failed: {e}")

if __name__ == "__main__":
    main()
