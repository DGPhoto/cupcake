# src/analysis_engine.py

import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
import gc
import logging

# Import the new GPU Manager
from .gpu_utils import GPUManager

@dataclass
class ImageAnalysisResult:
    """Data class to store image analysis results."""
    # Technical quality metrics
    sharpness_score: float = 0.0
    exposure_score: float = 0.0
    contrast_score: float = 0.0
    noise_score: float = 0.0
    
    # Composition metrics
    composition_score: float = 0.0
    rule_of_thirds_score: float = 0.0
    symmetry_score: float = 0.0
    
    # Subject detection
    face_count: int = 0
    face_qualities: List[float] = field(default_factory=list)
    subject_position_score: float = 0.0
    
    # Overall metrics
    overall_technical_score: float = 0.0
    overall_composition_score: float = 0.0
    overall_score: float = 0.0
    
    # Raw analysis data for plugin use
    histogram_data: Optional[np.ndarray] = None
    edge_map: Optional[np.ndarray] = None
    face_locations: List[Tuple[int, int, int, int]] = field(default_factory=list)
    
    # Plugin extension point - additional scores from plugins
    plugin_scores: Dict[str, float] = field(default_factory=dict)

class AnalysisEngine:
    """Analyzes images for technical quality and composition metrics."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the analysis engine with configuration.
        
        Args:
            config: Dictionary of configuration parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger("cupcake.analysis")
        
        # Initialize GPU manager
        self.gpu_manager = GPUManager(suppress_tf_warnings=True)
        
        # Check if GPU support is available
        self.use_gpu = self.config.get('use_gpu', True)
        self.gpu_available = self.gpu_manager.is_gpu_available()
        
        if self.use_gpu:
            if self.gpu_available:
                self.logger.info(f"GPU support enabled. Using {self.gpu_manager.get_opencv_backend()} backend.")
            else:
                self.logger.info("GPU requested but not available. Using CPU fallback.")
        else:
            self.logger.info("GPU support disabled. Using CPU for image processing.")
            self.gpu_available = False
        
        # Initialize face detection with more reliable approach
        try:
            # Try to load a better face detector if available
            # In this case we're using the Haar cascade, but with better parameters
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # Add profile face detection for better coverage
            profile_cascade_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
            self.profile_cascade = cv2.CascadeClassifier(profile_cascade_path)
        except:
            self.face_cascade = None
            self.profile_cascade = None
            self.logger.warning("Face detection not available. OpenCV haar cascades not found.")
    
    def analyze_image(self, image_data: np.ndarray, metadata: Dict[str, Any] = None) -> ImageAnalysisResult:
        """
        Analyze an image and return quality metrics.
        
        Args:
            image_data: NumPy array containing the image data
            metadata: Optional metadata about the image
            
        Returns:
            ImageAnalysisResult object with analysis results
        """
        result = ImageAnalysisResult()
        
        # Ottimizzazione memoria: downsampling per immagini grandi
        max_dimension = 2500  # Limita dimensione massima per l'analisi
        original_shape = image_data.shape
        
        # Applica downsampling se necessario
        if len(image_data.shape) >= 2 and (image_data.shape[0] > max_dimension or image_data.shape[1] > max_dimension):
            # Calcola fattore di scala
            scale_factor = max_dimension / max(image_data.shape[0], image_data.shape[1])
            
            # Calcola nuove dimensioni
            new_height = int(image_data.shape[0] * scale_factor)
            new_width = int(image_data.shape[1] * scale_factor)
            
            # Ridimensiona l'immagine
            try:
                self.logger.debug(f"Downsampling immagine da {image_data.shape} a ({new_height}, {new_width})")
                # Use GPU manager for resizing
                image_data = self.gpu_manager.resize_image(
                    image_data, 
                    (new_width, new_height),
                    interpolation=cv2.INTER_AREA
                )
            except Exception as e:
                self.logger.warning(f"Errore nel ridimensionamento: {e}, usando l'immagine originale")
        
        # Converti a float32 se è float64 per risparmiare memoria
        if hasattr(image_data, 'dtype') and image_data.dtype == np.float64:
            self.logger.debug(f"Convertendo immagine da float64 a float32 per risparmiare memoria")
            image_data = image_data.astype(np.float32)
        
        # Check if image needs to be rotated based on metadata
        if metadata and 'orientation' in metadata:
            image_data = self._correct_orientation(image_data, metadata['orientation'])
        
        # Convert to grayscale for some analyses
        if len(image_data.shape) == 3 and image_data.shape[2] >= 3:
            grayscale = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
            
            # Also convert to LAB color space for better exposure analysis
            lab_image = cv2.cvtColor(image_data, cv2.COLOR_RGB2LAB)
        else:
            grayscale = image_data
            lab_image = None
        
        # Calculate technical metrics with improved methods
        result.sharpness_score = self._calculate_sharpness_improved(grayscale)
        result.exposure_score = self._calculate_exposure_improved(image_data, lab_image)
        result.contrast_score = self._calculate_contrast_improved(grayscale)
        result.noise_score = self._calculate_noise(grayscale)
        
        # Calculate composition metrics with more realistic scoring
        result.rule_of_thirds_score = self._analyze_rule_of_thirds_improved(grayscale)
        result.symmetry_score = self._analyze_symmetry(grayscale)
        
        # Detect faces with improved reliability
        face_locations = self._detect_faces_improved(grayscale)
        
        # Filter out likely false positives
        face_locations = self._filter_faces(face_locations, grayscale.shape)
        
        result.face_locations = face_locations
        result.face_count = len(face_locations)
        
        if result.face_count > 0:
            result.face_qualities = self._evaluate_face_qualities(grayscale, result.face_locations)
            result.subject_position_score = self._analyze_subject_position_improved(grayscale, result.face_locations)
        else:
            # If no faces found, use general subject detection
            result.subject_position_score = self._analyze_generic_subject_position(grayscale)
        
        # Store raw data for plugin use
        result.histogram_data = self._calculate_histogram(image_data)
        result.edge_map = self._calculate_edge_map(grayscale)
        
        # Calculate overall scores with more balanced weights
        result.overall_technical_score = self._calculate_overall_technical_improved(result)
        result.overall_composition_score = self._calculate_overall_composition(result)
        result.overall_score = self._calculate_overall_score_improved(result)
        
        # Libera memoria esplicitamente alla fine dell'analisi
        del grayscale
        if lab_image is not None:
            del lab_image
        gc.collect()
        
        return result
    
    def _correct_orientation(self, image_data: np.ndarray, orientation: int) -> np.ndarray:
        """
        Correct image orientation based on EXIF data.
        
        Args:
            image_data: Original image data
            orientation: EXIF orientation value
            
        Returns:
            Correctly oriented image
        """
        # EXIF orientation values:
        # 1: Normal (no rotation)
        # 3: Rotated 180°
        # 6: Rotated 90° CW
        # 8: Rotated 270° CW
        
        if orientation == 1:
            return image_data
        elif orientation == 3:
            return cv2.rotate(image_data, cv2.ROTATE_180)
        elif orientation == 6:
            return cv2.rotate(image_data, cv2.ROTATE_90_CLOCKWISE)
        elif orientation == 8:
            return cv2.rotate(image_data, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Default: no rotation
        return image_data
    
    def _calculate_sharpness_improved(self, grayscale: np.ndarray) -> float:
        """
        Calculate image sharpness using an improved method that combines
        multiple approaches for more reliable results.
        """
        # Use GPU manager to apply filters
        if self.use_gpu and self.gpu_available:
            try:
                # Laplacian filter
                laplacian = self.gpu_manager.apply_filter(grayscale, 'laplacian')
                
                # Sobel filters
                sobelx = self.gpu_manager.apply_filter(grayscale, 'sobel_x')
                sobely = self.gpu_manager.apply_filter(grayscale, 'sobel_y')
                
                # Gaussian blur for high-pass
                blur = self.gpu_manager.apply_filter(
                    grayscale.astype(np.float32), 
                    'gaussian', 
                    ksize=(9, 9), 
                    sigma=0
                )
            except Exception as e:
                self.logger.warning(f"Errore nell'elaborazione GPU per sharpness: {e}. Fallback a CPU.")
                # Fallback a CPU
                laplacian = cv2.Laplacian(grayscale, cv2.CV_64F)
                sobelx = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=3)
                blur = cv2.GaussianBlur(grayscale.astype(np.float32), (9, 9), 0)
        else:
            # Metodo CPU standard
            laplacian = cv2.Laplacian(grayscale, cv2.CV_64F)
            sobelx = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=3)
            blur = cv2.GaussianBlur(grayscale.astype(np.float32), (9, 9), 0)
        
        # Calcoli standard indipendenti dall'uso di GPU/CPU
        lap_variance = np.var(laplacian)
        
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        sobel_score = np.mean(sobel_magnitude)
        
        high_pass = grayscale.astype(np.float32) - blur
        high_pass_score = np.std(high_pass)
        
        # Normalize each score from 0-100
        lap_normalized = min(100, max(0, lap_variance / 1000 * 100))
        sobel_normalized = min(100, max(0, sobel_score / 20 * 100))
        highpass_normalized = min(100, max(0, high_pass_score / 20 * 100))
        
        # Combine scores with weights
        sharpness_score = (0.5 * lap_normalized + 
                          0.3 * sobel_normalized + 
                          0.2 * highpass_normalized)
        
        # Apply a more intuitive curve - severe penalties for very low sharpness,
        # but diminishing returns for very high sharpness
        adjusted_score = np.power(sharpness_score / 100, 0.7) * 100
        
        # Libera memoria
        del laplacian, sobelx, sobely, sobel_magnitude, blur, high_pass
        
        return adjusted_score
    
    def _calculate_exposure_improved(self, image_data: np.ndarray, lab_image: np.ndarray = None) -> float:
        """
        Calculate exposure quality with an improved approach that considers
        luminance distribution and highlight/shadow clipping.
        """
        # For color images, use L channel from LAB for luminance if available
        if lab_image is not None:
            # Extract L channel (lightness)
            L = lab_image[:,:,0]
            
            # L channel ranges from 0-255 where 0 is black and 255 is white
            # Calculate histogram
            hist = cv2.calcHist([L], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()  # Normalize
            
            # Analyze luminance distribution
            mean_L = np.mean(L)
            std_L = np.std(L)
            
            # Check for highlight clipping (percentage of pixels near max value)
            highlight_threshold = 250  # Near white
            highlight_clipping = np.sum(L >= highlight_threshold) / L.size
            
            # Check for shadow clipping (percentage of pixels near min value)
            shadow_threshold = 5  # Near black
            shadow_clipping = np.sum(L <= shadow_threshold) / L.size
            
            # Calculate exposure quality metrics
            
            # Ideal mean would be around 115-120 (slightly darker than middle)
            # for more natural looking images
            ideal_mean = 118
            mean_score = 100 - min(100, abs(mean_L - ideal_mean) * 1.5)
            
            # Standard deviation should be significant but not extreme
            # Too low: flat, boring image, too high: likely excessive contrast
            ideal_std_min, ideal_std_max = 40, 70
            if std_L < ideal_std_min:
                std_score = (std_L / ideal_std_min) * 100
            elif std_L > ideal_std_max:
                std_score = max(0, 100 - ((std_L - ideal_std_max) / ideal_std_max) * 100)
            else:
                std_score = 100
                
            # Penalize for clipped highlights and shadows
            highlight_penalty = min(100, highlight_clipping * 500)
            shadow_penalty = min(100, shadow_clipping * 500)
            clipping_score = 100 - max(highlight_penalty, shadow_penalty)
            
            # Final exposure score is a weighted combination
            exposure_score = (0.4 * mean_score + 
                             0.3 * std_score + 
                             0.3 * clipping_score)
                             
        else:
            # Fallback to grayscale method
            if len(image_data.shape) == 3 and image_data.shape[2] >= 3:
                gray = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_data
                
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()  # Normalize
            
            mean = np.mean(gray)
            std = np.std(gray)
            
            # Ideal mean would be around 115-120
            mean_score = 100 - min(100, abs(mean - 118) * 1.5)
            
            # Penalize if std is too low or too high
            if std < 40:
                std_score = (std / 40) * 100
            elif std > 70:
                std_score = max(0, 100 - ((std - 70) / 70) * 100)
            else:
                std_score = 100
            
            # Check for clipping
            highlight_clipping = np.sum(gray >= 250) / gray.size
            shadow_clipping = np.sum(gray <= 5) / gray.size
            
            highlight_penalty = min(100, highlight_clipping * 500)
            shadow_penalty = min(100, shadow_clipping * 500)
            clipping_score = 100 - max(highlight_penalty, shadow_penalty)
            
            exposure_score = (0.4 * mean_score + 
                             0.3 * std_score + 
                             0.3 * clipping_score)
            
            # Libera memoria
            if 'gray' in locals() and gray is not image_data:
                del gray
        
        # Libera memoria
        if 'hist' in locals():
            del hist
        
        return exposure_score
    
    def _calculate_contrast_improved(self, grayscale: np.ndarray) -> float:
        """
        Calculate image contrast using histogram analysis and local contrast.
        """
        # Global contrast: analyze histogram
        hist = cv2.calcHist([grayscale], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()  # Normalize
        
        # Calculate percentiles for more robust min/max
        cumsum = np.cumsum(hist)
        p2 = np.searchsorted(cumsum, 0.02)  # 2nd percentile
        p98 = np.searchsorted(cumsum, 0.98)  # 98th percentile
        
        # Dynamic range as difference between percentiles
        dynamic_range = p98 - p2
        
        # Calculate local contrast using GPU manager if available
        if self.use_gpu and self.gpu_available:
            try:
                sobelx = self.gpu_manager.apply_filter(grayscale, 'sobel_x')
                sobely = self.gpu_manager.apply_filter(grayscale, 'sobel_y')
            except Exception as e:
                self.logger.warning(f"Errore nell'elaborazione GPU per contrast: {e}. Fallback a CPU.")
                sobelx = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=3)
        else:
            sobelx = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=3)
        
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Calculate mean gradient magnitude, excluding very low values
        # to avoid counting flat areas
        local_contrast = np.mean(sobel_magnitude[sobel_magnitude > 10])
        
        # Combine global and local contrast
        # Normalize scores to 0-100
        global_score = min(100, dynamic_range / 180 * 100)
        
        # Avoid division by zero by adding a small epsilon
        if np.isnan(local_contrast) or local_contrast < 1e-6:
            local_score = 0
        else:
            local_score = min(100, local_contrast / 30 * 100)
        
        # Final contrast score
        contrast_score = 0.6 * global_score + 0.4 * local_score
        
        # Libera memoria
        del hist, sobelx, sobely, sobel_magnitude
        
        return contrast_score
    
    def _calculate_noise(self, grayscale: np.ndarray) -> float:
        """
        Estimate image noise level.
        Returns a score where higher means less noise.
        """
        # Apply more sophisticated noise estimation
        
        # Use GPU manager if available
        if self.use_gpu and self.gpu_available:
            try:
                # Apply filters using GPU
                denoised_median = cv2.medianBlur(grayscale, 3)  # No GPU version in OpenCV
                denoised_gauss = self.gpu_manager.apply_filter(grayscale, 'gaussian', ksize=(5, 5), sigma=0)
                
                # Calculate differences
                noise_median = cv2.absdiff(grayscale, denoised_median)
                noise_gauss = cv2.absdiff(grayscale, denoised_gauss)
                
                # Get Canny edges for texture detection
                edges = self.gpu_manager.apply_filter(
                    grayscale, 
                    'canny', 
                    low_threshold=50, 
                    high_threshold=150
                )
                
            except Exception as e:
                self.logger.warning(f"Errore nell'elaborazione GPU per noise: {e}. Fallback a CPU.")
                # Fallback a CPU
                denoised_median = cv2.medianBlur(grayscale, 3)
                denoised_gauss = cv2.GaussianBlur(grayscale, (5, 5), 0)
                noise_median = cv2.absdiff(grayscale, denoised_median)
                noise_gauss = cv2.absdiff(grayscale, denoised_gauss)
                edges = cv2.Canny(grayscale, 50, 150)
        else:
            # Metodo CPU standard
            denoised_median = cv2.medianBlur(grayscale, 3)
            denoised_gauss = cv2.GaussianBlur(grayscale, (5, 5), 0)
            noise_median = cv2.absdiff(grayscale, denoised_median)
            noise_gauss = cv2.absdiff(grayscale, denoised_gauss)
            edges = cv2.Canny(grayscale, 50, 150)
        
        # Calcoli standard indipendenti dall'uso di GPU/CPU
        noise_level_median = np.mean(noise_median)
        noise_level_gauss = np.mean(noise_gauss)
        
        # Combined noise level (median is more sensitive to outliers/noise)
        noise_level = 0.7 * noise_level_median + 0.3 * noise_level_gauss
        
        # Account for image content - textures can be mistaken for noise
        # Check if image has a lot of edges (textured)
        edge_ratio = np.count_nonzero(edges) / edges.size
        
        # Adjust noise level based on texture - textured images get a "discount"
        # on their noise penalty since some of the "noise" is likely texture
        texture_factor = 1.0 - min(0.5, edge_ratio * 10)  # Max 50% reduction
        adjusted_noise_level = noise_level * texture_factor
        
        # Convert to score (0-100) where higher is better (less noise)
        noise_score = max(0, 100 - (adjusted_noise_level * 10))
        
        # Libera memoria
        del denoised_median, noise_median
        del denoised_gauss, noise_gauss
        del edges
        
        return noise_score
    
    def _analyze_rule_of_thirds_improved(self, grayscale: np.ndarray) -> float:
        """
        Analyze how well the image follows the rule of thirds
        with a more realistic scoring system.
        """
        height, width = grayscale.shape
        
        # Define rule of thirds lines with more precision
        h_lines = [height / 3, 2 * height / 3]
        v_lines = [width / 3, 2 * width / 3]
        
        # Use GPU manager for Sobel filters if available
        if self.use_gpu and self.gpu_available:
            try:
                sobelx = self.gpu_manager.apply_filter(grayscale, 'sobel_x')
                sobely = self.gpu_manager.apply_filter(grayscale, 'sobel_y')
            except Exception as e:
                self.logger.warning(f"Errore nell'elaborazione GPU per rule of thirds: {e}. Fallback a CPU.")
                # Fallback a CPU
                sobelx = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=3)
        else:
            # Metodo CPU standard
            sobelx = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Threshold gradient map to find significant edges/features
        mean_mag = np.mean(magnitude)
        std_mag = np.std(magnitude)
        thresh = mean_mag + std_mag
        significant_features = magnitude > thresh
        
        # Create a mask with weighted "interest" at rule of thirds points
        interest_map = np.zeros_like(magnitude, dtype=np.float32)
        
        # Determine line thickness based on image size (proportional)
        line_thickness = max(width, height) // 40
        
        # Add weight to the rule of thirds lines
        for h in h_lines:
            h_min = max(0, int(h - line_thickness))
            h_max = min(height-1, int(h + line_thickness))
            interest_map[h_min:h_max, :] = 1.0
            
        for v in v_lines:
            v_min = max(0, int(v - line_thickness))
            v_max = min(width-1, int(v + line_thickness))
            interest_map[:, v_min:v_max] = 1.0
        
        # Add extra weight to intersection points (power points)
        for h in h_lines:
            for v in v_lines:
                h_min = max(0, int(h - line_thickness*1.5))
                h_max = min(height-1, int(h + line_thickness*1.5))
                v_min = max(0, int(v - line_thickness*1.5))
                v_max = min(width-1, int(v + line_thickness*1.5))
                
                # Create a weighted circle for intersection points
                for i in range(h_min, h_max+1):
                    for j in range(v_min, v_max+1):
                        dist = np.sqrt((i-h)**2 + (j-v)**2)
                        if dist <= line_thickness*1.5:
                            weight = 1.0 - (dist / (line_thickness*1.5))
                            interest_map[i, j] = max(interest_map[i, j], weight*2)  # Higher weight
        
        # Calculate how much significant features align with rule of thirds
        thirds_alignment = np.sum(significant_features * interest_map)
        total_significant = np.sum(significant_features)
        
        # Avoid division by zero
        if total_significant < 1:
            return 50.0  # Default if no significant features
        
        # Calculate alignment score
        alignment_ratio = thirds_alignment / total_significant
        
        # Nonlinear scaling to make the score more realistic
        # Most photos won't perfectly align with rule of thirds
        rule_thirds_score = 50 + min(40, alignment_ratio * 150)
        
        # Bonus for having strong features at power points
        power_points = []
        for h in h_lines:
            for v in v_lines:
                power_points.append((int(h), int(v)))
        
        power_point_bonus = 0
        for pp_h, pp_v in power_points:
            # Check a small region around each power point
            region_size = max(width, height) // 30
            h_min = max(0, pp_h - region_size)
            h_max = min(height-1, pp_h + region_size)
            v_min = max(0, pp_v - region_size)
            v_max = min(width-1, pp_v + region_size)
            
            region = significant_features[h_min:h_max, v_min:v_max]
            if np.any(region):
                power_point_bonus += 2.5  # Each power point with a feature adds bonus
        
        final_score = min(100, rule_thirds_score + power_point_bonus)
        
        # Libera memoria
        del sobelx, sobely, magnitude, significant_features, interest_map
        
        return final_score
    
    def _analyze_symmetry(self, grayscale: np.ndarray) -> float:
        """
        Analyze image symmetry with improvements to handle
        partial symmetry more accurately.
        """
        height, width = grayscale.shape
        
        # Check horizontal symmetry
        left = grayscale[:, :width//2]
        right = np.fliplr(grayscale[:, width//2:])
        
        # Resize to match dimensions if needed
        min_width = min(left.shape[1], right.shape[1])
        left = left[:, :min_width]
        right = right[:, :min_width]
                
        h_diff = np.abs(left.astype(float) - right.astype(float))
        
        # Weight the center of the image more strongly than the edges
        # for symmetry evaluation
        h_weight = np.ones_like(h_diff)
        for i in range(h_diff.shape[1]):
            h_weight[:, i] = 1.0 - 0.5 * (i / h_diff.shape[1])
        
        h_symmetry = 1.0 - (np.mean(h_diff * h_weight) / 255)
        
        # Check vertical symmetry
        top = grayscale[:height//2, :]
        bottom = np.flipud(grayscale[height//2:, :])
        
        # Resize to match dimensions if needed
        min_height = min(top.shape[0], bottom.shape[0])
        top = top[:min_height, :]
        bottom = bottom[:min_height, :]
                
        v_diff = np.abs(top.astype(float) - bottom.astype(float))
        
        # Weight the center more strongly
        v_weight = np.ones_like(v_diff)
        for i in range(v_diff.shape[0]):
            v_weight[i, :] = 1.0 - 0.5 * (i / v_diff.shape[0])
        
        v_symmetry = 1.0 - (np.mean(v_diff * v_weight) / 255)
        
        # Take the maximum of horizontal and vertical symmetry
        # This assumes the image is either horizontally or vertically symmetric
        # But use proper scaling
        h_score = h_symmetry ** 0.5 * 100  # Power of 0.5 for more realistic scaling
        v_score = v_symmetry ** 0.5 * 100
        
        symmetry_score = max(h_score, v_score)
        
        # Normalize to 0-100
        symmetry_score = min(100, symmetry_score)
        
        # Libera memoria
        del left, right, h_diff, h_weight
        del top, bottom, v_diff, v_weight
        
        return symmetry_score
    
    def _detect_faces_improved(self, grayscale: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Improved face detection that combines multiple detectors
        and reduces false positives.
        """
        if self.face_cascade is None:
            return []
        
        # Prepare image - enhance contrast to improve detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(grayscale)
        
        # Detect frontal faces with improved parameters
        frontal_faces = self.face_cascade.detectMultiScale(
            enhanced,
            scaleFactor=1.1,
            minNeighbors=6,  # Higher number to reduce false positives
            minSize=(40, 40),  # Larger minimum size
            maxSize=(grayscale.shape[1]//2, grayscale.shape[0]//2)  # Maximum size limitation
        )
        
        # Convert to list to ensure we can append to it
        all_faces = []
        for face in frontal_faces:
            all_faces.append(tuple(face))
        
        # Detect profile faces if available
        if self.profile_cascade is not None:
            # Detect profile faces
            profile_faces = self.profile_cascade.detectMultiScale(
                enhanced,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(40, 40),
                maxSize=(grayscale.shape[1]//2, grayscale.shape[0]//2)
            )
            
            # Add profile faces to the list
            for face in profile_faces:
                all_faces.append(tuple(face))
                
            # Also check flipped image for profiles facing the other way
            flipped = cv2.flip(enhanced, 1)
            flipped_faces = self.profile_cascade.detectMultiScale(
                flipped,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(40, 40),
                maxSize=(grayscale.shape[1]//2, grayscale.shape[0]//2)
            )
            
            # Correct the x-coordinate for the flipped image and add to list
            for face in flipped_faces:
                x, y, w, h = face
                all_faces.append((grayscale.shape[1] - x - w, y, w, h))
        
        # Libera memoria
        del enhanced
        if 'flipped' in locals():
            del flipped
        
        return self._merge_overlapping_faces(all_faces)
    
    def _merge_overlapping_faces(self, faces: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """
        Merge overlapping face detections to avoid double counting.
        """
        if not faces:
            return []
            
        # Convert to a more manageable format
        faces_list = [(x, y, x+w, y+h) for (x, y, w, h) in faces]
        
        # Sort by area (larger first)
        faces_list.sort(key=lambda face: (face[2] - face[0]) * (face[3] - face[1]), reverse=True)
        
        merged_faces = []
        
        while faces_list:
            current_face = faces_list.pop(0)
            
            # Check if this face overlaps significantly with any in merged_faces
            overlaps = False
            
            for i, (x1, y1, x2, y2) in enumerate(merged_faces):
                # Calculate overlap area
                overlap_x1 = max(current_face[0], x1)
                overlap_y1 = max(current_face[1], y1)
                overlap_x2 = min(current_face[2], x2)
                overlap_y2 = min(current_face[3], y2)
                
                # Check if there's an overlap
                if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                    current_area = (current_face[2] - current_face[0]) * (current_face[3] - current_face[1])
                    merged_area = (x2 - x1) * (y2 - y1)
                    
                    # If significant overlap (>50% of the smaller face)
                    if overlap_area > 0.5 * min(current_area, merged_area):
                        # Merge the faces
                        merged_faces[i] = (
                            min(x1, current_face[0]),
                            min(y1, current_face[1]),
                            max(x2, current_face[2]),
                            max(y2, current_face[3])
                        )
                        overlaps = True
                        break
            
            # If no significant overlap, add this face
            if not overlaps:
                merged_faces.append(current_face)
        
        # Convert back to (x, y, w, h) format
        return [(x1, y1, x2-x1, y2-y1) for (x1, y1, x2, y2) in merged_faces]