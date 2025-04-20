# src/analysis_engine.py

import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
import gc
import logging

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
        
        # Flag per indicare se la GPU è disponibile e in uso
        self.gpu_available = False
        
        # Verifica se utilizzare la GPU
        self.use_gpu = self.config.get('use_gpu', False)
        
        if self.use_gpu:
            try:
                # Verifica se OpenCV può utilizzare CUDA
                cv_build_info = cv2.getBuildInformation()
                if 'CUDA' in cv_build_info and 'YES' in cv_build_info[cv_build_info.find('CUDA'):cv_build_info.find('\n', cv_build_info.find('CUDA'))]:
                    self.logger.info("OpenCV con supporto CUDA disponibile. GPU abilitata.")
                    # Imposta OpenCV per usare CUDA quando possibile
                    cv2.setUseOptimized(True)
                    self.gpu_available = True
                else:
                    self.logger.warning("OpenCV senza supporto CUDA. GPU richiesta ma non disponibile.")
                    
                # Verifica la disponibilità di librerie GPU aggiuntive
                try:
                    import tensorflow as tf
                    gpus = tf.config.list_physical_devices('GPU')
                    if gpus:
                        self.logger.info(f"TensorFlow con supporto GPU disponibile. {len(gpus)} GPU trovate.")
                        # Consenti crescita di memoria dinamica sulla GPU
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                        self.gpu_available = True
                except ImportError:
                    self.logger.debug("TensorFlow non installato, saltando il controllo GPU di TensorFlow.")
                except Exception as e:
                    self.logger.warning(f"Errore nel controllare le GPU TensorFlow: {e}")
                    
            except Exception as e:
                self.logger.warning(f"Errore nell'inizializzazione GPU: {e}")
                self.use_gpu = False
        
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
                # Usa la GPU se disponibile per il ridimensionamento
                if self.use_gpu and self.gpu_available and hasattr(cv2, 'cuda'):
                    gpu_img = cv2.cuda_GpuMat()
                    gpu_img.upload(image_data)
                    gpu_resized = cv2.cuda.resize(gpu_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    image_data = gpu_resized.download()
                else:
                    # Metodo CPU standard
                    if len(image_data.shape) == 3 and image_data.shape[2] >= 3:
                        self.logger.debug(f"Downsampling immagine da {image_data.shape} a ({new_height}, {new_width}, {image_data.shape[2]})")
                        image_data = cv2.resize(image_data, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    else:
                        self.logger.debug(f"Downsampling immagine da {image_data.shape} a ({new_height}, {new_width})")
                        image_data = cv2.resize(image_data, (new_width, new_height), interpolation=cv2.INTER_AREA)
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
        # Usa CUDA se disponibile
        if self.use_gpu and self.gpu_available and hasattr(cv2, 'cuda'):
            try:
                # Carica l'immagine sulla GPU
                gpu_grayscale = cv2.cuda_GpuMat()
                gpu_grayscale.upload(grayscale)
                
                # Laplacian
                gpu_laplacian = cv2.cuda.createLaplacianFilter(cv2.CV_64F, 1)
                gpu_result = gpu_laplacian.apply(gpu_grayscale)
                laplacian = gpu_result.download()
                
                # Sobel X e Y
                gpu_sobelx = cv2.cuda.createSobelFilter(cv2.CV_64F, 1, 0)
                gpu_sobely = cv2.cuda.createSobelFilter(cv2.CV_64F, 0, 1)
                
                gpu_result_x = gpu_sobelx.apply(gpu_grayscale)
                gpu_result_y = gpu_sobely.apply(gpu_grayscale)
                
                sobelx = gpu_result_x.download()
                sobely = gpu_result_y.download()
                
                # Blur per high-pass
                gpu_blur = cv2.cuda.createGaussianFilter(cv2.CV_32F, cv2.CV_32F, (9, 9), 0)
                gpu_grayscale_float = cv2.cuda_GpuMat()
                gpu_grayscale_float.upload(grayscale.astype(np.float32))
                
                gpu_blur_result = gpu_blur.apply(gpu_grayscale_float)
                blur = gpu_blur_result.download()
                
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
        
        # Calculate local contrast - tenta di usare la GPU se disponibile
        if self.use_gpu and self.gpu_available and hasattr(cv2, 'cuda'):
            try:
                # Carica l'immagine sulla GPU
                gpu_grayscale = cv2.cuda_GpuMat()
                gpu_grayscale.upload(grayscale)
                
                # Sobel X e Y
                gpu_sobelx = cv2.cuda.createSobelFilter(cv2.CV_64F, 1, 0)
                gpu_sobely = cv2.cuda.createSobelFilter(cv2.CV_64F, 0, 1)
                
                gpu_result_x = gpu_sobelx.apply(gpu_grayscale)
                gpu_result_y = gpu_sobely.apply(gpu_grayscale)
                
                sobelx = gpu_result_x.download()
                sobely = gpu_result_y.download()
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
        
        # Tenta di usare la GPU se disponibile
        if self.use_gpu and self.gpu_available and hasattr(cv2, 'cuda'):
            try:
                # Carica l'immagine sulla GPU
                gpu_grayscale = cv2.cuda_GpuMat()
                gpu_grayscale.upload(grayscale)
                
                # Filtri di denoising
                gpu_median = cv2.cuda.createMedianFilter(cv2.CV_8U, 3)
                gpu_gauss = cv2.cuda.createGaussianFilter(cv2.CV_8U, cv2.CV_8U, (5, 5), 0)
                
                # Applica filtri
                gpu_denoised_median = gpu_median.apply(gpu_grayscale)
                gpu_denoised_gauss = gpu_gauss.apply(gpu_grayscale)
                
                # Scarica risultati
                denoised_median = gpu_denoised_median.download()
                denoised_gauss = gpu_denoised_gauss.download()
                
                # Calcola differenze
                noise_median = cv2.absdiff(grayscale, denoised_median)
                noise_gauss = cv2.absdiff(grayscale, denoised_gauss)
                
                # Calcola Canny per la texture
                gpu_edges = cv2.cuda.createCannyEdgeDetector(50, 150)
                gpu_edges_result = gpu_edges.detect(gpu_grayscale)
                edges = gpu_edges_result.download()
                
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
        
        # Tenta di usare la GPU se disponibile
        if self.use_gpu and self.gpu_available and hasattr(cv2, 'cuda'):
            try:
                # Carica l'immagine sulla GPU
                gpu_grayscale = cv2.cuda_GpuMat()
                gpu_grayscale.upload(grayscale)
                
                # Sobel X e Y
                gpu_sobelx = cv2.cuda.createSobelFilter(cv2.CV_64F, 1, 0)
                gpu_sobely = cv2.cuda.createSobelFilter(cv2.CV_64F, 0, 1)
                
                gpu_result_x = gpu_sobelx.apply(gpu_grayscale)
                gpu_result_y = gpu_sobely.apply(gpu_grayscale)
                
                sobelx = gpu_result_x.download()
                sobely = gpu_result_y.download()
                
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
    def _filter_faces(self, face_locations: List[Tuple[int, int, int, int]], 
                     image_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        """
        Filter out likely false positive face detections.
        """
        filtered_faces = []
        height, width = image_shape
        image_area = height * width
        
        for (x, y, w, h) in face_locations:
            face_area = w * h
            
            # Filter based on several criteria:
            
            # 1. Size relative to image (reject faces that are too small or too large)
            relative_size = face_area / image_area
            if relative_size < 0.005 or relative_size > 0.5:
                continue
                
            # 2. Aspect ratio (most faces have aspect ratios between 0.8 and 1.4)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 0.7 or aspect_ratio > 1.5:
                continue
                
            # 3. Position (penalize faces near edges)
            edge_margin = min(width, height) * 0.05
            if (x < edge_margin or y < edge_margin or 
                x + w > width - edge_margin or y + h > height - edge_margin):
                # Not filtered, but we could de-prioritize if needed
                pass
            
            filtered_faces.append((x, y, w, h))
        
        return filtered_faces
    
    def _evaluate_face_qualities(self, grayscale: np.ndarray, 
                               face_locations: List[Tuple[int, int, int, int]]) -> List[float]:
        """
        Evaluate the quality of detected faces with improved metrics.
        
        Returns:
            List of quality scores for each face
        """
        qualities = []
        
        for (x, y, w, h) in face_locations:
            # Extract face region
            face_roi = grayscale[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                qualities.append(0.0)
                continue
                
            # Calculate face sharpness using edge detection
            laplacian = cv2.Laplacian(face_roi, cv2.CV_64F)
            sharpness = np.var(laplacian)
            
            # Calculate face contrast using percentile method
            hist = cv2.calcHist([face_roi], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()
            cumsum = np.cumsum(hist)
            p5 = np.searchsorted(cumsum, 0.05)
            p95 = np.searchsorted(cumsum, 0.95)
            contrast = (p95 - p5) / 255.0
            
            # Calculate face exposure using mean and checking for clipping
            mean_val = np.mean(face_roi) / 255.0
            
            # Check for over/under exposure
            overexposed = np.mean(face_roi > 240) > 0.1  # More than 10% near-white
            underexposed = np.mean(face_roi < 20) > 0.1   # More than 10% near-black
            
            # Penalize if face is over or under exposed
            if overexposed or underexposed:
                exposure_quality = 0.3
            else:
                # Ideal face exposure is around 0.5-0.6 (middle to slightly brighter)
                exposure_quality = 1.0 - 2.0 * abs(mean_val - 0.55)
            
            # Calculate face size relative to image - larger faces get higher quality
            face_area = w * h
            image_area = grayscale.shape[0] * grayscale.shape[1]
            size_quality = min(1.0, face_area / (image_area * 0.05))  # 5% of image is a good size
            
            # Combine metrics with adjusted weights
            quality = (0.35 * min(1.0, sharpness / 200) + 
                      0.25 * contrast + 
                      0.25 * exposure_quality +
                      0.15 * size_quality) * 100
                      
            qualities.append(min(100, quality))
            
            # Libera memoria
            del laplacian, hist
            
        return qualities
    
    def _analyze_subject_position_improved(self, grayscale: np.ndarray, 
                                face_locations: List[Tuple[int, int, int, int]]) -> float:
        """
        Analyze how well subjects (faces) are positioned using improved
        composition rules including rule of thirds and eye-line positioning.
        """
        height, width = grayscale.shape
        
        if not face_locations:
            return self._analyze_generic_subject_position(grayscale)
            
        # Define rule of thirds points
        thirds_points = [
            (width // 3, height // 3),
            (width // 3, 2 * height // 3),
            (2 * width // 3, height // 3),
            (2 * width // 3, 2 * height // 3)
        ]
        
        # Calculate position scores for all faces
        position_scores = []
        
        # Get the largest face (likely the main subject)
        main_face = max(face_locations, key=lambda f: f[2] * f[3])
        main_face_area = main_face[2] * main_face[3]
        
        for (x, y, w, h) in face_locations:
            # Calculate face center
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            
            # Approximate eye line position (typically 40% from top of face)
            eye_y = int(y + h * 0.4)
            
            # Calculate distance to rule of thirds points and lines
            
            # Distance to nearest thirds point
            min_point_distance = float('inf')
            for (tx, ty) in thirds_points:
                dist = np.sqrt((face_center_x - tx)**2 + (face_center_y - ty)**2)
                min_point_distance = min(min_point_distance, dist)
            
            # Distance to horizontal thirds lines
            h_line_distances = [abs(eye_y - (height // 3)), abs(eye_y - (2 * height // 3))]
            min_h_line_distance = min(h_line_distances)
            
            # Distance to vertical thirds lines
            v_line_distances = [abs(face_center_x - (width // 3)), abs(face_center_x - (2 * width // 3))]
            min_v_line_distance = min(v_line_distances)
            
            # Normalize distances to diagonal length
            diagonal = np.sqrt(width**2 + height**2)
            norm_point_distance = min_point_distance / diagonal
            norm_h_line_distance = min_h_line_distance / height
            norm_v_line_distance = min_v_line_distance / width
            
            # Calculate final position score with different weights
            point_score = max(0, 100 - (norm_point_distance * 200))
            h_line_score = max(0, 100 - (norm_h_line_distance * 300))
            v_line_score = max(0, 100 - (norm_v_line_distance * 300))
            
            # Combine scores based on the best alignment
            position_score = max(point_score, 0.7 * h_line_score + 0.3 * v_line_score)
            
            # Weight by face size relative to main face
            face_area = w * h
            size_weight = 0.5 + 0.5 * (face_area / main_face_area)
            
            position_scores.append(position_score * size_weight)
            
        # Calculate final score - weighted by face areas
        if position_scores:
            # Take the highest positioning score as the primary indicator
            # with a small influence from other faces
            max_score = max(position_scores)
            avg_score = sum(position_scores) / len(position_scores)
            
            return 0.8 * max_score + 0.2 * avg_score
        else:
            return 50.0  # Default value if calculation fails
    
    def _analyze_generic_subject_position(self, grayscale: np.ndarray) -> float:
        """
        Analyze subject position for images without faces.
        Improved to better detect natural subjects and landscapes.
        """
        height, width = grayscale.shape
        
        # Usa la GPU se disponibile
        if self.use_gpu and self.gpu_available and hasattr(cv2, 'cuda'):
            try:
                # Carica l'immagine sulla GPU
                gpu_grayscale = cv2.cuda_GpuMat()
                gpu_grayscale.upload(grayscale)
                
                # Canny edge detection
                gpu_edges = cv2.cuda.createCannyEdgeDetector(50, 150)
                gpu_edges_result = gpu_edges.detect(gpu_grayscale)
                edges = gpu_edges_result.download()
                
                # Sobel per il gradiente
                gpu_sobelx = cv2.cuda.createSobelFilter(cv2.CV_64F, 1, 0)
                gpu_sobely = cv2.cuda.createSobelFilter(cv2.CV_64F, 0, 1)
                
                gpu_result_x = gpu_sobelx.apply(gpu_grayscale)
                gpu_result_y = gpu_sobely.apply(gpu_grayscale)
                
                sobelx = gpu_result_x.download()
                sobely = gpu_result_y.download()
                
            except Exception as e:
                self.logger.warning(f"Errore nell'elaborazione GPU per subject position: {e}. Fallback a CPU.")
                # Fallback a CPU
                edges = cv2.Canny(grayscale, 50, 150)
                sobelx = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=3)
        else:
            # Metodo CPU standard
            edges = cv2.Canny(grayscale, 50, 150)
            sobelx = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Create a saliency map combining both
        saliency = np.zeros_like(grayscale, dtype=np.float32)
        saliency += edges / 255.0  # Normalize to 0-1
        saliency += magnitude / np.max(magnitude) if np.max(magnitude) > 0 else 0
        
        # Threshold the saliency map
        thresh = np.mean(saliency) + 0.5 * np.std(saliency)
        binary = (saliency > thresh).astype(np.uint8) * 255
        
        # Find contours of salient regions
        contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Libera memoria prima dell'operazione che potrebbe usare molta memoria
        del sobelx, sobely, magnitude, edges
        gc.collect()
        
        if not contours:
            # Special case for landscapes or minimalist compositions
            # Check for horizon line
            
            # Detect horizontal lines using Hough transform
            edges = cv2.Canny(grayscale, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                                   minLineLength=width//3, maxLineGap=20)
            
            # Libera memoria
            del edges, binary, saliency
            
            if lines is not None:
                horizon_candidates = []
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Filter for mostly horizontal lines
                    angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    if angle < 10 or angle > 170:
                        horizon_candidates.append((y1 + y2) // 2)  # Average y-position
                
                if horizon_candidates:
                    # Find the most common horizon line position
                    horizon_y = int(np.median(horizon_candidates))
                    
                    # Check if it follows rule of thirds
                    thirds = [height // 3, 2 * height // 3]
                    min_dist = min(abs(horizon_y - third) for third in thirds)
                    
                    # Score based on distance to thirds line
                    horizon_score = max(0, 100 - (min_dist / (height / 6) * 100))
                    
                    # Libera memoria
                    del lines, horizon_candidates
                    
                    return horizon_score
            
            # If no clear subject or horizon detected, assume balanced composition
            return 60.0
        
        # Find the largest few contours as potential subjects
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
        
        # Define rule of thirds points
        thirds_points = [
            (width // 3, height // 3),
            (width // 3, 2 * height // 3),
            (2 * width // 3, height // 3),
            (2 * width // 3, 2 * height // 3)
        ]
        
        # Calculate position scores for the subjects
        position_scores = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Skip very small regions
                continue
                
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate center of mass
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = x + w // 2
                cy = y + h // 2
            
            # Find closest rule of thirds point
            min_distance = float('inf')
            for (tx, ty) in thirds_points:
                dist = np.sqrt((cx - tx)**2 + (cy - ty)**2)
                min_distance = min(min_distance, dist)
            
            # Normalize distance to diagonal length
            diagonal = np.sqrt(width**2 + height**2)
            norm_distance = min_distance / diagonal
            
            # Higher score for subjects closer to rule of thirds points
            position_score = max(0, 100 - (norm_distance * 200))
            
            # Weight by contour area relative to image
            area_weight = min(1.0, area / (width * height * 0.05))
            
            position_scores.append(position_score * (0.5 + 0.5 * area_weight))
        
        # Libera memoria
        del contours, binary, saliency
        
        # Calculate final score
        if position_scores:
            # Prioritize the main subject but consider secondary elements
            max_score = max(position_scores)
            avg_score = sum(position_scores) / len(position_scores)
            
            result = 0.7 * max_score + 0.3 * avg_score
            return result
        else:
            return 50.0  # Default value
    
    def _calculate_histogram(self, image_data: np.ndarray) -> np.ndarray:
        """
        Calculate image histograms for all channels.
        """
        if len(image_data.shape) == 3 and image_data.shape[2] >= 3:
            # Color image - calculate histogram for each channel
            hist_r = cv2.calcHist([image_data], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([image_data], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([image_data], [2], None, [256], [0, 256])
            return np.hstack([hist_r, hist_g, hist_b])
        else:
            # Grayscale image
            return cv2.calcHist([image_data], [0], None, [256], [0, 256])
    
    def _calculate_edge_map(self, grayscale: np.ndarray) -> np.ndarray:
        """
        Calculate Canny edge map for the image.
        """
        # Usa la GPU se disponibile
        if self.use_gpu and self.gpu_available and hasattr(cv2, 'cuda'):
            try:
                # Carica l'immagine sulla GPU
                gpu_grayscale = cv2.cuda_GpuMat()
                gpu_grayscale.upload(grayscale)
                
                # Calcola valori medi e deviazione standard per threshold adattivo
                mean_val = np.mean(grayscale)
                std_val = np.std(grayscale)
                
                # Calcola threshold per Canny
                lower = max(10, int(max(0, mean_val - std_val)))
                upper = min(250, int(min(255, mean_val + std_val)))
                
                # Crea e applica il detector Canny
                gpu_edges = cv2.cuda.createCannyEdgeDetector(lower, upper)
                gpu_edges_result = gpu_edges.detect(gpu_grayscale)
                return gpu_edges_result.download()
                
            except Exception as e:
                self.logger.warning(f"Errore nell'elaborazione GPU per edge map: {e}. Fallback a CPU.")
                # Fallback a CPU
        
        # Adaptive thresholding for better edge detection
        mean_val = np.mean(grayscale)
        std_val = np.std(grayscale)
        
        # Adjust thresholds based on image content
        lower = max(10, int(max(0, mean_val - std_val)))
        upper = min(250, int(min(255, mean_val + std_val)))
        
        return cv2.Canny(grayscale, lower, upper)
    
    def _calculate_overall_technical_improved(self, result: ImageAnalysisResult) -> float:
        """
        Calculate overall technical score with improved weighting system.
        """
        # Define content-adaptive weights
        
        # Base weights
        weights = {
            'sharpness': 0.35,
            'exposure': 0.35,
            'contrast': 0.2,
            'noise': 0.1
        }
        
        # If image is very noisy, increase the weight of noise
        if result.noise_score < 40:
            weights['noise'] = 0.3
            weights['sharpness'] = 0.3
            weights['exposure'] = 0.25
            weights['contrast'] = 0.15
            
        # If image has exposure issues, prioritize that
        elif result.exposure_score < 50:
            weights['exposure'] = 0.5
            weights['sharpness'] = 0.25
            weights['contrast'] = 0.15
            weights['noise'] = 0.1
            
        # If image is very soft/blurry, prioritize sharpness
        elif result.sharpness_score < 40:
            weights['sharpness'] = 0.5
            weights['exposure'] = 0.25
            weights['contrast'] = 0.15
            weights['noise'] = 0.1
        
        # Calculate weighted sum
        technical_score = (
            weights['sharpness'] * result.sharpness_score +
            weights['exposure'] * result.exposure_score +
            weights['contrast'] * result.contrast_score +
            weights['noise'] * result.noise_score
        )
        
        return technical_score
    
    def _calculate_overall_composition(self, result: ImageAnalysisResult) -> float:
        """
        Calculate overall composition score with improved balancing.
        """
        # Base composition factors
        if result.face_count > 0:
            # Portrait weights
            composition_factors = [
                (result.rule_of_thirds_score, 0.3),
                (result.symmetry_score, 0.2),
                (result.subject_position_score, 0.5)  # Higher weight for subject position
            ]
        else:
            # Landscape/generic weights
            composition_factors = [
                (result.rule_of_thirds_score, 0.6),
                (result.symmetry_score, 0.4)
            ]
        
        # Calculate weighted sum
        composition_score = sum(score * weight for score, weight in composition_factors)
        
        return composition_score
    
    def _calculate_overall_score_improved(self, result: ImageAnalysisResult) -> float:
        """
        Calculate overall image quality score with improved context-aware weighting.
        """
        # Adjust weights based on image content
        technical_weight = 0.6
        composition_weight = 0.4
        
        # If faces detected, adjust weights based on face count and quality
        if result.face_count > 0:
            # Calculate average face quality
            avg_face_quality = sum(result.face_qualities) / len(result.face_qualities) if result.face_qualities else 0
            
            # For portrait-style photos (with good faces)
            if avg_face_quality > 60 and result.face_count <= 3:
                technical_weight = 0.5
                composition_weight = 0.5
            # For group photos
            elif result.face_count > 3:
                technical_weight = 0.65
                composition_weight = 0.35
        
        # If technical score is very poor, it should have more influence
        if result.overall_technical_score < 40:
            technical_weight = min(0.8, technical_weight + 0.2)
            composition_weight = 1.0 - technical_weight
        
        # Calculate weighted score
        overall_score = (
            technical_weight * result.overall_technical_score +
            composition_weight * result.overall_composition_score
        )
        
        # Check for any plugin scores
        if result.plugin_scores:
            # Include plugin scores with a moderate weight
            plugin_average = sum(result.plugin_scores.values()) / len(result.plugin_scores)
            overall_score = 0.75 * overall_score + 0.25 * plugin_average
        
        return overall_score