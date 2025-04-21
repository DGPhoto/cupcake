# src/gpu_utils.py

import os
import logging
import cv2
import numpy as np
import warnings

logger = logging.getLogger("cupcake.gpu")

class GPUManager:
    """Manages GPU detection and configuration for the Cupcake application."""
    
    def __init__(self, suppress_tf_warnings=True):
        """
        Initialize the GPU manager.
        
        Args:
            suppress_tf_warnings: If True, suppress TensorFlow warnings about oneDNN
        """
        self.cuda_available = False
        self.tensorflow_gpu_available = False
        self.suppress_tf_warnings = suppress_tf_warnings
        
        # Suppress TensorFlow warnings if requested
        if suppress_tf_warnings:
            self._suppress_tensorflow_warnings()
        
        # Detect CUDA and TensorFlow GPU support
        self._detect_gpu_support()
        
    def _suppress_tensorflow_warnings(self):
        """Suppress TensorFlow warnings about oneDNN and other messages."""
        # Suppress TensorFlow logging
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Disable oneDNN custom operations warnings
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        
        # Suppress other warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        
    def _detect_gpu_support(self):
        """Detect available GPU support for OpenCV and TensorFlow."""
        # Check OpenCV CUDA support
        try:
            cv_build_info = cv2.getBuildInformation()
            if 'CUDA' in cv_build_info and 'YES' in cv_build_info[cv_build_info.find('CUDA'):cv_build_info.find('\n', cv_build_info.find('CUDA'))]:
                self.cuda_available = True
                logger.info("OpenCV CUDA support detected.")
                
                # Set OpenCV to use optimizations
                cv2.setUseOptimized(True)
            else:
                logger.info("OpenCV CUDA support NOT detected. Using CPU fallback for image processing.")
        except Exception as e:
            logger.warning(f"Error checking OpenCV CUDA support: {e}")
        
        # Check TensorFlow GPU support (only if needed by plugins)
        try:
            import tensorflow as tf
            
            # Try to list physical devices
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                self.tensorflow_gpu_available = True
                logger.info(f"TensorFlow GPU support detected. {len(gpus)} GPU(s) available.")
                
                # Configure TensorFlow to use memory growth
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        logger.debug(f"Enabled memory growth for GPU: {gpu}")
                    except Exception as e:
                        logger.warning(f"Error configuring GPU memory growth: {e}")
            else:
                logger.info("TensorFlow GPU support NOT detected. Using CPU for ML operations.")
        except ImportError:
            logger.debug("TensorFlow not installed. Skipping TensorFlow GPU check.")
        except Exception as e:
            logger.warning(f"Error checking TensorFlow GPU support: {e}")
    
    def is_gpu_available(self):
        """Return True if any GPU support is available."""
        return self.cuda_available or self.tensorflow_gpu_available
    
    def get_opencv_backend(self):
        """Return the name of the OpenCV backend in use."""
        if self.cuda_available:
            return "CUDA"
        return "CPU"
    
    def resize_image(self, image, size, interpolation=cv2.INTER_AREA):
        """
        Resize an image using GPU if available.
        
        Args:
            image: Image data as numpy array
            size: (width, height) tuple
            interpolation: Interpolation method
            
        Returns:
            Resized image
        """
        if self.cuda_available and hasattr(cv2, 'cuda'):
            try:
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(image)
                gpu_resized = cv2.cuda.resize(gpu_img, size, interpolation=interpolation)
                return gpu_resized.download()
            except Exception as e:
                logger.warning(f"GPU resize failed: {e}. Falling back to CPU.")
                return cv2.resize(image, size, interpolation=interpolation)
        else:
            return cv2.resize(image, size, interpolation=interpolation)
    
    def apply_filter(self, image, filter_type, **kwargs):
        """
        Apply a filter to an image using GPU if available.
        
        Args:
            image: Image data as numpy array
            filter_type: Type of filter (e.g., 'gaussian', 'laplacian', 'sobel_x', 'sobel_y')
            **kwargs: Filter-specific parameters
            
        Returns:
            Filtered image
        """
        if self.cuda_available and hasattr(cv2, 'cuda'):
            try:
                # Upload image to GPU
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(image)
                
                # Apply the appropriate filter
                if filter_type == 'gaussian':
                    ksize = kwargs.get('ksize', (5, 5))
                    sigma = kwargs.get('sigma', 0)
                    gpu_filter = cv2.cuda.createGaussianFilter(
                        src_type=image.dtype, 
                        dst_type=-1, 
                        ksize=ksize, 
                        sigma=sigma
                    )
                    gpu_result = gpu_filter.apply(gpu_img)
                
                elif filter_type == 'laplacian':
                    gpu_filter = cv2.cuda.createLaplacianFilter(
                        src_type=cv2.CV_64F, 
                        dst_type=-1, 
                        ksize=kwargs.get('ksize', 1)
                    )
                    gpu_result = gpu_filter.apply(gpu_img)
                
                elif filter_type == 'sobel_x':
                    gpu_filter = cv2.cuda.createSobelFilter(
                        src_type=cv2.CV_64F, 
                        dst_type=-1, 
                        dx=1, 
                        dy=0, 
                        ksize=kwargs.get('ksize', 3)
                    )
                    gpu_result = gpu_filter.apply(gpu_img)
                
                elif filter_type == 'sobel_y':
                    gpu_filter = cv2.cuda.createSobelFilter(
                        src_type=cv2.CV_64F, 
                        dst_type=-1, 
                        dx=0, 
                        dy=1, 
                        ksize=kwargs.get('ksize', 3)
                    )
                    gpu_result = gpu_filter.apply(gpu_img)
                
                elif filter_type == 'canny':
                    low_threshold = kwargs.get('low_threshold', 50)
                    high_threshold = kwargs.get('high_threshold', 150)
                    gpu_filter = cv2.cuda.createCannyEdgeDetector(low_threshold, high_threshold)
                    gpu_result = gpu_filter.detect(gpu_img)
                
                else:
                    raise ValueError(f"Unsupported filter type: {filter_type}")
                
                # Download result from GPU
                return gpu_result.download()
                
            except Exception as e:
                logger.warning(f"GPU filter '{filter_type}' failed: {e}. Falling back to CPU.")
        
        # CPU fallback
        if filter_type == 'gaussian':
            ksize = kwargs.get('ksize', (5, 5))
            sigma = kwargs.get('sigma', 0)
            return cv2.GaussianBlur(image, ksize, sigma)
        
        elif filter_type == 'laplacian':
            return cv2.Laplacian(image, cv2.CV_64F, ksize=kwargs.get('ksize', 1))
        
        elif filter_type == 'sobel_x':
            return cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kwargs.get('ksize', 3))
        
        elif filter_type == 'sobel_y':
            return cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kwargs.get('ksize', 3))
        
        elif filter_type == 'canny':
            low_threshold = kwargs.get('low_threshold', 50)
            high_threshold = kwargs.get('high_threshold', 150)
            return cv2.Canny(image, low_threshold, high_threshold)
        
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")