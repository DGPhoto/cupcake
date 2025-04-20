# src/image_loader.py

import os
from PIL import Image
import exifread
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from .image_formats import ImageFormats

# Import the error suppressor
try:
    from .error_suppressor import ErrorSuppressor
except ImportError:
    # Create a fallback implementation if module is not found
    class ErrorSuppressor:
        @staticmethod
        def suppress_stderr():
            import contextlib
            import io
            @contextlib.contextmanager
            def _suppress_stderr():
                stderr = io.StringIO()
                import sys
                old_stderr = sys.stderr
                sys.stderr = stderr
                try:
                    yield
                finally:
                    sys.stderr = old_stderr
            return _suppress_stderr()

class ImageLoader:
    """Handles loading images from various sources and extracting metadata."""
    
    def __init__(self):
        self.image_cache = {}  # Simple cache to avoid reloading
    
    def load_from_path(self, path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load an image from a file path.
        
        Args:
            path: Path to the image file
            
        Returns:
            Tuple of (image_data, metadata)
        """
        if path in self.image_cache:
            return self.image_cache[path]
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")
        
        extension = path.split('.')[-1].lower()
        
        if not ImageFormats.is_supported_format(extension):
            raise ValueError(f"Unsupported file format: {extension}")
        
        # Load image data based on format
        if ImageFormats.is_raw_format(extension):
            # RAW file handling
            try:
                import rawpy
                # Use error suppressor to hide libraw error messages
                with ErrorSuppressor.suppress_stderr():
                    with rawpy.imread(path) as raw:
                        image_data = raw.postprocess()
            except ImportError:
                raise ImportError("rawpy is required for RAW file support. Please install it with: pip install rawpy")
            except Exception as e:
                raise IOError(f"Error processing RAW file: {str(e)}")
        else:
            # Standard formats
            try:
                with Image.open(path) as img:
                    image_data = np.array(img)
            except Exception as e:
                raise IOError(f"Error opening image: {str(e)}")
        
        # Extract metadata
        metadata = self.extract_metadata(path)
        
        # Store in cache
        self.image_cache[path] = (image_data, metadata)
        
        return image_data, metadata
    
    def load_from_directory(self, directory: str) -> List[Tuple[str, np.ndarray, Dict[str, Any]]]:
        """
        Load all supported images from a directory.
        
        Args:
            directory: Path to directory containing images
            
        Returns:
            List of tuples: (file_path, image_data, metadata)
        """
        results = []
        
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"Directory not found: {directory}")
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                extension = filename.split('.')[-1].lower()
                if ImageFormats.is_supported_format(extension):
                    try:
                        with ErrorSuppressor.suppress_stderr():
                            image_data, metadata = self.load_from_path(file_path)
                        results.append((file_path, image_data, metadata))
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
        
        return results
    
    def extract_metadata(self, path: str) -> Dict[str, Any]:
        """
        Extract EXIF and other metadata from an image file.
        
        Args:
            path: Path to the image file
            
        Returns:
            Dictionary of metadata
        """
        extension = path.split('.')[-1].lower()
        
        metadata = {
            'filename': os.path.basename(path),
            'filepath': path,
            'filesize': os.path.getsize(path),
            'extension': extension,
            'mime_type': ImageFormats.get_mime_type(extension),
            'is_raw': ImageFormats.is_raw_format(extension)
        }
        
        # Add manufacturer for RAW files
        if metadata['is_raw']:
            metadata['manufacturer'] = ImageFormats.get_manufacturer_for_raw_format(extension)
        
        # Extract EXIF data
        try:
            with open(path, 'rb') as f:
                try:
                    with ErrorSuppressor.suppress_stderr():
                        exif_tags = exifread.process_file(f)
                        
                        # Process common EXIF tags
                        if 'EXIF DateTimeOriginal' in exif_tags:
                            metadata['datetime'] = str(exif_tags['EXIF DateTimeOriginal'])
                        
                        if 'EXIF FocalLength' in exif_tags:
                            metadata['focal_length'] = self._extract_rational(exif_tags['EXIF FocalLength'])
                        
                        if 'EXIF ExposureTime' in exif_tags:
                            metadata['exposure_time'] = self._extract_rational(exif_tags['EXIF ExposureTime'])
                        
                        if 'EXIF FNumber' in exif_tags:
                            metadata['f_number'] = self._extract_rational(exif_tags['EXIF FNumber'])
                        
                        if 'EXIF ISOSpeedRatings' in exif_tags:
                            metadata['iso'] = str(exif_tags['EXIF ISOSpeedRatings'])
                        
                        if 'EXIF LensModel' in exif_tags:
                            metadata['lens_model'] = str(exif_tags['EXIF LensModel'])
                        
                        if 'Image Model' in exif_tags:
                            metadata['camera_model'] = str(exif_tags['Image Model'])
                        
                        if 'Image Make' in exif_tags:
                            metadata['camera_make'] = str(exif_tags['Image Make'])
                except Exception as e:
                    metadata['exif_error'] = str(e)
        
        except Exception as e:
            metadata['exif_error'] = str(e)
        
        # Set today's date if no datetime in metadata (fallback for organization)
        if 'datetime' not in metadata:
            import datetime
            metadata['datetime'] = datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")
        
        # Try to get more metadata from RAW files if available
        if metadata['is_raw'] and not metadata.get('camera_make') and not metadata.get('camera_model'):
            self._extract_raw_metadata(path, metadata)
        
        return metadata
    
    def _extract_rational(self, tag) -> float:
        """Helper method to convert an EXIF rational value to a float."""
        try:
            return float(tag.values[0].num) / float(tag.values[0].den)
        except (AttributeError, ZeroDivisionError):
            return 0.0
    
    def _extract_raw_metadata(self, path: str, metadata: Dict[str, Any]) -> None:
        """
        Try to extract additional metadata from RAW files.
        
        Args:
            path: Path to the RAW file
            metadata: Metadata dictionary to update
        """
        try:
            import rawpy
            with ErrorSuppressor.suppress_stderr():
                with rawpy.imread(path) as raw:
                    if hasattr(raw, 'raw_type') and raw.raw_type:
                        metadata['raw_type'] = str(raw.raw_type)
                    
                    if hasattr(raw, 'camera_maker') and raw.camera_maker:
                        metadata['camera_make'] = raw.camera_maker
                    
                    if hasattr(raw, 'camera_model') and raw.camera_model:
                        metadata['camera_model'] = raw.camera_model
        except (ImportError, Exception):
            # Just skip if rawpy is not available or errors occur
            pass
    
    def clear_cache(self):
        """Clear the image cache to free memory."""
        self.image_cache.clear()