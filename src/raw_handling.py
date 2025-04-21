# src/raw_handling.py

import os
import io
import struct
import numpy as np
from PIL import Image, ExifTags
from typing import Dict, Any, Optional, Tuple, List, Union
import logging
import traceback

logger = logging.getLogger("cupcake.raw_handling")

class RawHandlingError(Exception):
    """Base exception for raw handling errors."""
    pass

class RawFormatNotSupported(RawHandlingError):
    """Exception raised when a RAW format is not supported."""
    pass

class RawProcessingError(RawHandlingError):
    """Exception raised when there's an error processing a RAW file."""
    pass

class BasicRawHandler:
    """
    Basic RAW file handler that extracts embedded previews from RAW files.
    This serves as a fallback when no specialized RAW plugins are available.
    """
    
    # RAW formats and their manufacturers
    RAW_FORMATS = {
        'cr2': 'Canon',
        'cr3': 'Canon',
        'crw': 'Canon',
        'nef': 'Nikon',
        'nrw': 'Nikon',
        'arw': 'Sony',
        'srf': 'Sony',
        'sr2': 'Sony',
        'raf': 'Fujifilm',
        'orf': 'Olympus',
        'rw2': 'Panasonic',
        'pef': 'Pentax',
        'dng': 'Adobe',
        'raw': 'Generic',
        'rwl': 'Leica',
        '3fr': 'Hasselblad',
        'fff': 'Hasselblad',
        'iiq': 'Phase One',
        'x3f': 'Sigma'
    }
    
    # Common tags to extract from EXIF data
    EXIF_TAGS = {
        'Make': 'camera_make',
        'Model': 'camera_model',
        'LensModel': 'lens_model',
        'DateTimeOriginal': 'datetime',
        'ExposureTime': 'exposure_time',
        'FNumber': 'f_number',
        'ISOSpeedRatings': 'iso',
        'FocalLength': 'focal_length'
    }
    
    def __init__(self):
        self.registered_plugins = []
        self.logger = logger
    
    def register_plugin(self, plugin):
        """
        Register a RAW handling plugin that will be given priority.
        
        Args:
            plugin: A plugin object with a process_raw_file method
        """
        self.registered_plugins.append(plugin)
        self.logger.info(f"Registered RAW handling plugin: {plugin.__class__.__name__}")
    
    def is_raw_format(self, extension: str) -> bool:
        """
        Check if a file extension corresponds to a RAW format.
        
        Args:
            extension: File extension (with or without leading dot)
            
        Returns:
            True if the extension is a known RAW format
        """
        ext = extension.lower().lstrip('.')
        return ext in self.RAW_FORMATS
    
    def get_manufacturer(self, extension: str) -> str:
        """
        Get the camera manufacturer for a RAW format.
        
        Args:
            extension: File extension (with or without leading dot)
            
        Returns:
            Manufacturer name or "Unknown"
        """
        ext = extension.lower().lstrip('.')
        return self.RAW_FORMATS.get(ext, "Unknown")
    
    def process_raw_file(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a RAW file and return image data and metadata.
        Tries registered plugins first, falls back to basic handling.
        
        Args:
            file_path: Path to the RAW file
            
        Returns:
            Tuple of (image_data, metadata)
            
        Raises:
            RawFormatNotSupported: If the RAW format is not supported
            RawProcessingError: If there's an error processing the RAW file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Normalize path to handle Windows path separators
        file_path = os.path.normpath(file_path)
        
        extension = os.path.splitext(file_path)[1].lower().lstrip('.')
        
        if not self.is_raw_format(extension):
            raise RawFormatNotSupported(f"Not a supported RAW format: {extension}")
        
        # Try registered plugins first
        for plugin in self.registered_plugins:
            try:
                self.logger.debug(f"Trying plugin {plugin.__class__.__name__} for {file_path}")
                if hasattr(plugin, "can_handle") and plugin.can_handle(extension):
                    return plugin.process_raw_file(file_path)
            except Exception as e:
                self.logger.warning(f"Plugin {plugin.__class__.__name__} failed to process {file_path}: {e}")
                self.logger.debug(traceback.format_exc())
        
        # Fall back to basic handling if no plugin succeeded
        try:
            return self._extract_preview_and_metadata(file_path)
        except Exception as e:
            self.logger.error(f"Failed to process RAW file {file_path}: {e}")
            self.logger.debug(traceback.format_exc())
            raise RawProcessingError(f"Failed to process {file_path}: {str(e)}")
    
    def _extract_preview_and_metadata(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract embedded preview and metadata from RAW file with improved RAF handling.
        
        Args:
            file_path: Path to the RAW file
            
        Returns:
            Tuple of (image_data, metadata)
        """
        extension = os.path.splitext(file_path)[1].lower().lstrip('.')
        metadata = self._extract_basic_metadata(file_path)
        
        # Extract preview based on format
        try:
            if extension in ['cr2', 'nef', 'arw', 'dng']:
                image_data = self._extract_jpeg_preview(file_path)
            elif extension == 'raf':  # Fujifilm RAF files need special handling
                image_data = self._extract_raf_preview_improved(file_path)
            else:
                # For unsupported formats, create a placeholder image
                image_data = self._create_placeholder_image(file_path, metadata)
        except Exception as e:
            self.logger.warning(f"Failed to extract preview from {file_path}: {e}")
            self.logger.debug(traceback.format_exc())
            image_data = self._create_placeholder_image(file_path, metadata)
        
        return image_data, metadata

    def _extract_raf_preview_improved(self, file_path: str) -> np.ndarray:
        """
        Extract preview from Fujifilm RAF files with enhanced robustness.
        RAF files have a more complex structure and require special handling.
        
        Args:
            file_path: Path to the RAF file
            
        Returns:
            NumPy array containing the preview image
        """
        try:
            # Try to open the RAF file directly with PIL first
            # Some RAF files can be opened directly by PIL
            try:
                with Image.open(file_path) as img:
                    # Force loading the image to ensure it's valid
                    img.load()
                    self.logger.debug(f"Successfully opened RAF file directly with PIL: {file_path}")
                    return np.array(img)
            except Exception as e:
                self.logger.debug(f"PIL couldn't directly open RAF: {e}, falling back to preview extraction")
            
            # Read the file in binary mode
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # RAF files typically have "FUJIFILMCCD-RAW" marker
            if b"FUJIFILMCCD-RAW" not in data[:100]:
                self.logger.warning(f"File doesn't appear to be a valid RAF file: {file_path}")
            
            # Multiple approaches to find JPEG preview
            approaches = [
                # Approach 1: Look for JPEG after "JPEG" marker
                lambda d: (d.find(b"JPEG"), d.find(b'\xff\xd8', d.find(b"JPEG")) if d.find(b"JPEG") != -1 else -1),
                # Approach 2: Look for JPEG after "FUJIFILMCCD-RAW" marker
                lambda d: (d.find(b"FUJIFILMCCD-RAW"), d.find(b'\xff\xd8', d.find(b"FUJIFILMCCD-RAW")) if d.find(b"FUJIFILMCCD-RAW") != -1 else -1),
                # Approach 3: Look for first JPEG SOI marker
                lambda d: (-1, d.find(b'\xff\xd8')),
                # Approach 4: Look for JPEG SOI marker after position 1000 (to skip metadata)
                lambda d: (-1, d.find(b'\xff\xd8', 1000)),
                # Approach 5: Look for JPEG after position 2000
                lambda d: (-1, d.find(b'\xff\xd8', 2000)),
                # Approach 6: Try to find after a known RAF header marker
                lambda d: (-1, d.find(b'\xff\xd8', d.find(b"RAF")) if d.find(b"RAF") != -1 else -1)
            ]
            
            jpeg_start = -1
            for approach in approaches:
                marker_pos, start_pos = approach(data)
                if start_pos != -1:
                    jpeg_start = start_pos
                    self.logger.debug(f"Found JPEG start at position {jpeg_start} after marker at {marker_pos}")
                    break
            
            if jpeg_start == -1:
                self.logger.warning(f"No JPEG preview found in RAF file: {file_path}")
                return self._create_placeholder_image(file_path, {"filename": os.path.basename(file_path)})
            
            # Find the end of the JPEG data
            jpeg_end = data.find(b'\xff\xd9', jpeg_start)
            
            # If we can't find a proper EOI marker, try to find the next SOI marker
            # as an indicator of where this JPEG might end
            if jpeg_end == -1:
                next_soi = data.find(b'\xff\xd8', jpeg_start + 2)
                if next_soi != -1:
                    # Use the position before the next SOI as the end of current JPEG
                    jpeg_end = next_soi - 2
                else:
                    # If all else fails, use a reasonable chunk of data after the start
                    jpeg_end = min(jpeg_start + 1000000, len(data) - 2)  # 1MB limit or end of file
            
            # Extract the JPEG data including the EOI marker if available
            if jpeg_end > jpeg_start:
                jpeg_data = data[jpeg_start:jpeg_end+2]
            else:
                # Fallback: just take a reasonable chunk after start
                jpeg_data = data[jpeg_start:min(jpeg_start+1000000, len(data))]  # 1MB limit
            
            # Check if we got valid JPEG data
            if not jpeg_data.startswith(b'\xff\xd8'):
                self.logger.warning(f"Invalid JPEG data extracted from RAF file: {file_path}")
                return self._create_placeholder_image(file_path, {"filename": os.path.basename(file_path)})
            
            # Convert to image, with error handling for truncated files
            try:
                image = Image.open(io.BytesIO(jpeg_data))
                # Force load to catch truncated file issues early
                image.load()
                return np.array(image)
            except Exception as e:
                self.logger.warning(f"Failed to decode JPEG preview: {str(e)}")
                # Try again with just part of the data to handle truncated files
                try:
                    # Take the first 90% of the data to avoid potential corruption
                    safe_size = int(len(jpeg_data) * 0.9)
                    safe_jpeg = jpeg_data[:safe_size]
                    image = Image.open(io.BytesIO(safe_jpeg))
                    image.load()
                    return np.array(image)
                except Exception as e2:
                    self.logger.warning(f"Failed to decode JPEG preview with reduced size: {str(e2)}")
                    # One last attempt: try to find another JPEG in the file
                    try:
                        # Look for another JPEG SOI marker after the failed one
                        next_jpeg_start = data.find(b'\xff\xd8', jpeg_start + 100)
                        if next_jpeg_start != -1:
                            next_jpeg_end = data.find(b'\xff\xd9', next_jpeg_start)
                            if next_jpeg_end != -1:
                                next_jpeg_data = data[next_jpeg_start:next_jpeg_end+2]
                                image = Image.open(io.BytesIO(next_jpeg_data))
                                image.load()
                                return np.array(image)
                    except Exception:
                        # If all attempts failed, create placeholder
                        pass
                    
                    return self._create_placeholder_image(file_path, {"filename": os.path.basename(file_path)})
        
        except Exception as e:
            self.logger.error(f"Error extracting RAF preview: {e}")
            self.logger.debug(traceback.format_exc())
            # Create a placeholder image as fallback
            return self._create_placeholder_image(file_path, {"filename": os.path.basename(file_path)})
    
    def _extract_jpeg_preview(self, file_path: str) -> np.ndarray:
        """
        Extract embedded JPEG preview from a RAW file.
        
        Args:
            file_path: Path to the RAW file
            
        Returns:
            NumPy array containing the preview image
        """
        # Try to open with PIL first
        try:
            with Image.open(file_path) as img:
                img.load()  # Force load to catch errors early
                return np.array(img)
        except Exception as e:
            self.logger.debug(f"PIL couldn't open file directly: {e}, trying binary extraction")
        
        # Read the file in binary mode
        with open(file_path, 'rb') as f:
            data = f.read()
        
        # Look for JPEG SOI marker (0xFFD8) followed by JPEG EOI marker (0xFFD9)
        jpeg_start = data.find(b'\xff\xd8')
        if jpeg_start == -1:
            raise RawProcessingError("No JPEG preview found in file")
        
        # Find the end of the JPEG data
        jpeg_end = data.find(b'\xff\xd9', jpeg_start)
        if jpeg_end == -1:
            # Try a more lenient approach - take a reasonable chunk
            chunk_size = min(10*1024*1024, len(data) - jpeg_start)  # 10MB or end of file
            jpeg_data = data[jpeg_start:jpeg_start + chunk_size]
        else:
            # Extract the JPEG data including the EOI marker
            jpeg_data = data[jpeg_start:jpeg_end+2]
        
        # Convert to image
        try:
            image = Image.open(io.BytesIO(jpeg_data))
            image.load()  # Force load to catch errors early
            return np.array(image)
        except Exception as e:
            try:
                # Try with reduced size for possibly truncated files
                safe_size = int(len(jpeg_data) * 0.9)
                if safe_size > 1000:  # Ensure we have enough data to work with
                    safe_jpeg = jpeg_data[:safe_size]
                    image = Image.open(io.BytesIO(safe_jpeg))
                    image.load()
                    return np.array(image)
            except Exception:
                pass
                
            raise RawProcessingError(f"Failed to decode JPEG preview: {str(e)}")
    
    def _create_placeholder_image(self, file_path: str, metadata: Dict[str, Any]) -> np.ndarray:
        """
        Create a placeholder image for when preview extraction fails.
        
        Args:
            file_path: Path to the RAW file
            metadata: Metadata dictionary
            
        Returns:
            NumPy array containing a placeholder image
        """
        # Create a simple gradient image with file info
        width, height = 800, 600
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create a gradient background
        for y in range(height):
            for x in range(width):
                image[y, x, 0] = int(255 * (x / width))
                image[y, x, 1] = int(255 * (y / height))
                image[y, x, 2] = 100
        
        # Convert to PIL Image to add text
        pil_image = Image.fromarray(image)
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(pil_image)
        
        # Try to use a basic font
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Add file info
        filename = os.path.basename(file_path)
        camera = f"{metadata.get('camera_make', 'Unknown')} {metadata.get('camera_model', '')}"
        
        draw.text((20, 20), "Preview Not Available", fill=(255, 255, 255), font=font)
        draw.text((20, 60), f"File: {filename}", fill=(255, 255, 255), font=font)
        draw.text((20, 100), f"Camera: {camera}", fill=(255, 255, 255), font=font)
        
        if 'datetime' in metadata:
            draw.text((20, 140), f"Date: {metadata['datetime']}", fill=(255, 255, 255), font=font)
        
        return np.array(pil_image)
    
    def _extract_basic_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract basic metadata from a RAW file.
        
        Args:
            file_path: Path to the RAW file
            
        Returns:
            Dictionary with metadata
        """
        metadata = {
            'filename': os.path.basename(file_path),
            'filepath': file_path,
            'filesize': os.path.getsize(file_path),
            'extension': os.path.splitext(file_path)[1].lower().lstrip('.'),
            'is_raw': True
        }
        
        # Add manufacturer
        metadata['manufacturer'] = self.get_manufacturer(metadata['extension'])
        
        # Try to extract EXIF data using PIL
        try:
            with Image.open(file_path) as img:
                exif_data = img._getexif()
                if exif_data:
                    for tag_id, tag_value in exif_data.items():
                        tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                        if tag_name in self.EXIF_TAGS:
                            metadata[self.EXIF_TAGS[tag_name]] = tag_value
        except Exception as e:
            self.logger.debug(f"Failed to extract EXIF from {file_path}: {e}")
            # If EXIF extraction fails, try a more basic approach
            self._extract_minimal_metadata(file_path, metadata)
        
        return metadata
    
    def _extract_minimal_metadata(self, file_path: str, metadata: Dict[str, Any]) -> None:
        """
        Extract minimal metadata from file headers when EXIF extraction fails.
        Modifies the metadata dictionary in place.
        
        Args:
            file_path: Path to the RAW file
            metadata: Metadata dictionary to update
        """
        try:
            # Read first 4KB of the file for headers
            with open(file_path, 'rb') as f:
                header = f.read(4096)
            
            # Look for common maker information in the header
            if b'Canon' in header:
                metadata['camera_make'] = 'Canon'
            elif b'NIKON' in header:
                metadata['camera_make'] = 'Nikon'
            elif b'SONY' in header:
                metadata['camera_make'] = 'Sony'
            elif b'FUJIFILM' in header:
                metadata['camera_make'] = 'Fujifilm'
            
            # Set today's date if no datetime in metadata (fallback for organization)
            if 'datetime' not in metadata:
                import datetime
                metadata['datetime'] = datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")
        
        except Exception as e:
            self.logger.debug(f"Failed to extract minimal metadata from {file_path}: {e}")


class IRawPlugin:
    """Interface for RAW processing plugins."""
    
    @staticmethod
    def can_handle(extension: str) -> bool:
        """
        Check if the plugin can handle a specific RAW format.
        
        Args:
            extension: File extension (without leading dot)
            
        Returns:
            True if the plugin can handle this format
        """
        raise NotImplementedError("Subclasses must implement can_handle")
    
    @staticmethod
    def process_raw_file(file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a RAW file and return image data and metadata.
        
        Args:
            file_path: Path to the RAW file
            
        Returns:
            Tuple of (image_data, metadata)
        """
        raise NotImplementedError("Subclasses must implement process_raw_file")


# Create a singleton instance
_raw_handler = None

def get_raw_handler() -> BasicRawHandler:
    """
    Get the global RAW handler instance.
    
    Returns:
        The global BasicRawHandler instance
    """
    global _raw_handler
    if _raw_handler is None:
        _raw_handler = BasicRawHandler()
    return _raw_handler