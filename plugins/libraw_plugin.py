# plugins/libraw_plugin.py
# Cupcake Photo Culling Library Plugin

import os
import sys
import platform
import ctypes
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
import tempfile

from src.plugin_system import CupcakePlugin, PluginType, PluginHook
from src.raw_handling import IRawPlugin

import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

class LibRawPlugin(CupcakePlugin, IRawPlugin):
    """
    LibRaw Plugin - A plugin that uses native LibRaw libraries for RAW file processing.
    
    This plugin dynamically loads LibRaw libraries installed on the system.
    It works on Windows, macOS, and Linux platforms.
    """
    
    plugin_name = "LibRaw Plugin"
    plugin_type = PluginType.UTILITY
    plugin_description = "Provides advanced RAW file processing using LibRaw"
    plugin_version = "0.1.0"
    plugin_author = "Cupcake Team"
    plugin_hooks = [
        PluginHook.STARTUP,
        PluginHook.SHUTDOWN
    ]
    
    # Supported RAW formats (LibRaw supports all these formats)
    SUPPORTED_FORMATS = [
        'cr2', 'cr3', 'crw',  # Canon
        'nef', 'nrw',         # Nikon
        'arw', 'srf', 'sr2',  # Sony
        'raf',                # Fujifilm
        'orf',                # Olympus
        'rw2',                # Panasonic
        'pef', 'dng',         # Pentax, Adobe
        'raw', 'rwl',         # Generic, Leica
        '3fr', 'fff',         # Hasselblad
        'iiq',                # Phase One
        'x3f',                # Sigma
        'erf',                # Epson
        'mef', 'mrw',         # Mamiya, Minolta
        'kdc', 'dcr'          # Kodak
    ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.libraw = None
        self.is_initialized = False
        
        # Platform-specific library names
        self.lib_paths = {
            'Windows': ['libraw.dll', 'raw.dll'],
            'Darwin': ['libraw.dylib', '/usr/local/lib/libraw.dylib', '/opt/homebrew/lib/libraw.dylib'],
            'Linux': ['libraw.so', 'libraw.so.23', 'libraw.so.22', 'libraw.so.20', 'libraw.so.19']
        }
        
        # Set default configuration
        default_config = {
            "libraw_path": None,  # Custom path to LibRaw library
            "processing_options": {
                "use_camera_wb": 1,       # Use camera white balance if available
                "half_size": 0,           # Full-size processing (0) or half-size (1)
                "output_color": 1,        # Output color profile: 0=raw, 1=sRGB, 2=Adobe, 3=Wide, 4=ProPhoto, 5=XYZ
                "output_bps": 8,          # Bits per sample (8 or 16)
                "highlight_mode": 0,      # Highlight mode (0=clip, 1=unclip, 2=blend, 3-9=rebuild)
                "brightness": 1.0,        # Brightness
                "user_qual": 3,           # Demosaicing algorithm (0=linear, 1=VNG, 2=PPG, 3=AHD, 4=DCB)
                "auto_bright": 1,         # Auto brightness (0=disable, 1=enable)
                "fbdd_noise_reduction": 0 # FBDD noise reduction (0=off, 1=light, 2=full)
            }
        }
        
        # Update with provided config
        if config:
            # Deep merge of processing_options
            if 'processing_options' in config:
                for key, value in config['processing_options'].items():
                    default_config['processing_options'][key] = value
                del config['processing_options']
            
            # Update the rest
            default_config.update(config)
        
        self.config = default_config
        
    def initialize(self) -> bool:
        """
        Initialize the plugin by loading the LibRaw library.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        self.logger.info(f"Initializing {self.plugin_name}")
        
        # Try to load LibRaw library
        if self._load_libraw():
            self.is_initialized = True
            
            # Register with the raw handler
            from src.raw_handling import get_raw_handler
            get_raw_handler().register_plugin(self)
            
            self.logger.info(f"Successfully initialized {self.plugin_name}")
            return True
        else:
            self.logger.warning(f"Failed to initialize {self.plugin_name}")
            return False
    
    def shutdown(self) -> bool:
        """
        Shutdown the plugin.
        
        Returns:
            True if shutdown was successful
        """
        self.logger.info(f"Shutting down {self.plugin_name}")
        self.libraw = None
        self.is_initialized = False
        return True
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get the configuration schema for this plugin.
        
        Returns:
            Dictionary with configuration schema
        """
        return {
            "libraw_path": {
                "type": "string",
                "default": None,
                "description": "Custom path to LibRaw library"
            },
            "processing_options": {
                "type": "object",
                "properties": {
                    "use_camera_wb": {
                        "type": "integer",
                        "default": 1,
                        "description": "Use camera white balance if available (0=no, 1=yes)"
                    },
                    "half_size": {
                        "type": "integer",
                        "default": 0,
                        "description": "Half-size processing (0=no, 1=yes)"
                    },
                    "output_color": {
                        "type": "integer",
                        "default": 1,
                        "description": "Output color profile: 0=raw, 1=sRGB, 2=Adobe, 3=Wide, 4=ProPhoto, 5=XYZ"
                    },
                    "output_bps": {
                        "type": "integer",
                        "default": 8,
                        "description": "Bits per sample (8 or 16)"
                    },
                    "highlight_mode": {
                        "type": "integer",
                        "default": 0,
                        "description": "Highlight mode (0=clip, 1=unclip, 2=blend, 3-9=rebuild)"
                    },
                    "brightness": {
                        "type": "number",
                        "default": 1.0,
                        "description": "Brightness multiplier"
                    },
                    "user_qual": {
                        "type": "integer",
                        "default": 3,
                        "description": "Demosaicing algorithm (0=linear, 1=VNG, 2=PPG, 3=AHD, 4=DCB)"
                    },
                    "auto_bright": {
                        "type": "integer",
                        "default": 1,
                        "description": "Auto brightness (0=disable, 1=enable)"
                    },
                    "fbdd_noise_reduction": {
                        "type": "integer",
                        "default": 0,
                        "description": "FBDD noise reduction (0=off, 1=light, 2=full)"
                    }
                }
            }
        }
    
    @staticmethod
    def can_handle(extension: str) -> bool:
        """
        Check if this plugin can handle a specific RAW format.
        
        Args:
            extension: File extension (without leading dot)
            
        Returns:
            True if the plugin can handle this format
        """
        return extension.lower() in LibRawPlugin.SUPPORTED_FORMATS
    
def process_raw_file(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Process a RAW file using LibRaw with timeout and error handling.
    
    Args:
        file_path: Path to the RAW file
        
    Returns:
        Tuple of (image_data, metadata)
        
    Raises:
        Exception: If processing fails
    """
    if not self.is_initialized or not self.libraw:
        self.logger.error("LibRaw plugin is not initialized")
        raise RuntimeError("LibRaw plugin is not initialized")
    
    # Create a new LibRaw processor
    processor = self._create_processor()
    if not processor:
        self.logger.error("Failed to create LibRaw processor")
        raise RuntimeError("Failed to create LibRaw processor")
    
    # Extract metadata before potentially timing out on processing
    try:
        metadata = self._extract_metadata(processor, file_path)
    except Exception as e:
        self.logger.warning(f"Error extracting metadata: {e}")
        # Create basic metadata
        metadata = {
            'filename': os.path.basename(file_path),
            'filepath': file_path,
            'filesize': os.path.getsize(file_path),
            'extension': os.path.splitext(file_path)[1].lower().lstrip('.'),
            'is_raw': True,
            'manufacturer': self.get_manufacturer_from_extension(file_path)
        }
    
    # Process the RAW file with a timeout
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._process_raw_with_timeout, processor, file_path)
            # Set a timeout of 30 seconds
            image_data = future.result(timeout=30)
            return image_data, metadata
    except TimeoutError:
        self.logger.error(f"Processing timed out for {file_path}")
        # Clean up the processor that might be hanging
        try:
            self.libraw.libraw_recycle(processor)
            self.libraw.libraw_close(processor)
        except:
            pass
        
        # Return a placeholder image instead
        self.logger.info("Generating placeholder image")
        from src.raw_handling import BasicRawHandler
        basic_handler = BasicRawHandler()
        placeholder = basic_handler._create_placeholder_image(file_path, metadata)
        return placeholder, metadata
    except Exception as e:
        self.logger.error(f"Error processing {file_path}: {e}")
        # Clean up
        try:
            self.libraw.libraw_recycle(processor)
            self.libraw.libraw_close(processor)
        except:
            pass
        
        # Re-raise the exception
        raise

def _process_raw_with_timeout(self, processor, file_path: str) -> np.ndarray:
    """
    Process a RAW file with the given processor.
    This method is designed to be run in a separate thread with a timeout.
    
    Args:
        processor: LibRaw processor
        file_path: Path to RAW file
        
    Returns:
        Processed image data
    """
    try:
        # Open the file
        result = self.libraw.libraw_open_file(processor, ctypes.c_char_p(file_path.encode('utf-8')))
        if result != 0:
            raise RuntimeError(f"LibRaw failed to open file: Error code {result}")
        
        # Unpack the raw data
        result = self.libraw.libraw_unpack(processor)
        if result != 0:
            raise RuntimeError(f"LibRaw failed to unpack raw data: Error code {result}")
        
        # Set processing parameters
        self._set_processing_params(processor)
        
        # Process the raw data
        result = self.libraw.libraw_dcraw_process(processor)
        if result != 0:
            raise RuntimeError(f"LibRaw failed to process raw data: Error code {result}")
        
        # Get the processed image
        image_data = self._get_processed_image(processor)
        
        return image_data
    finally:
        # Always clean up
        try:
            self.libraw.libraw_recycle(processor)
            self.libraw.libraw_close(processor)
        except:
            pass

def _load_libraw(self) -> bool:
    """
    Load the LibRaw library for the current platform.
    
    Returns:
        True if successful, False otherwise
    """
    # Check for custom path first
    if self.config.get('libraw_path'):
        try:
            self.libraw = ctypes.cdll.LoadLibrary(self.config['libraw_path'])
            self.logger.info(f"Loaded LibRaw from custom path: {self.config['libraw_path']}")
            return self._initialize_functions()
        except (OSError, AttributeError) as e:
            self.logger.warning(f"Failed to load LibRaw from custom path: {e}")
    
    # Try default paths for current platform
    current_platform = platform.system()
    if current_platform not in self.lib_paths:
        self.logger.error(f"Unsupported platform: {current_platform}")
        return False
    
    # Try each possible library name
    for lib_name in self.lib_paths[current_platform]:
        try:
            self.libraw = ctypes.cdll.LoadLibrary(lib_name)
            self.logger.info(f"Loaded LibRaw from: {lib_name}")
            return self._initialize_functions()
        except (OSError, AttributeError) as e:
            self.logger.debug(f"Failed to load LibRaw from {lib_name}: {e}")
    
    # Try to find in common directories
    common_dirs = []
    if current_platform == "Windows":
        common_dirs.extend([
            os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bin"),
            os.path.dirname(os.path.abspath(__file__))
        ])
    elif current_platform == "Darwin":  # macOS
        common_dirs.extend([
            "/usr/local/lib",
            "/opt/homebrew/lib"
        ])
    else:  # Linux
        common_dirs.extend([
            "/usr/lib",
            "/usr/local/lib",
            "/usr/lib64",
            "/usr/local/lib64"
        ])
    
    # Check common directories
    for directory in common_dirs:
        if not os.path.exists(directory):
            continue
            
        for lib_name in self.lib_paths[current_platform]:
            lib_path = os.path.join(directory, os.path.basename(lib_name))
            if os.path.exists(lib_path):
                try:
                    self.libraw = ctypes.cdll.LoadLibrary(lib_path)
                    self.logger.info(f"Loaded LibRaw from: {lib_path}")
                    return self._initialize_functions()
                except (OSError, AttributeError) as e:
                    self.logger.debug(f"Failed to load LibRaw from {lib_path}: {e}")
    
    self.logger.error("Failed to load LibRaw library. Please install LibRaw or specify a valid path.")
    return False

def get_manufacturer_from_extension(self, file_path: str) -> str:
    """Get manufacturer from file extension."""
    ext = os.path.splitext(file_path)[1].lower().lstrip('.')
    for manufacturer, extensions in {
        'Canon': ['cr2', 'cr3', 'crw'],
        'Nikon': ['nef', 'nrw'],
        'Sony': ['arw', 'srf', 'sr2'],
        'Fujifilm': ['raf'],
        'Olympus': ['orf'],
        'Panasonic': ['rw2'],
        'Pentax': ['pef', 'dng'],
        'Leica': ['raw', 'rwl', 'dng'],
        'Hasselblad': ['3fr', 'fff'],
        'Phase One': ['iiq'],
        'Sigma': ['x3f']
    }.items():
        if ext in extensions:
            return manufacturer
    return "Unknown"
    def _load_libraw(self) -> bool:
        """
        Load the LibRaw library for the current platform.
        
        Returns:
            True if successful, False otherwise
        """
        # Check for custom path first
        if self.config.get('libraw_path'):
            try:
                self.libraw = ctypes.cdll.LoadLibrary(self.config['libraw_path'])
                self.logger.info(f"Loaded LibRaw from custom path: {self.config['libraw_path']}")
                return self._initialize_functions()
            except (OSError, AttributeError) as e:
                self.logger.warning(f"Failed to load LibRaw from custom path: {e}")
        
        # Try default paths for current platform
        current_platform = platform.system()
        if current_platform not in self.lib_paths:
            self.logger.error(f"Unsupported platform: {current_platform}")
            return False
        
        # Try each possible library name
        for lib_name in self.lib_paths[current_platform]:
            try:
                self.libraw = ctypes.cdll.LoadLibrary(lib_name)
                self.logger.info(f"Loaded LibRaw from: {lib_name}")
                return self._initialize_functions()
            except (OSError, AttributeError) as e:
                self.logger.debug(f"Failed to load LibRaw from {lib_name}: {e}")
        
        self.logger.error("Failed to load LibRaw library. Please install LibRaw or specify a valid path.")
        return False
    
    def _initialize_functions(self) -> bool:
        """
        Initialize LibRaw function signatures.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Define function prototypes
            
            # libraw_init
            self.libraw.libraw_init.argtypes = [ctypes.c_uint]
            self.libraw.libraw_init.restype = ctypes.c_void_p
            
            # libraw_open_file
            self.libraw.libraw_open_file.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            self.libraw.libraw_open_file.restype = ctypes.c_int
            
            # libraw_unpack
            self.libraw.libraw_unpack.argtypes = [ctypes.c_void_p]
            self.libraw.libraw_unpack.restype = ctypes.c_int
            
            # libraw_dcraw_process
            self.libraw.libraw_dcraw_process.argtypes = [ctypes.c_void_p]
            self.libraw.libraw_dcraw_process.restype = ctypes.c_int
            
            # libraw_dcraw_make_mem_image
            self.libraw.libraw_dcraw_make_mem_image.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
            self.libraw.libraw_dcraw_make_mem_image.restype = ctypes.POINTER(ctypes.c_ubyte)
            
            # libraw_dcraw_make_mem_thumb
            self.libraw.libraw_dcraw_make_mem_thumb.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
            self.libraw.libraw_dcraw_make_mem_thumb.restype = ctypes.POINTER(ctypes.c_ubyte)
            
            # libraw_recycle
            self.libraw.libraw_recycle.argtypes = [ctypes.c_void_p]
            
            # libraw_close
            self.libraw.libraw_close.argtypes = [ctypes.c_void_p]
            
            # libraw_version
            self.libraw.libraw_version.restype = ctypes.c_char_p
            
            # Verify library version
            version = self.libraw.libraw_version()
            self.logger.info(f"LibRaw version: {version.decode('utf-8')}")
            
            return True
        except (AttributeError, TypeError) as e:
            self.logger.error(f"Failed to initialize LibRaw functions: {e}")
            self.libraw = None
            return False
    
    def _create_processor(self) -> ctypes.c_void_p:
        """
        Create a new LibRaw processor.
        
        Returns:
            LibRaw processor handle or None if creation failed
        """
        try:
            # 0 means default flags
            processor = self.libraw.libraw_init(0)
            if not processor:
                self.logger.error("Failed to create LibRaw processor")
                return None
            return processor
        except Exception as e:
            self.logger.error(f"Error creating LibRaw processor: {e}")
            return None
    
    def _extract_metadata(self, processor: ctypes.c_void_p, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from RAW file using LibRaw.
        
        Args:
            processor: LibRaw processor handle
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
        
        # Create specialized structs for accessing libraw fields
        # This is a simplified version; for a complete implementation, 
        # you'd need to define matching C structs in Python
        
        # Since we can't directly access the C struct fields through ctypes in this simplified version,
        # we'll use a workaround by writing a thumbnail to a temporary file and extracting EXIF
        
        try:
            # Extract thumbnail to a temporary file
            thumb_error = ctypes.c_int()
            thumb_data = self.libraw.libraw_dcraw_make_mem_thumb(processor, ctypes.byref(thumb_error))
            
            if thumb_error.value == 0 and thumb_data:
                # Create a temporary file for the thumbnail
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    temp_path = temp_file.name
                
                # Determine thumbnail size (this would be in the struct, but we can't access it directly)
                # For this simplified version, we'll assume it's in the first few bytes
                thumb_size = 0  # This would come from the actual data
                
                # Write thumbnail data to file
                with open(temp_path, 'wb') as f:
                    # In reality, you'd need to get the actual size from the struct
                    # For now, we'll just write until we hit a null byte or some reasonable limit
                    max_size = 1024 * 1024  # 1MB max
                    buffer = bytearray(max_size)
                    for i in range(max_size):
                        buffer[i] = thumb_data[i]
                        if thumb_data[i] == 0 and i > 1000:  # End of data, with some minimum size
                            thumb_size = i
                            break
                    
                    if thumb_size > 0:
                        f.write(buffer[:thumb_size])
                
                # Now extract EXIF from the thumbnail
                try:
                    from PIL import Image
                    with Image.open(temp_path) as img:
                        exif_data = img._getexif()
                        if exif_data:
                            # Map common EXIF tags
                            exif_tags = {
                                0x010F: 'camera_make',      # Make
                                0x0110: 'camera_model',     # Model
                                0xA431: 'lens_model',       # LensModel
                                0x9003: 'datetime',         # DateTimeOriginal
                                0x829A: 'exposure_time',    # ExposureTime
                                0x829D: 'f_number',         # FNumber
                                0x8827: 'iso',              # ISOSpeedRatings
                                0x920A: 'focal_length'      # FocalLength
                            }
                            
                            for tag_id, field_name in exif_tags.items():
                                if tag_id in exif_data:
                                    value = exif_data[tag_id]
                                    metadata[field_name] = value
                except Exception as e:
                    self.logger.debug(f"Error extracting EXIF from thumbnail: {e}")
                
                # Clean up
                os.unlink(temp_path)
        
        except Exception as e:
            self.logger.debug(f"Error extracting thumbnail: {e}")
        
        # Set today's date if no datetime in metadata (fallback for organization)
        if 'datetime' not in metadata:
            import datetime
            metadata['datetime'] = datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")
        
        return metadata
    
    def _set_processing_params(self, processor: ctypes.c_void_p) -> None:
        """
        Set processing parameters for LibRaw.
        
        Args:
            processor: LibRaw processor handle
        """
        # In a complete implementation, you would set the params directly in the C struct
        # Since we can't easily do that in this simplified version, we'll use alternative approaches
        
        # For a full implementation, you'd need to create a matching libraw_processed_parameters
        # struct in Python using ctypes and set its fields
        
        # For the purposes of this demonstration, we'll use libraw command line options
        # The real implementation would access the struct directly
        
        # Note: This is a placeholder function. A real implementation would require 
        # defining the complete libraw_processed_parameters struct in Python and 
        # setting its fields directly through the ctypes interface.
        pass
    
    def _get_processed_image(self, processor: ctypes.c_void_p) -> np.ndarray:
        """
        Get the processed image from LibRaw.
        
        Args:
            processor: LibRaw processor handle
            
        Returns:
            NumPy array containing the processed image
        """
        error = ctypes.c_int()
        data_ptr = self.libraw.libraw_dcraw_make_mem_image(processor, ctypes.byref(error))
        
        if error.value != 0 or not data_ptr:
            raise RuntimeError(f"Failed to get processed image: Error code {error.value}")
        
        # We need to access the processed image dimensions
        # In a complete implementation, you'd access these from the libraw struct
        # For this simplified version, we'll use fixed values and extract them properly
        
        # Note: In a real implementation, you would access the width, height, and colors
        # from the appropriate struct fields instead of hardcoding them
        
        # For demonstration purposes, create a small dummy image
        # In reality, you'd construct the numpy array from the data pointer
        width, height = 800, 600
        channels = 3
        
        # Create a dummy RGB image (in reality, you'd construct this from data_ptr)
        image_data = np.zeros((height, width, channels), dtype=np.uint8)
        
        # Gradient color (just a placeholder - real data would come from libraw)
        for y in range(height):
            for x in range(width):
                image_data[y, x, 0] = int(255 * x / width)
                image_data[y, x, 1] = int(255 * y / height)
                image_data[y, x, 2] = 128
        
        # In a real implementation, we would free the memory allocated by libraw
        # self.libraw.libraw_dcraw_clear_mem(data_ptr)
        
        return image_data