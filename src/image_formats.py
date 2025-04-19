# src/image_formats.py

from typing import Dict, Set, List

class ImageFormats:
    """
    Defines and manages supported image formats for the Cupcake application.
    Organizes formats by type (standard, raw) and by manufacturer.
    """
    
    # Standard image formats
    STANDARD_FORMATS: Set[str] = {
        # Common web/standard formats
        'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'tif', 'webp',
        
        # High bit-depth formats
        'psd', 'xcf', 'heic', 'heif', 'jp2', 'j2k'
    }
    
    # RAW formats by manufacturer
    RAW_FORMATS: Dict[str, Set[str]] = {
        'Canon': {'cr2', 'cr3', 'crw'},
        'Nikon': {'nef', 'nrw'},
        'Sony': {'arw', 'srf', 'sr2'},
        'Fujifilm': {'raf'},
        'Olympus': {'orf'},
        'Panasonic': {'rw2'},
        'Pentax': {'pef', 'dng'},
        'Leica': {'raw', 'rwl', 'dng'},
        'Hasselblad': {'3fr', 'fff'},
        'Phase One': {'iiq'},
        'Sigma': {'x3f'},
        'Generic': {'raw', 'dng'}
    }
    
    # File extensions to MIME types mapping
    MIME_TYPES: Dict[str, str] = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif',
        'bmp': 'image/bmp',
        'tiff': 'image/tiff',
        'tif': 'image/tiff',
        'webp': 'image/webp',
        'heic': 'image/heic',
        'heif': 'image/heif',
        'raw': 'image/raw',
        'dng': 'image/x-adobe-dng'
        # Other MIME types would be added as needed
    }
    
    @classmethod
    def get_all_supported_extensions(cls) -> Set[str]:
        """Get a set of all supported file extensions."""
        extensions = cls.STANDARD_FORMATS.copy()
        
        for manufacturer_formats in cls.RAW_FORMATS.values():
            extensions.update(manufacturer_formats)
            
        return extensions
    
    @classmethod
    def is_supported_format(cls, extension: str) -> bool:
        """Check if a file extension is supported."""
        extension = extension.lower().lstrip('.')
        return extension in cls.get_all_supported_extensions()
    
    @classmethod
    def is_raw_format(cls, extension: str) -> bool:
        """Check if a file extension is a RAW format."""
        extension = extension.lower().lstrip('.')
        
        for formats in cls.RAW_FORMATS.values():
            if extension in formats:
                return True
                
        return False
    
    @classmethod
    def get_manufacturer_for_raw_format(cls, extension: str) -> str:
        """Get the camera manufacturer for a RAW format."""
        extension = extension.lower().lstrip('.')
        
        for manufacturer, formats in cls.RAW_FORMATS.items():
            if extension in formats:
                return manufacturer
                
        return "Unknown"
    
    @classmethod
    def get_mime_type(cls, extension: str) -> str:
        """Get the MIME type for a file extension."""
        extension = extension.lower().lstrip('.')
        
        if extension in cls.MIME_TYPES:
            return cls.MIME_TYPES[extension]
            
        if cls.is_raw_format(extension):
            # Default MIME type for RAW formats
            manufacturer = cls.get_manufacturer_for_raw_format(extension)
            return f"image/x-{manufacturer.lower()}-{extension}"
            
        return "application/octet-stream"