# tests/test_image_loader.py

import os
import sys
import numpy as np

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.image_loader import ImageLoader
from src.image_formats import ImageFormats

def test_supported_formats():
    """Test the supported formats detection."""
    print("\nSupported formats:")
    print("Standard formats:", ', '.join(sorted(ImageFormats.STANDARD_FORMATS)))
    
    print("\nRAW formats by manufacturer:")
    for manufacturer, formats in ImageFormats.RAW_FORMATS.items():
        print(f"  {manufacturer}: {', '.join(sorted(formats))}")
    
    # Test format detection
    test_extensions = ['jpg', 'png', 'cr2', 'nef', 'raf', 'arw', 'dng', 'xyz']
    print("\nTesting format detection:")
    for ext in test_extensions:
        supported = ImageFormats.is_supported_format(ext)
        is_raw = ImageFormats.is_raw_format(ext)
        manufacturer = ImageFormats.get_manufacturer_for_raw_format(ext) if is_raw else "N/A"
        
        print(f"  .{ext}: Supported: {supported}, RAW: {is_raw}, Manufacturer: {manufacturer}")
    
    return True

def test_load_from_path():
    """Test loading a single image."""
    loader = ImageLoader()
    
    # Request a test image path
    while True:
        test_image_path = input("\nEnter path to a test image (or 'skip' to skip): ")
        
        if test_image_path.lower() == 'skip':
            print("Skipping single image test.")
            return True
        
        if not os.path.exists(test_image_path):
            print(f"File doesn't exist: {test_image_path}")
            continue
        
        extension = test_image_path.split('.')[-1].lower()
        if not ImageFormats.is_supported_format(extension):
            print(f"Warning: Format '{extension}' might not be supported.")
        
        try:
            image_data, metadata = loader.load_from_path(test_image_path)
            
            print(f"\nSuccessfully loaded image: {metadata['filename']}")
            print(f"Image dimensions: {image_data.shape}")
            print("\nMetadata:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
            
            return True
        except Exception as e:
            print(f"Error loading image: {e}")
            print("Would you like to try another image? (y/n)")
            if input().lower() != 'y':
                return False

def test_load_from_directory():
    """Test loading images from a directory."""
    loader = ImageLoader()
    
    # Request a test directory path
    while True:
        test_dir_path = input("\nEnter path to a directory with images (or 'skip' to skip): ")
        
        if test_dir_path.lower() == 'skip':
            print("Skipping directory test.")
            return True
        
        if not os.path.exists(test_dir_path):
            print(f"Directory doesn't exist: {test_dir_path}")
            continue
            
        if not os.path.isdir(test_dir_path):
            print(f"Not a directory: {test_dir_path}")
            continue
        
        try:
            print(f"Loading images from: {test_dir_path}")
            results = loader.load_from_directory(test_dir_path)
            
            print(f"\nSuccessfully loaded {len(results)} images from directory")
            
            # Print details for first 5 images
            for i, (path, image_data, metadata) in enumerate(results[:5]):
                print(f"\nImage {i+1}: {os.path.basename(path)}")
                print(f"  Dimensions: {image_data.shape}")
                print(f"  Size: {metadata['filesize'] / 1024:.1f} KB")
                print(f"  Format: {metadata['extension']} ({metadata['is_raw'] and 'RAW' or 'Standard'})")
                
                # Print some EXIF data if available
                if 'datetime' in metadata:
                    print(f"  Date: {metadata['datetime']}")
                if 'camera_model' in metadata:
                    print(f"  Camera: {metadata['camera_model']}")
            
            return True
        except Exception as e:
            print(f"Error loading images from directory: {e}")
            print("Would you like to try another directory? (y/n)")
            if input().lower() != 'y':
                return False

def run_tests():
    """Run all tests."""
    print("=== Testing Image Loader ===\n")
    
    print("1. Testing supported formats...")
    test_supported_formats()
    
    print("\n2. Testing single image loading...")
    test_load_from_path()
    
    print("\n3. Testing directory loading...")
    test_load_from_directory()

if __name__ == "__main__":
    run_tests()