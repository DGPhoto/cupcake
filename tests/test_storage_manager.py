# tests/test_storage_manager.py

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.image_loader import ImageLoader
from src.selection_manager import SelectionManager, SelectionStatus, ColorLabel
from src.storage_manager import StorageManager, ExportFormat, NamingPattern, FolderStructure

def test_basic_export():
    """Test basic export functionality."""
    print("\n=== Testing Basic Export Functionality ===")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Request a test directory with images
        test_dir_path = input("\nEnter path to a directory with images (or 'skip' to skip): ")
        
        if test_dir_path.lower() == 'skip':
            print("Skipping export test.")
            return True
        
        if not os.path.exists(test_dir_path) or not os.path.isdir(test_dir_path):
            print(f"Invalid directory: {test_dir_path}")
            return False
        
        # Set up the components
        loader = ImageLoader()
        selection_manager = SelectionManager("Export Test")
        storage_manager = StorageManager(temp_dir)
        
        # Load images
        print(f"Loading images from {test_dir_path}...")
        loaded_images = loader.load_from_directory(test_dir_path)
        
        if not loaded_images:
            print("No images found in directory.")
            return False
        
        print(f"Loaded {len(loaded_images)} images.")
        
        # Register images with selection manager
        selection_manager.register_images_from_loader(loaded_images)
        
        # Mark some images as selected
        print("Marking first 3 images as selected...")
        for i, (path, _, _) in enumerate(loaded_images[:3]):
            selection_manager.mark_as_selected(path)
            
            # Add some ratings and colors for variety
            selection_manager.set_rating(path, (i % 5) + 1)
            color_labels = [ColorLabel.RED, ColorLabel.GREEN, ColorLabel.BLUE]
            selection_manager.set_color_label(path, color_labels[i % 3])
        
        # Export using default settings
        output_dir = os.path.join(temp_dir, "default_export")
        print(f"\nExporting with default settings to {output_dir}...")
        
        stats = storage_manager.export_selected(
            selection_manager,
            output_dir
        )
        
        print(f"Export stats: {stats}")
        
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            print(f"Files in output directory: {len(files)}")
            for file in files[:5]:  # Show first 5 files
                print(f"  - {file}")
        
        # Export with JPEG format and sequence naming
        output_dir = os.path.join(temp_dir, "jpeg_export")
        print(f"\nExporting as JPEGs with sequence naming to {output_dir}...")
        
        stats = storage_manager.export_selected(
            selection_manager,
            output_dir,
            export_format=ExportFormat.JPEG,
            naming_pattern=NamingPattern.SEQUENCE,
            jpeg_quality=90
        )
        
        print(f"Export stats: {stats}")
        
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            print(f"Files in output directory: {len(files)}")
            for file in files[:5]:  # Show first 5 files
                print(f"  - {file}")
        
        # Export with date folder structure
        output_dir = os.path.join(temp_dir, "date_structure")
        print(f"\nExporting with date folder structure to {output_dir}...")
        
        stats = storage_manager.export_selected(
            selection_manager,
            output_dir,
            folder_structure=FolderStructure.DATE
        )
        
        print(f"Export stats: {stats}")
        
        if os.path.exists(output_dir):
            print("Folder structure:")
            for root, dirs, files in os.walk(output_dir):
                level = root.replace(output_dir, '').count(os.sep)
                indent = ' ' * 4 * level
                print(f"{indent}{os.path.basename(root)}/")
                sub_indent = ' ' * 4 * (level + 1)
                for file in files[:3]:  # Show first 3 files in each directory
                    print(f"{sub_indent}{file}")
        
        # Export with custom naming pattern
        output_dir = os.path.join(temp_dir, "custom_naming")
        print(f"\nExporting with custom naming pattern to {output_dir}...")
        
        stats = storage_manager.export_selected(
            selection_manager,
            output_dir,
            naming_pattern=NamingPattern.CUSTOM,
            custom_name_pattern="{rating}stars_{camera_model}_{filename}"
        )
        
        print(f"Export stats: {stats}")
        
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            print(f"Files in output directory: {len(files)}")
            for file in files[:5]:  # Show first 5 files
                print(f"  - {file}")
        
        # Export as Lightroom catalog
        output_dir = os.path.join(temp_dir, "lightroom_catalog")
        print(f"\nExporting as Lightroom catalog to {output_dir}...")
        
        stats = storage_manager.export_as_catalog(
            selection_manager,
            output_dir,
            catalog_format="lightroom"
        )
        
        print(f"Export stats: {stats}")
        
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            print(f"Files in output directory: {len(files)}")
            for file in files[:5]:  # Show first 5 files
                print(f"  - {file}")
        
        # Export metadata to CSV
        output_file = os.path.join(temp_dir, "metadata.csv")
        print(f"\nExporting metadata to CSV: {output_file}...")
        
        stats = storage_manager.export_with_metadata(
            selection_manager,
            metadata_fields=["camera_make", "camera_model", "lens_model", "focal_length", "f_number", "exposure_time", "iso"],
            output_file=output_file,
            format="csv"
        )
        
        print(f"Export stats: {stats}")
        
        if os.path.exists(output_file):
            # Read first few lines of CSV
            with open(output_file, 'r') as f:
                print("\nCSV preview:")
                for i, line in enumerate(f):
                    if i > 5:  # Show first 5 lines
                        break
                    print(f"  {line.strip()}")
    
    return True

def test_file_organization():
    """Test file organization functionality."""
    print("\n=== Testing File Organization Functionality ===")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Request a test directory with images
        test_dir_path = input("\nEnter path to a directory with images (or 'skip' to skip): ")
        
        if test_dir_path.lower() == 'skip':
            print("Skipping organization test.")
            return True
        
        if not os.path.exists(test_dir_path) or not os.path.isdir(test_dir_path):
            print(f"Invalid directory: {test_dir_path}")
            return False
        
        # Set up the storage manager
        storage_manager = StorageManager(temp_dir)
        
        # Organize files by date
        output_dir = os.path.join(temp_dir, "organized_by_date")
        print(f"\nOrganizing files by date to {output_dir}...")
        
        stats = storage_manager.organize_files(
            test_dir_path,
            output_dir,
            folder_structure=FolderStructure.DATE
        )
        
        print(f"Organization stats: {stats}")
        
        if os.path.exists(output_dir):
            print("Folder structure:")
            for root, dirs, files in os.walk(output_dir):
                level = root.replace(output_dir, '').count(os.sep)
                indent = ' ' * 4 * level
                print(f"{indent}{os.path.basename(root)}/")
                if level >= 3:  # If we're 3 levels deep (year/month/day)
                    sub_indent = ' ' * 4 * (level + 1)
                    file_count = len(files)
                    if file_count > 0:
                        print(f"{sub_indent}[{file_count} files]")
                        for file in files[:3]:  # Show first 3 files
                            print(f"{sub_indent}- {file}")
        
        # Organize files with custom pattern
        output_dir = os.path.join(temp_dir, "organized_custom")
        print(f"\nOrganizing files with custom pattern to {output_dir}...")
        
        stats = storage_manager.organize_files(
            test_dir_path,
            output_dir,
            folder_structure=FolderStructure.CUSTOM,
            custom_folder_pattern="{camera_make}/{camera_model}",
            naming_pattern=NamingPattern.DATETIME
        )
        
        print(f"Organization stats: {stats}")
        
        if os.path.exists(output_dir):
            print("Folder structure:")
            for root, dirs, files in os.walk(output_dir):
                level = root.replace(output_dir, '').count(os.sep)
                indent = ' ' * 4 * level
                print(f"{indent}{os.path.basename(root)}/")
                if level >= 2:  # If we're at the camera model level
                    sub_indent = ' ' * 4 * (level + 1)
                    file_count = len(files)
                    if file_count > 0:
                        print(f"{sub_indent}[{file_count} files]")
                        for file in files[:3]:  # Show first 3 files
                            print(f"{sub_indent}- {file}")
    
    return True

def run_tests():
    """Run all tests."""
    print("=== Testing Storage Manager ===\n")
    
    tests = [
        ("Basic Export", test_basic_export),
        ("File Organization", test_file_organization)
    ]
    
    results = []
    
    for name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running Test: {name}")
        print(f"{'='*50}")
        
        try:
            success = test_func()
            results.append((name, success))
            print(f"\nTest '{name}' {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            print(f"\nTest '{name}' FAILED with exception: {e}")
            results.append((name, False))
    
    # Print summary
    print("\n\n=== Test Summary ===")
    for name, success in results:
        print(f"{name}: {'PASSED' if success else 'FAILED'}")

if __name__ == "__main__":
    run_tests()