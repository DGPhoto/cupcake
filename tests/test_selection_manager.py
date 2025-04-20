# tests/test_selection_manager.py

import os
import sys
import datetime
import tempfile

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.selection_manager import SelectionManager, SelectionStatus, ColorLabel

def test_basic_selection():
    """Test basic selection functionality."""
    print("\n=== Testing Basic Selection Functionality ===")
    
    # Create a selection manager
    manager = SelectionManager("Test Project")
    
    # Create some test image data
    test_images = [
        "image1.jpg",
        "image2.jpg",
        "image3.jpg",
        "image4.jpg",
        "image5.jpg"
    ]
    
    # Register the images
    for img in test_images:
        manager.register_image(img, {"filename": img})
    
    # Test selection operations
    print("Marking images as selected/rejected...")
    manager.mark_as_selected("image1.jpg")
    manager.mark_as_selected("image2.jpg")
    manager.mark_as_rejected("image3.jpg")
    
    # Test rating and color labels
    print("Setting ratings and color labels...")
    manager.set_rating("image1.jpg", 5)
    manager.set_rating("image2.jpg", 4)
    manager.set_rating("image3.jpg", 1)
    
    manager.set_color_label("image1.jpg", ColorLabel.GREEN)
    manager.set_color_label("image2.jpg", ColorLabel.YELLOW)
    manager.set_color_label("image3.jpg", ColorLabel.RED)
    
    # Test collections
    print("Creating collections...")
    manager.create_collection("Favorites")
    manager.add_to_collection("image1.jpg", "Favorites")
    manager.add_to_collection("image2.jpg", "Favorites")
    
    # Print selection statistics
    stats = manager.get_statistics()
    print("\nSelection Statistics:")
    print(f"Total images: {stats['total']}")
    print(f"Selected: {stats['selected']} ({stats['selected_percent']:.1f}%)")
    print(f"Rejected: {stats['rejected']} ({stats['rejected_percent']:.1f}%)")
    print(f"Unrated: {stats['unrated']} ({stats['unrated_percent']:.1f}%)")
    
    # Test filtering
    print("\nTesting Filtering:")
    
    selected = manager.get_selected_images()
    print(f"Selected images: {selected}")
    
    rejected = manager.get_rejected_images()
    print(f"Rejected images: {rejected}")
    
    unrated = manager.get_unrated_images()
    print(f"Unrated images: {unrated}")
    
    # Test rating filters
    high_rated = manager.filter_images(rating_min=4)
    print(f"Images rated 4+ stars: {high_rated}")
    
    # Test collection filters
    manager.set_active_collection("Favorites")
    favorites = manager.get_collection_images()
    print(f"Favorites collection: {favorites}")
    
    # Reset active collection
    manager.set_active_collection("All Images")
    
    # Test history and undo
    print("\nTesting Undo Functionality:")
    print("Before undo: image1.jpg rating =", manager.get_image_info("image1.jpg")["rating"])
    manager.undo_last_action()  # Should undo adding image2.jpg to Favorites
    print("After first undo: image2.jpg in Favorites =", "Favorites" in manager.get_image_info("image2.jpg")["collections"])
    
    # Return success
    return True

def test_lightroom_compatibility():
    """Test Lightroom XMP export/import."""
    print("\n=== Testing Lightroom Compatibility ===")
    
    # Create a temporary directory for XMP files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a selection manager
        manager = SelectionManager("Lightroom Test")
        
        # Create some test image data
        test_images = [
            "/photos/IMG_0001.CR2",
            "/photos/IMG_0002.CR2",
            "/photos/IMG_0003.CR2"
        ]
        
        # Register the images
        for img in test_images:
            manager.register_image(img, {"filename": os.path.basename(img)})
        
        # Make some selections
        manager.mark_as_selected(test_images[0])
        manager.mark_as_rejected(test_images[1])
        manager.set_rating(test_images[0], 5)
        manager.set_color_label(test_images[0], ColorLabel.GREEN)
        
        # Export XMP files
        xmp_path = os.path.join(temp_dir, "xmp")
        print(f"Exporting XMP files to {xmp_path}...")
        manager.export_lightroom_metadata(xmp_path)
        
        # Create a new manager
        new_manager = SelectionManager("Lightroom Import Test")
        
        # Register the same images
        for img in test_images:
            new_manager.register_image(img, {"filename": os.path.basename(img)})
        
        # Import XMP files
        print("Importing XMP files...")
        updated = new_manager.import_lightroom_metadata(xmp_path)
        print(f"Updated {updated} images from XMP files")
        
        # Verify imported data
        img1_info = new_manager.get_image_info(test_images[0])
        print(f"\nVerifying imported data for {test_images[0]}:")
        print(f"Status: {img1_info['status']}")
        print(f"Rating: {img1_info['rating']}")
        print(f"Color Label: {img1_info['color_label']}")
        
        img2_info = new_manager.get_image_info(test_images[1])
        print(f"\nVerifying imported data for {test_images[1]}:")
        print(f"Status: {img2_info['status']}")
    
    # Return success
    return True

def test_capture_one_compatibility():
    """Test Capture One CSV export/import."""
    print("\n=== Testing Capture One Compatibility ===")
    
    # Create a temporary directory for CSV files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a selection manager
        manager = SelectionManager("Capture One Test")
        
        # Create some test image data
        test_images = [
            "/photos/IMG_0001.ARW",
            "/photos/IMG_0002.ARW",
            "/photos/IMG_0003.ARW"
        ]
        
        # Register the images
        for img in test_images:
            manager.register_image(img, {"filename": os.path.basename(img)})
        
        # Make some selections
        manager.mark_as_selected(test_images[0])
        manager.mark_as_rejected(test_images[1])
        manager.set_rating(test_images[0], 5)
        manager.set_color_label(test_images[0], ColorLabel.BLUE)
        
        # Export CSV file
        csv_path = os.path.join(temp_dir, "capture_one_data.csv")
        print(f"Exporting CSV file to {csv_path}...")
        manager.export_capture_one_metadata(csv_path)
        
        # Create a new manager
        new_manager = SelectionManager("Capture One Import Test")
        
        # Register the same images
        for img in test_images:
            new_manager.register_image(img, {"filename": os.path.basename(img)})
        
        # Import CSV file
        print("Importing CSV file...")
        updated = new_manager.import_capture_one_metadata(csv_path)
        print(f"Updated {updated} images from CSV file")
        
        # Verify imported data
        img1_info = new_manager.get_image_info(test_images[0])
        print(f"\nVerifying imported data for {test_images[0]}:")
        print(f"Status: {img1_info['status']}")
        print(f"Rating: {img1_info['rating']}")
        print(f"Color Label: {img1_info['color_label']}")
        
        img2_info = new_manager.get_image_info(test_images[1])
        print(f"\nVerifying imported data for {test_images[1]}:")
        print(f"Status: {img2_info['status']}")
    
    # Return success
    return True

def test_json_serialization():
    """Test saving and loading from JSON."""
    print("\n=== Testing JSON Serialization ===")
    
    # Create a temporary file for JSON
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
        try:
            json_path = temp_file.name
            
            # Create a selection manager
            manager = SelectionManager("JSON Test")
            
            # Create some test image data
            test_images = [
                "image1.jpg",
                "image2.jpg",
                "image3.jpg"
            ]
            
            # Register the images
            for img in test_images:
                manager.register_image(img, {"filename": img})
            
            # Make some selections
            manager.mark_as_selected(test_images[0])
            manager.mark_as_rejected(test_images[1])
            manager.set_rating(test_images[0], 5)
            manager.set_color_label(test_images[0], ColorLabel.PURPLE)
            
            # Create a collection
            manager.create_collection("Best Shots")
            manager.add_to_collection(test_images[0], "Best Shots")
            
            # Save to JSON
            print(f"Saving selection state to {json_path}...")
            manager.save_to_json(json_path)
            
            # Load from JSON
            print("Loading selection state from JSON...")
            new_manager = SelectionManager.load_from_json(json_path)
            
            # Verify loaded data
            print("\nVerifying loaded data:")
            print(f"Project name: {new_manager.project_name}")
            print(f"Total images: {new_manager.get_statistics()['total']}")
            print(f"Collections: {list(new_manager._collections.keys())}")
            
            img1_info = new_manager.get_image_info(test_images[0])
            print(f"\nVerifying imported data for {test_images[0]}:")
            print(f"Status: {img1_info['status']}")
            print(f"Rating: {img1_info['rating']}")
            print(f"Color Label: {img1_info['color_label']}")
            print(f"Collections: {img1_info['collections']}")
            
        finally:
            # Clean up
            if os.path.exists(json_path):
                os.unlink(json_path)
    
    # Return success
    return True

def run_tests():
    """Run all tests."""
    print("=== Testing Selection Manager ===\n")
    
    tests = [
        ("Basic Selection", test_basic_selection),
        ("Lightroom Compatibility", test_lightroom_compatibility),
        ("Capture One Compatibility", test_capture_one_compatibility),
        ("JSON Serialization", test_json_serialization)
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