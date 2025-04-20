# tests/test_user_settings.py

import os
import sys
import tempfile
import shutil
import json
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.user_settings import UserSettings
from src.rating_system import RatingProfile, UserPreferenceModel

def test_basic_settings():
    """Test basic settings functionality."""
    print("\n=== Testing Basic Settings Functionality ===")
    
    # Create a temporary directory for settings
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create settings instance with custom directory
        settings = UserSettings(temp_dir)
        
        # Test getting default settings
        output_dir = settings.get_setting("output_directory")
        print(f"Default output directory: {output_dir}")
        
        # Test setting and getting a setting
        test_path = os.path.join(temp_dir, "test_output")
        settings.set_setting("output_directory", test_path)
        updated_path = settings.get_setting("output_directory")
        print(f"Updated output directory: {updated_path}")
        assert updated_path == test_path, "Setting was not updated correctly"
        
        # Test updating multiple settings
        updates = {
            "jpeg_quality": 92,
            "default_export_format": "jpeg",
            "default_naming_pattern": "sequence"
        }
        settings.update_settings(updates)
        
        # Verify multiple updates
        print("\nUpdated Settings:")
        for key, expected in updates.items():
            value = settings.get_setting(key)
            print(f"  {key}: {value}")
            assert value == expected, f"Setting {key} was not updated correctly"
        
        # Check that settings file was created
        config_file = os.path.join(temp_dir, "config.json")
        assert os.path.exists(config_file), "Config file was not created"
        
        print("\nTest passed: Basic settings functionality")
    
    return True

def test_profile_management():
    """Test profile management functionality."""
    print("\n=== Testing Profile Management ===")
    
    # Create a temporary directory for settings
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create settings instance with custom directory
        settings = UserSettings(temp_dir)
        
        # Check default profile
        default_profile = settings.get_profile("default")
        print(f"Default profile: {default_profile.name}")
        assert default_profile is not None, "Default profile not found"
        
        # Create a custom profile
        custom_profile = RatingProfile(
            name="test_profile",
            description="Test profile description",
            technical_weight=0.7,
            composition_weight=0.3
        )
        
        # Save the profile
        saved = settings.save_profile(custom_profile)
        assert saved, "Failed to save profile"
        
        # Load the profile
        loaded_profile = settings.get_profile("test_profile")
        print(f"Loaded profile: {loaded_profile.name} - {loaded_profile.description}")
        assert loaded_profile is not None, "Failed to load profile"
        assert loaded_profile.technical_weight == 0.7, "Profile data incorrect"
        
        # Create a specialized profile
        specialized = settings.create_specialized_profile("landscape_test", "landscape")
        assert specialized is not None, "Failed to create specialized profile"
        print(f"Created specialized profile: {specialized.name} - {specialized.description}")
        
        # Get all profiles
        profiles = settings.load_profiles()
        print(f"\nAvailable profiles: {', '.join(profiles.keys())}")
        assert len(profiles) >= 3, "Expected at least 3 profiles"
        
        # Delete a profile
        deleted = settings.delete_profile("test_profile")
        assert deleted, "Failed to delete profile"
        
        # Check profile was deleted
        profiles = settings.load_profiles()
        assert "test_profile" not in profiles, "Profile was not deleted"
        print(f"Profiles after deletion: {', '.join(profiles.keys())}")
        
        print("\nTest passed: Profile management")
    
    return True

def test_preference_models():
    """Test preference model functionality."""
    print("\n=== Testing Preference Models ===")
    
    # Create a temporary directory for settings
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create settings instance with custom directory
        settings = UserSettings(temp_dir)
        
        # Get a new preference model
        model = settings.get_preference_model("test_model")
        assert model is not None, "Failed to create preference model"
        
        # Modify model
        model.sharpness_factor = 1.2
        model.exposure_factor = 0.8
        
        # Save model
        saved = settings.save_preference_model("test_model", model)
        assert saved, "Failed to save preference model"
        
        # Get the model again (should load from disk)
        loaded_model = settings.get_preference_model("test_model")
        assert loaded_model.sharpness_factor == 1.2, "Loaded model data incorrect"
        assert loaded_model.exposure_factor == 0.8, "Loaded model data incorrect"
        
        print(f"Preference model saved and loaded successfully")
        print(f"Factors: sharpness={loaded_model.sharpness_factor}, exposure={loaded_model.exposure_factor}")
        
        print("\nTest passed: Preference models")
    
    return True

def test_import_export():
    """Test import and export functionality."""
    print("\n=== Testing Import and Export ===")
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir1, tempfile.TemporaryDirectory() as temp_dir2:
        # Create settings instance with custom directory
        settings1 = UserSettings(temp_dir1)
        
        # Create some custom data
        settings1.set_setting("output_directory", "/custom/path")
        settings1.set_setting("jpeg_quality", 92)
        
        # Create a custom profile
        custom_profile = RatingProfile(
            name="export_test",
            description="Profile for export testing",
            technical_weight=0.6,
            composition_weight=0.4,
            sharpness_weight=0.4,
            exposure_weight=0.3,
            contrast_weight=0.2,
            noise_weight=0.1
        )
        settings1.save_profile(custom_profile)
        
        # Export settings
        export_path = os.path.join(temp_dir1, "export.json")
        exported = settings1.export_settings(export_path)
        assert exported, "Failed to export settings"
        assert os.path.exists(export_path), "Export file was not created"
        
        # Create second settings instance
        settings2 = UserSettings(temp_dir2)
        
        # Import settings
        imported = settings2.import_settings(export_path)
        assert imported, "Failed to import settings"
        
        # Verify imported settings
        output_dir = settings2.get_setting("output_directory")
        assert output_dir == "/custom/path", "Setting not imported correctly"
        
        jpeg_quality = settings2.get_setting("jpeg_quality")
        assert jpeg_quality == 92, "Setting not imported correctly"
        
        # Verify imported profile
        profile = settings2.get_profile("export_test")
        assert profile is not None, "Profile not imported"
        assert profile.description == "Profile for export testing", "Profile data incorrect"
        
        print(f"Settings exported to: {export_path}")
        print(f"Settings imported successfully")
        print(f"Imported profile: {profile.name} - {profile.description}")
        
        print("\nTest passed: Import and export")
    
    return True

def test_recent_directories():
    """Test recent directories functionality."""
    print("\n=== Testing Recent Directories ===")
    
    # Create a temporary directory for settings
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create settings instance with custom directory
        settings = UserSettings(temp_dir)
        
        # Add some directories
        test_dirs = [
            "/path/to/dir1",
            "/path/to/dir2",
            "/path/to/dir3",
            "/path/to/dir4"
        ]
        
        for directory in test_dirs:
            settings.add_recent_directory(directory)
        
        # Get recent directories
        recent = settings.get_recent_directories()
        print(f"Recent directories: {recent}")
        
        # Should be in reverse order (most recent first)
        assert recent[0] == test_dirs[-1], "Recent directories order incorrect"
        assert len(recent) == len(test_dirs), "Recent directories count incorrect"
        
        # Add a duplicate (should move to front)
        settings.add_recent_directory(test_dirs[0])
        recent = settings.get_recent_directories()
        assert recent[0] == test_dirs[0], "Duplicate directory not moved to front"
        
        # Test max entries
        for i in range(10):
            settings.add_recent_directory(f"/path/to/extra{i}")
        
        recent = settings.get_recent_directories()
        assert len(recent) <= 10, "Max entries not enforced"
        
        print(f"Recent directories functionality working correctly")
        print(f"Current recent directories (max 10): {len(recent)}")
        
        print("\nTest passed: Recent directories")
    
    return True

def run_tests():
    """Run all tests."""
    print("=== Testing User Settings ===\n")
    
    tests = [
        ("Basic Settings", test_basic_settings),
        ("Profile Management", test_profile_management),
        ("Preference Models", test_preference_models),
        ("Import/Export", test_import_export),
        ("Recent Directories", test_recent_directories)
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
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Print summary
    print("\n\n=== Test Summary ===")
    for name, success in results:
        print(f"{name}: {'PASSED' if success else 'FAILED'}")

if __name__ == "__main__":
    run_tests()