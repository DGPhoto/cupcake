# src/user_settings.py

import os
import json
import logging
from typing import Dict, Any, List, Optional, Set, Union
from dataclasses import asdict, is_dataclass
import datetime
from pathlib import Path

from .rating_system import RatingProfile, UserPreferenceModel


class UserSettings:
    """
    Centralized user settings manager for Cupcake Photo Culling Library.
    Handles storage and retrieval of user preferences, custom profiles, and application settings.
    """
    
    def __init__(self, settings_dir: Optional[str] = None):
        """
        Initialize the user settings manager.
        
        Args:
            settings_dir: Directory to store settings. Defaults to ~/.cupcake/settings
        """
        if settings_dir is None:
            # Default to ~/.cupcake/settings
            settings_dir = os.path.join(os.path.expanduser("~"), ".cupcake", "settings")
        
        self.settings_dir = settings_dir
        self.config_file = os.path.join(settings_dir, "config.json")
        self.profiles_dir = os.path.join(settings_dir, "profiles")
        self.preferences_dir = os.path.join(settings_dir, "preferences")
        
        # Create directories if they don't exist
        os.makedirs(self.settings_dir, exist_ok=True)
        os.makedirs(self.profiles_dir, exist_ok=True)
        os.makedirs(self.preferences_dir, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger("cupcake.settings")
        
        # Default configuration
        self.default_config = {
            "output_directory": os.path.join(os.path.expanduser("~"), "Pictures", "Cupcake"),
            "temp_directory": os.path.join(os.path.expanduser("~"), ".cupcake", "temp"),
            "default_export_format": "original",
            "default_naming_pattern": "original",
            "default_folder_structure": "flat",
            "jpeg_quality": 95,
            "tiff_compression": "lzw",
            "default_rating_profile": "default",
            "auto_culling_enabled": False,
            "auto_culling_threshold": 75.0,
            "plugins_enabled": True,
            "active_plugins": [],
            "learning_enabled": True,
            "ui_theme": "light",
            "file_extensions_filter": [],
            "last_directories": []
        }
        
        # Load configuration
        self.config = self.load_config()
        
        # Initialize profiles dictionary
        self.profiles = {}
        self.load_profiles()
        
        # Initialize user preference models dictionary
        self.preference_models = {}
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            Dictionary with configuration
        """
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                self.logger.info(f"Loaded configuration from {self.config_file}")
                
                # Merge with default config to ensure all keys exist
                merged_config = self.default_config.copy()
                merged_config.update(config)
                return merged_config
                
            except Exception as e:
                self.logger.error(f"Error loading configuration: {e}")
                return self.default_config.copy()
        else:
            self.logger.info("Configuration file not found, using defaults")
            return self.default_config.copy()
    
    def save_config(self) -> bool:
        """
        Save configuration to file.
        
        Returns:
            True if successful
        """
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            self.logger.info(f"Saved configuration to {self.config_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            return False
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a specific setting.
        
        Args:
            key: Setting key
            default: Default value if key not found
            
        Returns:
            Setting value
        """
        return self.config.get(key, default)
    
    def set_setting(self, key: str, value: Any) -> bool:
        """
        Set a specific setting.
        
        Args:
            key: Setting key
            value: Setting value
            
        Returns:
            True if successful
        """
        self.config[key] = value
        return self.save_config()
    
    def update_settings(self, settings: Dict[str, Any]) -> bool:
        """
        Update multiple settings at once.
        
        Args:
            settings: Dictionary of settings to update
            
        Returns:
            True if successful
        """
        self.config.update(settings)
        return self.save_config()
    
    def reset_to_defaults(self) -> bool:
        """
        Reset all settings to default values.
        
        Returns:
            True if successful
        """
        self.config = self.default_config.copy()
        return self.save_config()
    
    def load_profiles(self) -> Dict[str, RatingProfile]:
        """
        Load all rating profiles.
        
        Returns:
            Dictionary of profile names to RatingProfile objects
        """
        self.profiles = {}
        
        # Always load default profile
        default_profile = RatingProfile(name="default", description="Default rating profile")
        self.profiles["default"] = default_profile
        
        # Try to load user-defined profiles
        for filename in os.listdir(self.profiles_dir):
            if filename.endswith('.json'):
                try:
                    profile_path = os.path.join(self.profiles_dir, filename)
                    with open(profile_path, 'r') as f:
                        profile_data = json.load(f)
                        profile = RatingProfile.from_dict(profile_data)
                        self.profiles[profile.name] = profile
                        self.logger.info(f"Loaded profile: {profile.name}")
                except Exception as e:
                    self.logger.error(f"Error loading profile {filename}: {e}")
        
        return self.profiles
    
    def save_profile(self, profile: RatingProfile) -> bool:
        """
        Save a rating profile.
        
        Args:
            profile: RatingProfile to save
            
        Returns:
            True if successful
        """
        try:
            if not profile.validate():
                self.logger.error(f"Profile validation failed: {profile.name}")
                return False
            
            # Convert to dict and save
            profile_dict = profile.to_dict()
            profile_path = os.path.join(self.profiles_dir, f"{profile.name}.json")
            
            with open(profile_path, 'w') as f:
                json.dump(profile_dict, f, indent=2)
            
            # Update in-memory profiles
            self.profiles[profile.name] = profile
            
            # If this is the default profile, update the default setting
            if self.get_setting("default_rating_profile") == profile.name:
                self.config["default_rating_profile"] = profile.name
                self.save_config()
            
            self.logger.info(f"Saved profile: {profile.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving profile {profile.name}: {e}")
            return False
    
    def get_profile(self, name: str) -> Optional[RatingProfile]:
        """
        Get a rating profile by name.
        
        Args:
            name: Profile name
            
        Returns:
            RatingProfile or None if not found
        """
        return self.profiles.get(name)
    
    def delete_profile(self, name: str) -> bool:
        """
        Delete a rating profile.
        
        Args:
            name: Profile name
            
        Returns:
            True if successful
        """
        if name == "default":
            self.logger.error("Cannot delete default profile")
            return False
        
        if name not in self.profiles:
            self.logger.error(f"Profile not found: {name}")
            return False
        
        try:
            # Remove from disk
            profile_path = os.path.join(self.profiles_dir, f"{name}.json")
            if os.path.exists(profile_path):
                os.remove(profile_path)
            
            # Remove from memory
            del self.profiles[name]
            
            # If this was the default profile, reset to "default"
            if self.get_setting("default_rating_profile") == name:
                self.config["default_rating_profile"] = "default"
                self.save_config()
            
            self.logger.info(f"Deleted profile: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting profile {name}: {e}")
            return False
    
    def create_profile(self, name: str, description: str, 
                     template: Optional[str] = None, **kwargs) -> Optional[RatingProfile]:
        """
        Create a new rating profile with optional template.
        
        Args:
            name: Profile name
            description: Profile description
            template: Optional template profile name to base on
            **kwargs: Override parameters for the profile
            
        Returns:
            New RatingProfile or None if failed
        """
        if name in self.profiles and name != "default":
            self.logger.error(f"Profile already exists: {name}")
            return None
        
        try:
            # Start with default or template
            if template and template in self.profiles:
                # Clone the template profile
                template_profile = self.profiles[template]
                profile_dict = template_profile.to_dict()
                profile_dict["name"] = name
                profile_dict["description"] = description
                profile_dict["created_at"] = datetime.datetime.now().isoformat()
                profile_dict["modified_at"] = datetime.datetime.now().isoformat()
                
                # Override with kwargs
                profile_dict.update(kwargs)
                
                profile = RatingProfile.from_dict(profile_dict)
            else:
                # Create new profile with default values
                profile = RatingProfile(name=name, description=description, **kwargs)
            
            # Validate and save
            if not profile.validate():
                self.logger.error(f"Profile validation failed: {name}")
                return None
            
            self.save_profile(profile)
            return profile
            
        except Exception as e:
            self.logger.error(f"Error creating profile {name}: {e}")
            return None
    
    def get_preference_model(self, profile_name: str) -> UserPreferenceModel:
        """
        Get the preference model for a specific profile.
        Creates a new one if not found.
        
        Args:
            profile_name: Profile name
            
        Returns:
            UserPreferenceModel instance
        """
        if profile_name in self.preference_models:
            return self.preference_models[profile_name]
        
        # Try to load from disk
        model_path = os.path.join(self.preferences_dir, f"{profile_name}_preferences.pkl")
        
        if os.path.exists(model_path):
            try:
                model = UserPreferenceModel.load(model_path)
                self.preference_models[profile_name] = model
                return model
            except Exception as e:
                self.logger.error(f"Error loading preference model for {profile_name}: {e}")
        
        # Create new model
        model = UserPreferenceModel()
        self.preference_models[profile_name] = model
        return model
    
    def save_preference_model(self, profile_name: str, model: UserPreferenceModel) -> bool:
        """
        Save a preference model for a specific profile.
        
        Args:
            profile_name: Profile name
            model: UserPreferenceModel to save
            
        Returns:
            True if successful
        """
        try:
            model_path = os.path.join(self.preferences_dir, f"{profile_name}_preferences.pkl")
            model.save(model_path)
            self.preference_models[profile_name] = model
            self.logger.info(f"Saved preference model for {profile_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving preference model for {profile_name}: {e}")
            return False
    
    def get_output_directory(self) -> str:
        """
        Get the configured output directory, creating it if it doesn't exist.
        
        Returns:
            Output directory path
        """
        output_dir = self.get_setting("output_directory")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def get_temp_directory(self) -> str:
        """
        Get the configured temporary directory, creating it if it doesn't exist.
        
        Returns:
            Temporary directory path
        """
        temp_dir = self.get_setting("temp_directory")
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir
    
    def add_recent_directory(self, directory: str, max_entries: int = 10) -> None:
        """
        Add a directory to recent directories list.
        
        Args:
            directory: Directory path
            max_entries: Maximum number of entries to keep
        """
        recent_dirs = self.get_setting("last_directories", [])
        
        # Remove if already exists
        if directory in recent_dirs:
            recent_dirs.remove(directory)
        
        # Add to front
        recent_dirs.insert(0, directory)
        
        # Trim list
        recent_dirs = recent_dirs[:max_entries]
        
        # Save
        self.set_setting("last_directories", recent_dirs)
    
    def get_recent_directories(self) -> List[str]:
        """
        Get list of recent directories.
        
        Returns:
            List of directory paths
        """
        return self.get_setting("last_directories", [])
    
    def create_specialized_profile(self, name: str, specialization: str) -> Optional[RatingProfile]:
        """
        Create a specialized rating profile for a specific photography type.
        
        Args:
            name: Profile name
            specialization: Type of photography (portrait, landscape, etc.)
            
        Returns:
            New RatingProfile or None if failed
        """
        # Define specialized profiles with appropriate weights
        specializations = {
            "portrait": {
                "description": "Optimized for portrait photography",
                "technical_weight": 0.5,
                "composition_weight": 0.5,
                "sharpness_weight": 0.4,    # Faces need to be sharp
                "exposure_weight": 0.3,     # Good exposure for skin tones
                "contrast_weight": 0.2,
                "noise_weight": 0.1,
                "rule_of_thirds_weight": 0.3,
                "symmetry_weight": 0.2,
                "subject_position_weight": 0.5   # Subject positioning is important
            },
            "landscape": {
                "description": "Optimized for landscape photography",
                "technical_weight": 0.6,
                "composition_weight": 0.4,
                "sharpness_weight": 0.35,
                "exposure_weight": 0.35,    # Dynamic range is important
                "contrast_weight": 0.2,     # Good contrast for depth
                "noise_weight": 0.1,
                "rule_of_thirds_weight": 0.6,  # Strong composition emphasis
                "symmetry_weight": 0.2,
                "subject_position_weight": 0.2
            },
            "wildlife": {
                "description": "Optimized for wildlife photography",
                "technical_weight": 0.7,     # Technical aspects more important
                "composition_weight": 0.3,
                "sharpness_weight": 0.5,     # Critical for wildlife
                "exposure_weight": 0.3,
                "contrast_weight": 0.1,
                "noise_weight": 0.1,
                "rule_of_thirds_weight": 0.4,
                "symmetry_weight": 0.1,
                "subject_position_weight": 0.5  # Subject positioning is key
            },
            "street": {
                "description": "Optimized for street photography",
                "technical_weight": 0.4,     # Less emphasis on technical perfection
                "composition_weight": 0.6,    # More emphasis on composition and moment
                "sharpness_weight": 0.3,
                "exposure_weight": 0.4,
                "contrast_weight": 0.2,
                "noise_weight": 0.1,
                "rule_of_thirds_weight": 0.3,
                "symmetry_weight": 0.3,
                "subject_position_weight": 0.4
            },
            "architecture": {
                "description": "Optimized for architectural photography",
                "technical_weight": 0.6,
                "composition_weight": 0.4,
                "sharpness_weight": 0.3,
                "exposure_weight": 0.3,
                "contrast_weight": 0.3,      # Strong contrast for structure
                "noise_weight": 0.1,
                "rule_of_thirds_weight": 0.3,
                "symmetry_weight": 0.5,      # Symmetry is often important
                "subject_position_weight": 0.2
            },
            "macro": {
                "description": "Optimized for macro photography",
                "technical_weight": 0.8,     # Very technical genre
                "composition_weight": 0.2,
                "sharpness_weight": 0.6,     # Critical for macro
                "exposure_weight": 0.2,
                "contrast_weight": 0.1,
                "noise_weight": 0.1,
                "rule_of_thirds_weight": 0.4,
                "symmetry_weight": 0.4,
                "subject_position_weight": 0.2
            },
            "night": {
                "description": "Optimized for night photography",
                "technical_weight": 0.6,
                "composition_weight": 0.4,
                "sharpness_weight": 0.2,     # Less critical at night
                "exposure_weight": 0.4,      # Critical for night scenes
                "contrast_weight": 0.3,
                "noise_weight": 0.1,         # Noise more acceptable
                "rule_of_thirds_weight": 0.4,
                "symmetry_weight": 0.3,
                "subject_position_weight": 0.3
            },
            "black_and_white": {
                "description": "Optimized for black and white photography",
                "technical_weight": 0.5,
                "composition_weight": 0.5,
                "sharpness_weight": 0.3,
                "exposure_weight": 0.3,
                "contrast_weight": 0.4,      # Contrast is very important for B&W
                "noise_weight": 0.0,         # Noise can be artistic in B&W
                "rule_of_thirds_weight": 0.3,
                "symmetry_weight": 0.3,
                "subject_position_weight": 0.4
            },
            "product": {
                "description": "Optimized for product photography",
                "technical_weight": 0.8,     # Very technical focus
                "composition_weight": 0.2,
                "sharpness_weight": 0.4,     # Products need to be sharp
                "exposure_weight": 0.3,
                "contrast_weight": 0.2,
                "noise_weight": 0.1,
                "rule_of_thirds_weight": 0.3,
                "symmetry_weight": 0.5,      # Often centered or symmetric
                "subject_position_weight": 0.2
            },
            "sports": {
                "description": "Optimized for sports photography",
                "technical_weight": 0.7,
                "composition_weight": 0.3,
                "sharpness_weight": 0.5,     # Critical for action
                "exposure_weight": 0.3,
                "contrast_weight": 0.1,
                "noise_weight": 0.1,
                "rule_of_thirds_weight": 0.3,
                "symmetry_weight": 0.1,
                "subject_position_weight": 0.6  # Action positioning is key
            }
        }
        
        # Check if specialization exists
        if specialization.lower() not in specializations:
            self.logger.error(f"Unknown specialization: {specialization}")
            return None
        
        # Get params for this specialization
        params = specializations[specialization.lower()]
        
        # Estrai la descrizione e rimuovila dal dizionario dei parametri per evitare il doppio passaggio
        description = params.pop("description")
        
        # Create and return the profile
        return self.create_profile(name, description, **params)
    
    def export_settings(self, filepath: str) -> bool:
        """
        Export all settings to a JSON file.
        
        Args:
            filepath: Path to export to
            
        Returns:
            True if successful
        """
        try:
            # Prepare export data
            export_data = {
                "config": self.config,
                "profiles": {}
            }
            
            # Add profiles
            for name, profile in self.profiles.items():
                if name != "default":  # No need to export default profile
                    export_data["profiles"][name] = profile.to_dict()
            
            # Write to file
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
                
            self.logger.info(f"Exported settings to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting settings: {e}")
            return False
    
    def import_settings(self, filepath: str, merge: bool = False) -> bool:
        """
        Import settings from a JSON file.
        
        Args:
            filepath: Path to import from
            merge: If True, merge with existing settings; if False, replace
            
        Returns:
            True if successful
        """
        try:
            # Read file
            with open(filepath, 'r') as f:
                import_data = json.load(f)
            
            # Import configuration
            if "config" in import_data:
                if merge:
                    self.config.update(import_data["config"])
                else:
                    self.config = import_data["config"]
                self.save_config()
            
            # Import profiles
            if "profiles" in import_data:
                for name, profile_data in import_data["profiles"].items():
                    profile = RatingProfile.from_dict(profile_data)
                    self.save_profile(profile)
            
            self.logger.info(f"Imported settings from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error importing settings: {e}")
            return False