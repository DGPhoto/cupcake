# Aggiungiamo UserSettings al modulo __init__.py principale
# Aggiornamento di src/__init__.py

import os
import sys
import logging
from typing import Dict, Any

# Import the existing components
from .image_loader import ImageLoader
from .analysis_engine import AnalysisEngine, ImageAnalysisResult
from .rating_system import RatingSystem, RatingProfile, ImageRating, RatingCategory, UserPreferenceModel
from .selection_manager import SelectionManager, SelectionStatus, ColorLabel
from .storage_manager import StorageManager, ExportFormat, NamingPattern, FolderStructure
from .plugin_system import PluginManager, CupcakePlugin, PluginType, PluginHook
from .error_suppressor import ErrorSuppressor
from .user_settings import UserSettings  # Aggiungiamo il nostro nuovo modulo

__version__ = '0.2.0'  # Aggiorniamo la versione per includere il nuovo sistema di impostazioni

# Package information
__all__ = [
    'ImageLoader',
    'AnalysisEngine',
    'ImageAnalysisResult',
    'RatingSystem',
    'RatingProfile',
    'ImageRating',
    'RatingCategory',
    'UserPreferenceModel',
    'SelectionManager',
    'SelectionStatus',
    'ColorLabel',
    'StorageManager',
    'ExportFormat', 
    'NamingPattern',
    'FolderStructure',
    'PluginManager',
    'CupcakePlugin',
    'PluginType',
    'PluginHook',
    'ErrorSuppressor',
    'UserSettings'  # Aggiungiamo il nostro componente
]

# Singleton instance for global access
_settings = None

def get_settings() -> UserSettings:
    """
    Get the global settings instance.
    Creates it if it doesn't exist.
    
    Returns:
        UserSettings instance
    """
    global _settings
    if _settings is None:
        _settings = UserSettings()
    return _settings

# Initialize logging
def setup_logging(level=logging.INFO):
    """
    Configure logging for Cupcake.
    
    Args:
        level: Logging level
    """
    logger = logging.getLogger('cupcake')
    logger.setLevel(level)
    
    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    # Also log to file if enabled in settings
    if _settings and _settings.get_setting("enable_file_logging", False):
        log_dir = _settings.get_setting("log_directory", 
                                      os.path.join(os.path.expanduser("~"), ".cupcake", "logs"))
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"cupcake_{datetime.datetime.now().strftime('%Y%m%d')}.log")
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

# Esempio di utilizzo:
"""
# In qualsiasi altro modulo, possiamo importare e usare get_settings:

from src import get_settings

# Ottieni impostazioni
settings = get_settings()

# Ottieni e imposta valori
output_dir = settings.get_setting("output_directory")
settings.set_setting("jpeg_quality", 90)

# Carica un profilo di rating
profile = settings.get_profile("landscape")

# Oppure crea un profilo specializzato
settings.create_specialized_profile("my_landscape", "landscape")

# Ottieni il modello di preferenze per un profilo
preference_model = settings.get_preference_model("my_landscape")
"""

# Esempio di inizializzazione dell'applicazione

def initialize_application():
    """Initialize the Cupcake application with all components."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger('cupcake.init')
    logger.info("Initializing Cupcake Photo Culling Library...")
    
    # Get settings
    settings = get_settings()
    logger.info(f"Settings loaded from {settings.settings_dir}")
    
    # Initialize components
    image_loader = ImageLoader()
    analysis_engine = AnalysisEngine()
    
    # Get default rating profile from settings
    default_profile_name = settings.get_setting("default_rating_profile", "default")
    rating_system = RatingSystem()
    
    # Load available profiles
    profiles = settings.load_profiles()
    for name, profile in profiles.items():
        if name != "default":  # Default is already loaded by RatingSystem
            rating_system.add_profile(profile)
    
    # Initialize selection manager
    selection_manager = SelectionManager("Cupcake Session")
    
    # Initialize storage manager with configured output directory
    output_dir = settings.get_output_directory()
    storage_manager = StorageManager(output_dir)
    
    # Initialize plugin manager if enabled
    plugins_enabled = settings.get_setting("plugins_enabled", True)
    plugin_manager = None
    if plugins_enabled:
        plugin_dirs = settings.get_setting("plugin_directories", ["plugins"])
        plugin_manager = PluginManager(plugin_dirs)
        plugin_manager.discover_plugins()
        
        # Load only active plugins
        active_plugins = settings.get_setting("active_plugins", [])
        if active_plugins:
            for plugin_name in active_plugins:
                plugin_manager.load_plugin(plugin_name)
        else:
            # Or load all if none specified
            plugin_manager.load_all_plugins()
    
    logger.info("Cupcake initialization complete.")
    
    return {
        "settings": settings,
        "image_loader": image_loader,
        "analysis_engine": analysis_engine,
        "rating_system": rating_system,
        "selection_manager": selection_manager,
        "storage_manager": storage_manager,
        "plugin_manager": plugin_manager
    }