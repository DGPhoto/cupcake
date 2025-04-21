# Aggiungiamo UserSettings e GPUManager al modulo __init__.py principale
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
from .user_settings import UserSettings
from .gpu_utils import GPUManager  # Aggiungiamo il nuovo modulo

__version__ = '0.2.1'  # Aggiorniamo la versione per includere il supporto GPU migliorato

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
    'UserSettings',
    'GPUManager'  # Aggiungiamo il nostro componente
]

# Singleton instances for global access
_settings = None
_gpu_manager = None

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

def get_gpu_manager(suppress_tf_warnings=True) -> GPUManager:
    """
    Get the global GPU manager instance.
    Creates it if it doesn't exist.
    
    Args:
        suppress_tf_warnings: Whether to suppress TensorFlow warnings
        
    Returns:
        GPUManager instance
    """
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager(suppress_tf_warnings=suppress_tf_warnings)
    return _gpu_manager

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
        import datetime
        log_dir = _settings.get_setting("log_directory", 
                                     os.path.join(os.path.expanduser("~"), ".cupcake", "logs"))
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"cupcake_{datetime.datetime.now().strftime('%Y%m%d')}.log")
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

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
    
    # Initialize GPU manager
    gpu_manager = get_gpu_manager()
    use_gpu = settings.get_setting("use_gpu", True)
    
    if use_gpu:
        if gpu_manager.is_gpu_available():
            logger.info(f"GPU support enabled. Using {gpu_manager.get_opencv_backend()} backend.")
        else:
            logger.info("GPU support requested but not available. Using CPU fallback.")
    else:
        logger.info("GPU support disabled in settings. Using CPU for processing.")
    
    # Initialize components
    image_loader = ImageLoader()
    analysis_engine = AnalysisEngine({"use_gpu": use_gpu})
    
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
        "gpu_manager": gpu_manager,
        "image_loader": image_loader,
        "analysis_engine": analysis_engine,
        "rating_system": rating_system,
        "selection_manager": selection_manager,
        "storage_manager": storage_manager,
        "plugin_manager": plugin_manager
    }