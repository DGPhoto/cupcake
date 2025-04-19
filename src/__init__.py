# src/__init__.py

"""
Cupcake - Photo Culling Library
"""

__version__ = '0.1.0'

# Import only the components that exist
from .image_loader import ImageLoader
from .analysis_engine import AnalysisEngine, ImageAnalysisResult
from .rating_system import RatingSystem, RatingProfile, ImageRating, RatingCategory, UserPreferenceModel

# We'll uncomment these as we implement them
# from .selection_manager import SelectionManager
# from .storage_manager import StorageManager
# from .plugin_system import PluginManager, CupcakePlugin

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
    # 'SelectionManager',
    # 'StorageManager',
    # 'PluginManager',
    # 'CupcakePlugin'
]