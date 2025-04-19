# Cupcake Photo Culling Library

A powerful and extensible photo culling library for photographers.

## Overview

Cupcake is a comprehensive photo culling library designed to help photographers efficiently select the best images from their photoshoots. It uses advanced analysis to evaluate images based on technical quality, composition, subject detection, and user preferences.

## Features

- **Comprehensive Format Support**: Handles standard and RAW formats from all major camera manufacturers
- **Intelligent Image Analysis**: Assesses focus, exposure, composition, and other technical aspects
- **Machine Learning Integration**: Plugin system supports a small locally-run LLM that learns user preferences
- **Flexible Selection System**: Streamlined workflow for selecting and rejecting images
- **Extensible Plugin Architecture**: Easily extend functionality through the plugin system

## Project Structure

```
cupcake/
├── requirements.txt          # Project dependencies
├── src/
│   ├── __init__.py           # Package initialization
│   ├── image_loader.py       # Image loading and metadata extraction
│   ├── image_formats.py      # Supported image format definitions
│   ├── analysis_engine.py    # Image quality analysis
│   ├── rating_system.py      # Rating algorithms and preference learning
│   ├── selection_manager.py  # Selection and organization system
│   ├── storage_manager.py    # Storage and export utilities
│   └── plugin_system.py      # Plugin framework
├── plugins/
│   ├── __init__.py           # Plugin package initialization
│   └── llm_style_predictor.py # ML-based style prediction plugin
└── tests/
    ├── __init__.py           # Test package initialization
    ├── test_image_loader.py  # Tests for image loader
    └── ...                   # Tests for other components
```

## Installation

1. Create a virtual environment:
   ```
   python -m venv cupcake
   ```

2. Activate the virtual environment:
   - On Windows: `cupcake\Scripts\activate`
   - On macOS/Linux: `source cupcake/bin/activate`

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

```python
from src.image_loader import ImageLoader
from src.analysis_engine import AnalysisEngine
from src.rating_system import RatingSystem
from src.selection_manager import SelectionManager
from src.storage_manager import StorageManager

# Initialize components
image_loader = ImageLoader()
analysis_engine = AnalysisEngine()
rating_system = RatingSystem()
selection_manager = SelectionManager()
storage_manager = StorageManager("./output")

# Load images
images = image_loader.load_from_directory("./photos")

# Analyze and rate
for path, image_data, metadata in images:
    image_id = metadata['filename']
    analysis = analysis_engine.analyze_image(image_data)
    rating = rating_system.calculate_score(analysis)
    
    # Auto-select based on rating
    if rating['overall_score'] > 80:
        selection_manager.mark_as_selected(image_id)
    else:
        selection_manager.mark_as_rejected(image_id)

# Export selected images
storage_manager.export_selected(selection_manager, "./selected_photos")
```

## Core Components

### Image Loader
Handles loading images from various sources and extracting metadata. Supports a wide range of image formats including RAW files from all major camera manufacturers.

### Analysis Engine
Analyzes images for technical quality metrics including focus sharpness, exposure, composition, and face detection.

### Rating System
Calculates overall quality scores based on analysis results. Can learn from user selections to adapt to individual preferences.

### Selection Manager
Manages the selection state of images, organizing them into collections and tracking selection history.

### Storage Manager
Handles exporting selected images and organizing them according to various criteria.

### Plugin System
Provides an extensible framework for adding new functionality. Includes hooks for all major processing steps.

## License

MIT License