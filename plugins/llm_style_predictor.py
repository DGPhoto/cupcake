# plugins/llm_style_predictor.py
# Cupcake Photo Culling Library Plugin

import os
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import tensorflow as tf
from PIL import Image
import logging

from src.plugin_system import CupcakePlugin, PluginType, PluginHook

class LLMStylePredictorPlugin(CupcakePlugin):
    """
    LLM Style Predictor - A machine learning plugin for Cupcake Photo Culling Library
    
    This plugin uses a small locally-run machine learning model to predict the 
    photographic style of images and learn user preferences over time.
    """
    
    plugin_name = "LLM Style Predictor"
    plugin_type = PluginType.ML
    plugin_description = "Predicts photographic styles and learns user preferences"
    plugin_version = "0.1.0"
    plugin_author = "Cupcake Team"
    plugin_hooks = [
        PluginHook.POST_ANALYSIS,
        PluginHook.LEARN_PREFERENCES,
        PluginHook.STARTUP,
        PluginHook.SHUTDOWN
    ]
    
    # Photographic styles the model can recognize
    STYLES = [
        "Portrait", "Landscape", "Street", "Wildlife", "Macro",
        "Architecture", "Documentary", "Abstract", "Night", "Sport"
    ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model = None
        self.preference_model = None
        self.is_model_loaded = False
        self.user_preferences = {}
        
        # Set default configuration
        default_config = {
            "model_path": "models/style_predictor.keras",  # Added .keras extension
            "preference_model_path": "models/preference_model.json",
            "prediction_threshold": 0.6,
            "learning_rate": 0.1,
            "enable_style_prediction": True,
            "enable_preference_learning": True
        }
        
        # Update with provided config
        if config:
            default_config.update(config)
        
        self.config = default_config
    
    def initialize(self) -> bool:
        """
        Initialize the plugin. Called when the plugin is loaded.
        """
        self.logger.info(f"Initializing {self.plugin_name}")
        
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config["model_path"]), exist_ok=True)
        os.makedirs(os.path.dirname(self.config["preference_model_path"]), exist_ok=True)
        
        # Try to load models
        return self._load_models()
    
    def startup(self) -> bool:
        """
        Called when the application starts.
        """
        self.logger.info(f"Starting up {self.plugin_name}")
        
        # Nothing special to do here since we load models in initialize
        return True
    
    def shutdown(self) -> bool:
        """
        Shutdown the plugin. Called when the plugin is unloaded.
        """
        self.logger.info(f"Shutting down {self.plugin_name}")
        
        # Save user preferences
        self._save_user_preferences()
        
        # Clear models from memory
        if self.model:
            del self.model
        
        if self.preference_model:
            del self.preference_model
        
        self.is_model_loaded = False
        
        return True
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get the configuration schema for this plugin.
        """
        return {
            "model_path": {
                "type": "string",
                "default": "models/style_predictor.keras",
                "description": "Path to the style prediction model"
            },
            "preference_model_path": {
                "type": "string",
                "default": "models/preference_model.json",
                "description": "Path to the user preference model"
            },
            "prediction_threshold": {
                "type": "float",
                "default": 0.6,
                "description": "Threshold for style prediction confidence"
            },
            "learning_rate": {
                "type": "float",
                "default": 0.1,
                "description": "Learning rate for preference updates"
            },
            "enable_style_prediction": {
                "type": "boolean",
                "default": True,
                "description": "Enable style prediction"
            },
            "enable_preference_learning": {
                "type": "boolean",
                "default": True,
                "description": "Enable learning from user selections"
            }
        }
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate the configuration for this plugin.
        """
        if "prediction_threshold" in config:
            threshold = config["prediction_threshold"]
            if not (0.0 <= threshold <= 1.0):
                return False, "prediction_threshold must be between 0.0 and 1.0"
        
        if "learning_rate" in config:
            rate = config["learning_rate"]
            if not (0.0 <= rate <= 1.0):
                return False, "learning_rate must be between 0.0 and 1.0"
        
        return True, None
    
    def post_analysis(self, image_data: np.ndarray, analysis_results: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Called after image analysis to predict the style.
        
        Args:
            image_data: Image data as numpy array
            analysis_results: Analysis results from AnalysisEngine
            metadata: Image metadata
            
        Returns:
            Dictionary with style predictions to add to analysis results
        """
        if not self.config["enable_style_prediction"] or not self.is_model_loaded:
            return {}
        
        try:
            # Preprocess image for the model
            processed_image = self._preprocess_image(image_data)
            
            # Predict style
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Convert predictions to dictionary
            style_scores = {style: float(score) for style, score in zip(self.STYLES, predictions[0])}
            
            # Get dominant styles (above threshold)
            threshold = self.config["prediction_threshold"]
            dominant_styles = [style for style, score in style_scores.items() if score >= threshold]
            
            # Calculate preference score based on dominant styles
            preference_score = self._calculate_preference_score(style_scores)
            
            # Return style information
            return {
                "style_predictions": style_scores,
                "dominant_styles": dominant_styles,
                "preference_score": preference_score
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting style: {e}")
            return {}
    
    def learn_preferences(self, image_id: str, selected: bool, analysis_results: Dict[str, Any]) -> bool:
        """
        Learn from user selections to update the preference model.
        
        Args:
            image_id: ID of the image
            selected: Whether the image was selected by the user
            analysis_results: Analysis results for the image
            
        Returns:
            True if learning was successful
        """
        if not self.config["enable_preference_learning"] or not self.is_model_loaded:
            return False
        
        try:
            # Skip if no style predictions available
            if "style_predictions" not in analysis_results:
                return False
            
            style_scores = analysis_results["style_predictions"]
            
            # Learning rate
            learning_rate = self.config["learning_rate"]
            
            # Update user preferences
            for style, score in style_scores.items():
                if style not in self.user_preferences:
                    self.user_preferences[style] = 0.5  # Initial neutral preference
                
                # Increase preference if selected, decrease if rejected
                if selected:
                    # Increase preference proportionally to the style score
                    self.user_preferences[style] += learning_rate * score * (1 - self.user_preferences[style])
                else:
                    # Decrease preference proportionally to the style score
                    self.user_preferences[style] -= learning_rate * score * self.user_preferences[style]
                
                # Ensure preference stays between 0 and 1
                self.user_preferences[style] = max(0.0, min(1.0, self.user_preferences[style]))
            
            # Save preferences periodically
            self._save_user_preferences()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error learning preferences: {e}")
            return False
    
    def _preprocess_image(self, image_data: np.ndarray) -> np.ndarray:
        """
        Preprocess image for the model.
        
        Args:
            image_data: Original image data
            
        Returns:
            Processed image data ready for the model
        """
        # Resize to model input size
        img = Image.fromarray(image_data)
        img = img.resize((224, 224))
        img_array = np.array(img)
        
        # Convert to float and normalize
        img_array = img_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def _calculate_preference_score(self, style_scores: Dict[str, float]) -> float:
        """
        Calculate a preference score based on style predictions and user preferences.
        
        Args:
            style_scores: Style prediction scores
            
        Returns:
            Preference score (0.0 to 1.0)
        """
        if not self.user_preferences:
            return 0.5  # Neutral score if no preferences yet
        
        # Calculate weighted sum of style scores based on user preferences
        weighted_sum = 0.0
        total_weight = 0.0
        
        for style, score in style_scores.items():
            if style in self.user_preferences:
                preference = self.user_preferences[style]
                weighted_sum += score * preference
                total_weight += score
        
        # Return normalized preference score
        if total_weight > 0:
            return min(1.0, max(0.0, weighted_sum / total_weight))
        else:
            return 0.5
    # Nel metodo _load_models all'interno della classe LLMStylePredictorPlugin

    def _load_models(self) -> bool:
        """
        Load the style prediction and preference models with improved error handling.
        
        Returns:
            True if models loaded successfully or fallback created
        """
        try:
            # Check if style predictor model exists
            model_path = self.config["model_path"]
            if os.path.exists(model_path):
                self.logger.info(f"Loading style predictor model from {model_path}")
                
                # Attempt to load the model
                try:
                    self.model = tf.keras.models.load_model(model_path)
                    self.logger.info("Successfully loaded the model")
                except Exception as e:
                    self.logger.warning(f"Error loading model: {e}. Creating a dummy model instead.")
                    self._create_dummy_model()
            else:
                self.logger.warning(f"Style predictor model not found at {model_path}, creating a dummy model")
                self._create_dummy_model()
            
            # Load user preferences
            self._load_user_preferences()
            
            self.is_model_loaded = True
            return True
            
        except Exception as e:
            self.logger.error(f"Error in model loading: {e}")
            
            # Create an even simpler fallback model since the full dummy model creation failed
            try:
                self.logger.info("Creating a minimal fallback model")
                self._create_minimal_fallback_model()
                self._load_user_preferences()
                self.is_model_loaded = True
                return True
            except Exception as fallback_error:
                self.logger.error(f"Fatal error creating fallback model: {fallback_error}")
                self.is_model_loaded = False
                return False

    def _create_minimal_fallback_model(self):
        """Create an extremely simple model as a last resort fallback."""
        # Create the simplest possible model
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        outputs = tf.keras.layers.Dense(len(self.STYLES), activation='sigmoid')(x)
        
        self.model = tf.keras.Model(inputs, outputs)
        
        # Compile with basic settings
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.logger.info("Created minimal fallback model")
    
    def _create_dummy_model(self):
        """Create a simple dummy model for demonstration purposes."""
        # This is just a placeholder - in a real implementation, you would
        # either download a pre-trained model or train one with real data
        
        # Simple CNN model for image style classification
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(len(self.STYLES), activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        # Save the dummy model with the proper extension
        model_path = self.config["model_path"]
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        try:
            model.save(model_path)
            self.logger.info(f"Created and saved dummy model to {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to save dummy model: {e}")
            # As a fallback, don't save the model but keep it in memory
            self.logger.info("Using in-memory model instead")
    
    def _load_user_preferences(self):
        """Load user preferences from a file."""
        pref_path = self.config["preference_model_path"]
        
        if os.path.exists(pref_path):
            try:
                import json
                with open(pref_path, 'r') as f:
                    self.user_preferences = json.load(f)
                self.logger.info(f"Loaded user preferences from {pref_path}")
            except Exception as e:
                self.logger.error(f"Error loading user preferences: {e}")
                self.user_preferences = {style: 0.5 for style in self.STYLES}
        else:
            # Initialize with neutral preferences
            self.user_preferences = {style: 0.5 for style in self.STYLES}
    
    def _save_user_preferences(self):
        """Save user preferences to a file."""
        pref_path = self.config["preference_model_path"]
        
        try:
            import json
            os.makedirs(os.path.dirname(pref_path), exist_ok=True)
            with open(pref_path, 'w') as f:
                json.dump(self.user_preferences, f, indent=2)
            self.logger.debug(f"Saved user preferences to {pref_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving user preferences: {e}")
            return False
    
    def get_style_description(self, style: str) -> str:
        """
        Get a description of a photographic style.
        
        Args:
            style: Style name
            
        Returns:
            Description of the style
        """
        descriptions = {
            "Portrait": "Focuses on capturing the personality and essence of a person or group through posed or candid shots.",
            "Landscape": "Captures the beauty of natural scenery, emphasizing the grandeur of the outdoors.",
            "Street": "Documents everyday life in public places, often in urban environments with a raw, unposed aesthetic.",
            "Wildlife": "Captures animals in their natural habitats, often requiring patience and specialized equipment.",
            "Macro": "Extreme close-up photography, usually of very small subjects, showcasing details not visible to the naked eye.",
            "Architecture": "Focuses on buildings and structures, emphasizing their design, form, and sometimes historical significance.",
            "Documentary": "Tells a story or conveys information about events, social issues, or cultural phenomena.",
            "Abstract": "Focuses on shapes, colors, and textures rather than recognizable subjects, often creating a mood or impression.",
            "Night": "Photography in low-light conditions, often featuring artificial lighting, stars, or long exposures.",
            "Sport": "Captures the action and drama of athletic events, requiring fast shutter speeds and precise timing."
        }
        
        return descriptions.get(style, "No description available")
    
    def get_user_preference_report(self) -> Dict[str, Any]:
        """
        Generate a report on user preferences.
        
        Returns:
            Dictionary with preference information
        """
        if not self.user_preferences:
            return {"message": "No preference data available"}
        
        # Sort styles by preference
        sorted_preferences = sorted(
            self.user_preferences.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Format preferences
        formatted_preferences = [
            {
                "style": style,
                "preference": preference,
                "description": self.get_style_description(style)
            }
            for style, preference in sorted_preferences
        ]
        
        # Calculate summary statistics
        top_styles = [item["style"] for item in formatted_preferences[:3]]
        bottom_styles = [item["style"] for item in formatted_preferences[-3:]]
        
        return {
            "preferences": formatted_preferences,
            "top_styles": top_styles,
            "bottom_styles": bottom_styles,
            "preference_strength": sum(p for _, p in sorted_preferences) / len(sorted_preferences)
        }