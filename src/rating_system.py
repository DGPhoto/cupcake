# src/rating_system.py

from dataclasses import dataclass, field
import numpy as np
import json
import os
from typing import Dict, Any, List, Tuple, Optional, Set, Union
from datetime import datetime
import pickle
from enum import Enum

from .analysis_engine import ImageAnalysisResult


class RatingCategory(Enum):
    """Rating categories for photo evaluation"""
    EXCELLENT = 5
    GOOD = 4
    AVERAGE = 3
    BELOW_AVERAGE = 2
    REJECT = 1


@dataclass
class RatingProfile:
    """Defines a rating profile with customized weights and thresholds."""
    name: str
    description: str = "Custom rating profile"
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    
    # Weight factors (must sum to 1.0)
    technical_weight: float = 0.6
    composition_weight: float = 0.4
    
    # Technical sub-weights (must sum to 1.0)
    sharpness_weight: float = 0.35
    exposure_weight: float = 0.30
    contrast_weight: float = 0.25
    noise_weight: float = 0.10
    
    # Composition sub-weights (must sum to 1.0)
    rule_of_thirds_weight: float = 0.40
    symmetry_weight: float = 0.20
    subject_position_weight: float = 0.40
    
    # Minimum thresholds for rating categories
    excellent_threshold: float = 85.0
    good_threshold: float = 75.0
    average_threshold: float = 60.0
    below_average_threshold: float = 45.0
    
    # Learning parameters
    learning_rate: float = 0.1
    favorite_boost: float = 5.0
    reject_penalty: float = 5.0
    
    def validate(self) -> bool:
        """Validate that weights sum up to 1.0."""
        if not np.isclose(self.technical_weight + self.composition_weight, 1.0):
            return False
            
        if not np.isclose(self.sharpness_weight + self.exposure_weight + 
                        self.contrast_weight + self.noise_weight, 1.0):
            return False
            
        if not np.isclose(self.rule_of_thirds_weight + self.symmetry_weight + 
                        self.subject_position_weight, 1.0):
            return False
            
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary for serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'created_at': self.created_at.isoformat(),
            'modified_at': self.modified_at.isoformat(),
            'technical_weight': self.technical_weight,
            'composition_weight': self.composition_weight,
            'sharpness_weight': self.sharpness_weight,
            'exposure_weight': self.exposure_weight,
            'contrast_weight': self.contrast_weight,
            'noise_weight': self.noise_weight,
            'rule_of_thirds_weight': self.rule_of_thirds_weight,
            'symmetry_weight': self.symmetry_weight,
            'subject_position_weight': self.subject_position_weight,
            'excellent_threshold': self.excellent_threshold,
            'good_threshold': self.good_threshold,
            'average_threshold': self.average_threshold,
            'below_average_threshold': self.below_average_threshold,
            'learning_rate': self.learning_rate,
            'favorite_boost': self.favorite_boost,
            'reject_penalty': self.reject_penalty
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RatingProfile':
        """Create profile from dictionary."""
        profile = cls(name=data['name'])
        
        for key, value in data.items():
            if key in ('created_at', 'modified_at'):
                setattr(profile, key, datetime.fromisoformat(value))
            elif hasattr(profile, key):
                setattr(profile, key, value)
        
        return profile


@dataclass
class ImageRating:
    """Holds rating information for a single image."""
    image_id: str  # Unique identifier (typically the file path)
    rating_category: RatingCategory
    overall_score: float
    technical_score: float
    composition_score: float
    
    # Individual scores
    sharpness_score: float = 0.0
    exposure_score: float = 0.0
    contrast_score: float = 0.0
    noise_score: float = 0.0
    rule_of_thirds_score: float = 0.0
    symmetry_score: float = 0.0
    subject_position_score: float = 0.0
    
    # User feedback
    user_rating: Optional[RatingCategory] = None
    is_favorite: bool = False
    is_rejected: bool = False
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    profile_name: str = "default"
    
    def get_rating_description(self) -> str:
        """Get human-readable description of the rating."""
        return {
            RatingCategory.EXCELLENT: "Excellent - Top quality image",
            RatingCategory.GOOD: "Good - High quality image",
            RatingCategory.AVERAGE: "Average - Acceptable quality",
            RatingCategory.BELOW_AVERAGE: "Below average - Potential issues",
            RatingCategory.REJECT: "Reject - Significant issues"
        }.get(self.rating_category, "Unknown rating")
    
    def get_primary_issue(self) -> Optional[str]:
        """Identify the primary technical issue with the image."""
        if self.rating_category in (RatingCategory.EXCELLENT, RatingCategory.GOOD):
            return None
            
        scores = {
            "sharpness": self.sharpness_score,
            "exposure": self.exposure_score,
            "contrast": self.contrast_score,
            "noise": self.noise_score
        }
        
        worst_aspect = min(scores.items(), key=lambda x: x[1])
        
        if worst_aspect[1] < 50:
            issue_descriptions = {
                "sharpness": "Image lacks sharpness/focus",
                "exposure": "Image has exposure issues",
                "contrast": "Image lacks adequate contrast",
                "noise": "Image has excessive noise"
            }
            return issue_descriptions.get(worst_aspect[0])
            
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert rating to dictionary for serialization."""
        return {
            'image_id': self.image_id,
            'rating_category': self.rating_category.value,
            'overall_score': self.overall_score,
            'technical_score': self.technical_score,
            'composition_score': self.composition_score,
            'sharpness_score': self.sharpness_score,
            'exposure_score': self.exposure_score,
            'contrast_score': self.contrast_score,
            'noise_score': self.noise_score,
            'rule_of_thirds_score': self.rule_of_thirds_score,
            'symmetry_score': self.symmetry_score,
            'subject_position_score': self.subject_position_score,
            'user_rating': self.user_rating.value if self.user_rating else None,
            'is_favorite': self.is_favorite,
            'is_rejected': self.is_rejected,
            'timestamp': self.timestamp.isoformat(),
            'profile_name': self.profile_name
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ImageRating':
        """Create rating from dictionary."""
        # Convert enum values back to enums
        rating_category = RatingCategory(data['rating_category'])
        
        user_rating = None
        if data.get('user_rating') is not None:
            user_rating = RatingCategory(data['user_rating'])
        
        # Create the rating object
        rating = cls(
            image_id=data['image_id'],
            rating_category=rating_category,
            overall_score=data['overall_score'],
            technical_score=data['technical_score'],
            composition_score=data['composition_score'],
            sharpness_score=data.get('sharpness_score', 0.0),
            exposure_score=data.get('exposure_score', 0.0),
            contrast_score=data.get('contrast_score', 0.0),
            noise_score=data.get('noise_score', 0.0),
            rule_of_thirds_score=data.get('rule_of_thirds_score', 0.0),
            symmetry_score=data.get('symmetry_score', 0.0),
            subject_position_score=data.get('subject_position_score', 0.0),
            user_rating=user_rating,
            is_favorite=data.get('is_favorite', False),
            is_rejected=data.get('is_rejected', False),
            profile_name=data.get('profile_name', 'default')
        )
        
        # Set timestamp if available
        if 'timestamp' in data:
            rating.timestamp = datetime.fromisoformat(data['timestamp'])
            
        return rating


class UserPreferenceModel:
    """
    A simple machine learning model to learn user preferences.
    This can be replaced with a more sophisticated model in the future.
    """
    
    def __init__(self):
        # Adjustment factors learned from user feedback
        self.sharpness_factor = 1.0
        self.exposure_factor = 1.0
        self.contrast_factor = 1.0
        self.noise_factor = 1.0
        self.rule_of_thirds_factor = 1.0
        self.symmetry_factor = 1.0
        self.subject_position_factor = 1.0
        
        # History of training samples
        self.training_history = []
        
    def update_from_feedback(self, analysis_result: ImageAnalysisResult, 
                           user_rating: RatingCategory, learning_rate: float = 0.1):
        """
        Update model based on user feedback.
        
        Args:
            analysis_result: Analysis results for an image
            user_rating: User's rating of the image
            learning_rate: How quickly the model should adapt
        """
        # Convert user rating to a scaled value between 0 and 1
        scaled_rating = (user_rating.value - 1) / 4.0  # Maps 1-5 to 0-1
        
        # Store this sample for later analysis
        self.training_history.append((analysis_result, user_rating))
        
        # Determine how much to adjust factors based on correlation between
        # current scores and user rating
        
        # Simple adjustment approach:
        # If rating is high but score is low, increase factor
        # If rating is low but score is high, decrease factor
        
        # For sharpness
        expected_sharpness = analysis_result.sharpness_score / 100.0
        if scaled_rating > expected_sharpness:
            # User likes this image more than we predicted based on sharpness
            # Reduce the importance of sharpness
            self.sharpness_factor -= learning_rate * (expected_sharpness - scaled_rating)
        else:
            # User likes this image less than we predicted based on sharpness
            # Increase the importance of sharpness
            self.sharpness_factor += learning_rate * (scaled_rating - expected_sharpness)
            
        # Apply similar logic to other factors
        self._update_factor('exposure_factor', analysis_result.exposure_score / 100.0, scaled_rating, learning_rate)
        self._update_factor('contrast_factor', analysis_result.contrast_score / 100.0, scaled_rating, learning_rate)
        self._update_factor('noise_factor', analysis_result.noise_score / 100.0, scaled_rating, learning_rate)
        self._update_factor('rule_of_thirds_factor', analysis_result.rule_of_thirds_score / 100.0, scaled_rating, learning_rate)
        self._update_factor('symmetry_factor', analysis_result.symmetry_score / 100.0, scaled_rating, learning_rate)
        self._update_factor('subject_position_factor', analysis_result.subject_position_score / 100.0, scaled_rating, learning_rate)
        
        # Normalize factors to prevent extreme values
        self._normalize_factors()
    
    def _update_factor(self, factor_name: str, score: float, scaled_rating: float, learning_rate: float):
        """Helper method to update a single factor."""
        factor = getattr(self, factor_name)
        
        if scaled_rating > score:
            # User rates higher than score suggests, so reduce this factor's importance
            factor -= learning_rate * (score - scaled_rating)
        else:
            # User rates lower than score suggests, so increase this factor's importance
            factor += learning_rate * (scaled_rating - score)
            
        # Ensure factor stays within reasonable bounds
        factor = max(0.5, min(2.0, factor))
        setattr(self, factor_name, factor)
    
    def _normalize_factors(self):
        """Keep factors within reasonable bounds."""
        for factor_name in [
            'sharpness_factor', 'exposure_factor', 'contrast_factor', 'noise_factor',
            'rule_of_thirds_factor', 'symmetry_factor', 'subject_position_factor'
        ]:
            current_value = getattr(self, factor_name)
            normalized_value = max(0.5, min(2.0, current_value))
            setattr(self, factor_name, normalized_value)
    
    def apply_preferences(self, analysis_result: ImageAnalysisResult) -> ImageAnalysisResult:
        """
        Apply learned preferences to analysis results.
        
        Args:
            analysis_result: Original analysis results
            
        Returns:
            Modified analysis results based on learned preferences
        """
        # Create a copy to avoid modifying the original
        result_copy = ImageAnalysisResult()
        
        # Copy all attributes from the original result
        for attr_name in dir(analysis_result):
            if not attr_name.startswith('_') and hasattr(result_copy, attr_name):
                setattr(result_copy, attr_name, getattr(analysis_result, attr_name))
        
        # Apply learned factors to adjust scores
        result_copy.sharpness_score = min(100, analysis_result.sharpness_score * self.sharpness_factor)
        result_copy.exposure_score = min(100, analysis_result.exposure_score * self.exposure_factor)
        result_copy.contrast_score = min(100, analysis_result.contrast_score * self.contrast_factor)
        result_copy.noise_score = min(100, analysis_result.noise_score * self.noise_factor)
        result_copy.rule_of_thirds_score = min(100, analysis_result.rule_of_thirds_score * self.rule_of_thirds_factor)
        result_copy.symmetry_score = min(100, analysis_result.symmetry_score * self.symmetry_factor)
        result_copy.subject_position_score = min(100, analysis_result.subject_position_score * self.subject_position_factor)
        
        # Recalculate overall scores
        # (Note: this is simplified - in reality, would need to match the same algorithm as AnalysisEngine)
        result_copy.overall_technical_score = (
            0.35 * result_copy.sharpness_score + 
            0.30 * result_copy.exposure_score + 
            0.25 * result_copy.contrast_score + 
            0.10 * result_copy.noise_score
        )
        
        result_copy.overall_composition_score = (
            0.40 * result_copy.rule_of_thirds_score + 
            0.20 * result_copy.symmetry_score + 
            0.40 * result_copy.subject_position_score
        )
        
        result_copy.overall_score = (
            0.60 * result_copy.overall_technical_score + 
            0.40 * result_copy.overall_composition_score
        )
        
        return result_copy
        
    def save(self, file_path: str):
        """Save model to file."""
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, file_path: str) -> 'UserPreferenceModel':
        """Load model from file."""
        with open(file_path, 'rb') as f:
            return pickle.load(f)


class RatingSystem:
    """
    Rates images based on analysis results and user preferences.
    """
    
    def __init__(self, profiles_dir: Optional[str] = None):
        """
        Initialize rating system.
        
        Args:
            profiles_dir: Directory to store rating profiles
        """
        self.profiles: Dict[str, RatingProfile] = {}
        self.profiles_dir = profiles_dir
        
        # Initialize default profile
        self.profiles['default'] = RatingProfile(name="default", description="Default rating profile")
        
        # Initialize user preference model
        self.preference_model = UserPreferenceModel()
        
        # Load existing profiles if directory provided
        if profiles_dir and os.path.exists(profiles_dir):
            self.load_profiles()
    
    def load_profiles(self):
        """Load profiles from profiles directory."""
        if not self.profiles_dir or not os.path.exists(self.profiles_dir):
            return
            
        for filename in os.listdir(self.profiles_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(self.profiles_dir, filename), 'r') as f:
                        profile_data = json.load(f)
                        profile = RatingProfile.from_dict(profile_data)
                        self.profiles[profile.name] = profile
                except Exception as e:
                    print(f"Error loading profile {filename}: {e}")
    
    def save_profiles(self):
        """Save profiles to profiles directory."""
        if not self.profiles_dir:
            return
            
        os.makedirs(self.profiles_dir, exist_ok=True)
        
        for name, profile in self.profiles.items():
            try:
                with open(os.path.join(self.profiles_dir, f"{name}.json"), 'w') as f:
                    json.dump(profile.to_dict(), f, indent=2)
            except Exception as e:
                print(f"Error saving profile {name}: {e}")
    
    def add_profile(self, profile: RatingProfile):
        """Add a new rating profile."""
        if not profile.validate():
            raise ValueError("Profile weights must sum to 1.0")
            
        self.profiles[profile.name] = profile
        self.save_profiles()
    
    def get_profile(self, name: str) -> RatingProfile:
        """Get a rating profile by name."""
        return self.profiles.get(name, self.profiles['default'])
    
    def rate_image(self, analysis_result: ImageAnalysisResult, 
                 image_id: str, profile_name: str = 'default',
                 apply_preferences: bool = True) -> ImageRating:
        """
        Rate an image based on analysis results and profile.
        
        Args:
            analysis_result: Analysis results for the image
            image_id: Unique identifier for the image
            profile_name: Name of the rating profile to use
            apply_preferences: Whether to apply learned user preferences
            
        Returns:
            ImageRating object with rating details
        """
        # Get the rating profile
        profile = self.get_profile(profile_name)
        
        # Apply user preferences if requested
        if apply_preferences:
            analysis_result = self.preference_model.apply_preferences(analysis_result)
        
        # Calculate technical score
        technical_score = (
            profile.sharpness_weight * analysis_result.sharpness_score +
            profile.exposure_weight * analysis_result.exposure_score +
            profile.contrast_weight * analysis_result.contrast_score +
            profile.noise_weight * analysis_result.noise_score
        )
        
        # Calculate composition score
        composition_score = (
            profile.rule_of_thirds_weight * analysis_result.rule_of_thirds_score +
            profile.symmetry_weight * analysis_result.symmetry_score +
            profile.subject_position_weight * analysis_result.subject_position_score
        )
        
        # Calculate overall score
        overall_score = (
            profile.technical_weight * technical_score +
            profile.composition_weight * composition_score
        )
        
        # Determine rating category
        rating_category = self._get_rating_category(overall_score, profile)
        
        # Create rating object
        rating = ImageRating(
            image_id=image_id,
            rating_category=rating_category,
            overall_score=overall_score,
            technical_score=technical_score,
            composition_score=composition_score,
            sharpness_score=analysis_result.sharpness_score,
            exposure_score=analysis_result.exposure_score,
            contrast_score=analysis_result.contrast_score,
            noise_score=analysis_result.noise_score,
            rule_of_thirds_score=analysis_result.rule_of_thirds_score,
            symmetry_score=analysis_result.symmetry_score,
            subject_position_score=analysis_result.subject_position_score,
            profile_name=profile_name
        )
        
        return rating
    
    def _get_rating_category(self, overall_score: float, profile: RatingProfile) -> RatingCategory:
        """Map overall score to rating category."""
        if overall_score >= profile.excellent_threshold:
            return RatingCategory.EXCELLENT
        elif overall_score >= profile.good_threshold:
            return RatingCategory.GOOD
        elif overall_score >= profile.average_threshold:
            return RatingCategory.AVERAGE
        elif overall_score >= profile.below_average_threshold:
            return RatingCategory.BELOW_AVERAGE
        else:
            return RatingCategory.REJECT
    
    def provide_feedback(self, rating: ImageRating, 
                       user_rating: Optional[RatingCategory] = None,
                       is_favorite: Optional[bool] = None,
                       is_rejected: Optional[bool] = None) -> ImageRating:
        """
        Provide feedback on an image rating to improve future ratings.
        
        Args:
            rating: The existing rating to update
            user_rating: User's own rating category
            is_favorite: Whether the image is marked as a favorite
            is_rejected: Whether the image is rejected
            
        Returns:
            Updated rating
        """
        # Update rating with user feedback
        if user_rating is not None:
            rating.user_rating = user_rating
            
        if is_favorite is not None:
            rating.is_favorite = is_favorite
            
        if is_rejected is not None:
            rating.is_rejected = is_rejected
            
        # TODO: Use this feedback to update the preference model
        # This would require storing the original analysis results
            
        return rating
    
    def batch_rate_images(self, analysis_results: Dict[str, ImageAnalysisResult], 
                        profile_name: str = 'default',
                        apply_preferences: bool = True) -> Dict[str, ImageRating]:
        """
        Rate multiple images in batch.
        
        Args:
            analysis_results: Dictionary mapping image IDs to analysis results
            profile_name: Name of rating profile to use
            apply_preferences: Whether to apply learned user preferences
            
        Returns:
            Dictionary mapping image IDs to ratings
        """
        ratings = {}
        
        for image_id, result in analysis_results.items():
            ratings[image_id] = self.rate_image(
                result, image_id, profile_name, apply_preferences
            )
            
        return ratings
    
    def get_rating_distribution(self, ratings: List[ImageRating]) -> Dict[RatingCategory, int]:
        """Get distribution of ratings by category."""
        distribution = {category: 0 for category in RatingCategory}
        
        for rating in ratings:
            distribution[rating.rating_category] += 1
            
        return distribution
    
    def update_preferences_from_ratings(self, analysis_results: Dict[str, ImageAnalysisResult], 
                                     ratings: Dict[str, ImageRating],
                                     learning_rate: Optional[float] = None):
        """
        Update user preference model based on a batch of ratings with user feedback.
        
        Args:
            analysis_results: Dictionary mapping image IDs to analysis results
            ratings: Dictionary mapping image IDs to ratings with user feedback
            learning_rate: Optional custom learning rate
        """
        # Get the profile to use for learning rate
        profile = self.get_profile(next(iter(ratings.values())).profile_name)
        lr = learning_rate if learning_rate is not None else profile.learning_rate
        
        # Process each rating with user feedback
        for image_id, rating in ratings.items():
            if rating.user_rating is not None and image_id in analysis_results:
                # Update based on explicit user rating
                self.preference_model.update_from_feedback(
                    analysis_results[image_id], rating.user_rating, lr
                )
            elif rating.is_favorite and image_id in analysis_results:
                # Favorites get an implicit boost
                boost_value = min(5, int(rating.rating_category.value) + 1)
                boosted_rating = RatingCategory(boost_value)
                self.preference_model.update_from_feedback(
                    analysis_results[image_id], boosted_rating, lr * 0.5  # Half learning rate for implicit feedback
                )
            elif rating.is_rejected and image_id in analysis_results:
                # Rejected images get an implicit penalty
                penalty_value = max(1, int(rating.rating_category.value) - 1)
                penalized_rating = RatingCategory(penalty_value)
                self.preference_model.update_from_feedback(
                    analysis_results[image_id], penalized_rating, lr * 0.5  # Half learning rate for implicit feedback
                )
    
    def save_preference_model(self, file_path: str):
        """Save the user preference model to a file."""
        self.preference_model.save(file_path)
    
    def load_preference_model(self, file_path: str):
        """Load the user preference model from a file."""
        if os.path.exists(file_path):
            self.preference_model = UserPreferenceModel.load(file_path)