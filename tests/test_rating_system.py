# tests/test_rating_system.py

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import tempfile
import io
import contextlib

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.image_loader import ImageLoader
from src.analysis_engine import AnalysisEngine, ImageAnalysisResult
from src.rating_system import RatingSystem, RatingProfile, ImageRating, RatingCategory, UserPreferenceModel

# Context manager to suppress stdout (for "File format not recognized" messages)
@contextlib.contextmanager
def suppress_stdout():
    # Save original stdout
    original_stdout = sys.stdout
    # Redirect stdout to a null IO stream
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        # Restore original stdout
        sys.stdout = original_stdout

def plot_rating_distribution(ratings: Dict[str, ImageRating], title: str = "Rating Distribution"):
    """Plot the distribution of ratings."""
    categories = [cat.name for cat in RatingCategory]
    counts = [0] * len(categories)
    
    for rating in ratings.values():
        counts[rating.rating_category.value - 1] += 1
    
    # Create a bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, counts, color=['red', 'orange', 'yellow', 'lightgreen', 'darkgreen'])
    
    # Add count labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom')
    
    plt.title(title)
    plt.ylabel('Number of Images')
    plt.xlabel('Rating Category')
    plt.tight_layout()
    plt.show()

def plot_score_comparison(ratings_before: Dict[str, ImageRating], 
                         ratings_after: Dict[str, ImageRating],
                         title: str = "Before vs After Learning"):
    """Plot a comparison of scores before and after learning."""
    # Extract image IDs and ensure they're the same in both dicts
    common_ids = set(ratings_before.keys()).intersection(set(ratings_after.keys()))
    
    # For each common image, get both scores
    scores_before = [ratings_before[img_id].overall_score for img_id in common_ids]
    scores_after = [ratings_after[img_id].overall_score for img_id in common_ids]
    
    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(scores_before, scores_after, alpha=0.7)
    
    # Add a diagonal line for reference
    min_score = min(min(scores_before), min(scores_after))
    max_score = max(max(scores_before), max(scores_after))
    plt.plot([min_score, max_score], [min_score, max_score], 'k--', alpha=0.5)
    
    # Add labels
    plt.xlabel("Score Before Learning")
    plt.ylabel("Score After Learning")
    plt.title(title)
    
    # Equal aspect ratio
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_learning_factors(preference_model: UserPreferenceModel, title: str = "Learned Preference Factors"):
    """Plot the learned preference factors."""
    factor_names = [
        'Sharpness', 'Exposure', 'Contrast', 'Noise',
        'Rule of Thirds', 'Symmetry', 'Subject Position'
    ]
    
    factor_values = [
        preference_model.sharpness_factor,
        preference_model.exposure_factor,
        preference_model.contrast_factor,
        preference_model.noise_factor,
        preference_model.rule_of_thirds_factor,
        preference_model.symmetry_factor,
        preference_model.subject_position_factor
    ]
    
    # Create a horizontal bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.barh(factor_names, factor_values, color='steelblue')
    
    # Add a vertical line at 1.0 to show baseline
    plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.7)
    
    # Add value labels to the right of each bar
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.05, bar.get_y() + bar.get_height()/2,
                 f'{width:.2f}', ha='left', va='center')
    
    plt.title(title)
    plt.xlabel('Factor Value (1.0 = Baseline)')
    plt.xlim(0, 2.0)  # Set x-axis limits
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()

def test_basic_rating():
    """Test basic rating functionality."""
    print("\n=== Testing Basic Rating Functionality ===")
    
    # Initialize components
    image_loader = ImageLoader()
    analysis_engine = AnalysisEngine()
    rating_system = RatingSystem()
    
    # Request a test image
    test_image_path = input("\nEnter path to a test image (or 'skip' to skip): ")
    
    if test_image_path.lower() == 'skip':
        print("Skipping single image test.")
        return True
    
    if not os.path.exists(test_image_path):
        print(f"File doesn't exist: {test_image_path}")
        return False
    
    try:
        # Load and analyze image - suppress the "File format not recognized" message
        print(f"Loading and analyzing image: {test_image_path}")
        with suppress_stdout():
            image_data, metadata = image_loader.load_from_path(test_image_path)
            analysis_result = analysis_engine.analyze_image(image_data, metadata)
        
        # Rate image with default profile
        image_id = os.path.basename(test_image_path)
        rating = rating_system.rate_image(analysis_result, image_id)
        
        # Print rating information
        print("\nRating Results:")
        print(f"Image: {image_id}")
        print(f"Rating Category: {rating.rating_category.name} ({rating.get_rating_description()})")
        print(f"Overall Score: {rating.overall_score:.1f}")
        print(f"Technical Score: {rating.technical_score:.1f}")
        print(f"Composition Score: {rating.composition_score:.1f}")
        
        # Check for primary issues
        primary_issue = rating.get_primary_issue()
        if primary_issue:
            print(f"Primary Issue: {primary_issue}")
        
        return True
    except Exception as e:
        print(f"Error rating image: {e}")
        return False

def test_custom_profile():
    """Test rating with a custom profile."""
    print("\n=== Testing Custom Rating Profile ===")
    
    # Create a custom rating profile
    custom_profile = RatingProfile(
        name="portrait",
        description="Profile optimized for portrait photography",
        technical_weight=0.5,
        composition_weight=0.5,
        sharpness_weight=0.4,
        exposure_weight=0.3,
        contrast_weight=0.2,
        noise_weight=0.1,
        rule_of_thirds_weight=0.2,
        symmetry_weight=0.3,
        subject_position_weight=0.5,
        excellent_threshold=80.0,  # Slightly more lenient
        good_threshold=70.0,
        average_threshold=55.0,
        below_average_threshold=40.0
    )
    
    # Initialize components
    image_loader = ImageLoader()
    analysis_engine = AnalysisEngine()
    rating_system = RatingSystem()
    
    # Add custom profile
    try:
        rating_system.add_profile(custom_profile)
        print(f"Created custom profile: {custom_profile.name}")
    except ValueError as e:
        print(f"Error creating profile: {e}")
        return False
    
    # Request a test image
    test_image_path = input("\nEnter path to a test image (or 'skip' to skip): ")
    
    if test_image_path.lower() == 'skip':
        print("Skipping custom profile test.")
        return True
    
    if not os.path.exists(test_image_path):
        print(f"File doesn't exist: {test_image_path}")
        return False
    
    try:
        # Load and analyze image
        print(f"Loading and analyzing image: {test_image_path}")
        with suppress_stdout():
            image_data, metadata = image_loader.load_from_path(test_image_path)
            analysis_result = analysis_engine.analyze_image(image_data, metadata)
        
        image_id = os.path.basename(test_image_path)
        
        # Rate image with both default and custom profiles
        default_rating = rating_system.rate_image(analysis_result, image_id, profile_name='default')
        custom_rating = rating_system.rate_image(analysis_result, image_id, profile_name='portrait')
        
        # Print comparison
        print("\nRating Comparison:")
        print(f"Image: {image_id}")
        print("\nDefault Profile:")
        print(f"  Rating Category: {default_rating.rating_category.name}")
        print(f"  Overall Score: {default_rating.overall_score:.1f}")
        print(f"  Technical Score: {default_rating.technical_score:.1f}")
        print(f"  Composition Score: {default_rating.composition_score:.1f}")
        
        print("\nPortrait Profile:")
        print(f"  Rating Category: {custom_rating.rating_category.name}")
        print(f"  Overall Score: {custom_rating.overall_score:.1f}")
        print(f"  Technical Score: {custom_rating.technical_score:.1f}")
        print(f"  Composition Score: {custom_rating.composition_score:.1f}")
        
        return True
    except Exception as e:
        print(f"Error in custom profile test: {e}")
        return False

def test_batch_rating():
    """Test batch rating of multiple images."""
    print("\n=== Testing Batch Rating ===")
    
    # Initialize components
    image_loader = ImageLoader()
    analysis_engine = AnalysisEngine()
    rating_system = RatingSystem()
    
    # Request a test directory
    test_dir_path = input("\nEnter path to a directory with images (or 'skip' to skip): ")
    
    if test_dir_path.lower() == 'skip':
        print("Skipping batch rating test.")
        return True
    
    if not os.path.exists(test_dir_path) or not os.path.isdir(test_dir_path):
        print(f"Invalid directory: {test_dir_path}")
        return False
    
    try:
        # Load images - with suppressed output for "File format not recognized"
        print(f"Loading images from: {test_dir_path}")
        with suppress_stdout():
            image_results = image_loader.load_from_directory(test_dir_path)
        
        if not image_results:
            print("No images found in directory.")
            return False
        
        print(f"Found {len(image_results)} images.")
        
        # Ask how many to process
        try:
            num_images = int(input(f"How many images to analyze (max {len(image_results)}, default=all): ") or len(image_results))
            num_images = min(max(1, num_images), len(image_results))
        except ValueError:
            num_images = len(image_results)
            
        print(f"Processing {num_images} images...")
        
        # Analyze and rate images
        analysis_results = {}
        
        for i, (path, image_data, metadata) in enumerate(image_results[:num_images]):
            image_id = os.path.basename(path)
            print(f"Analyzing image {i+1}/{num_images}: {image_id}", end='\r')
            with suppress_stdout():
                analysis_result = analysis_engine.analyze_image(image_data, metadata)
            analysis_results[image_id] = analysis_result
        
        print("\nRating images...")
        ratings = rating_system.batch_rate_images(analysis_results)
        
        # Print summary
        print("\nRating Summary:")
        print(f"Total images rated: {len(ratings)}")
        
        # Count by category
        distribution = rating_system.get_rating_distribution(list(ratings.values()))
        for category in RatingCategory:
            count = distribution[category]
            percentage = (count / len(ratings)) * 100 if ratings else 0
            print(f"{category.name}: {count} images ({percentage:.1f}%)")
        
        # Ask if user wants to see distribution chart
        if input("\nShow rating distribution chart? (y/n): ").lower() == 'y':
            plot_rating_distribution(ratings)
        
        return True
    except Exception as e:
        print(f"Error in batch rating test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_preference_learning():
    """Test preference learning based on user feedback."""
    print("\n=== Testing Preference Learning ===")
    
    # Initialize components
    image_loader = ImageLoader()
    analysis_engine = AnalysisEngine()
    rating_system = RatingSystem()
    
    # Request a test directory
    test_dir_path = input("\nEnter path to a directory with images (or 'skip' to skip): ")
    
    if test_dir_path.lower() == 'skip':
        print("Skipping preference learning test.")
        return True
    
    if not os.path.exists(test_dir_path) or not os.path.isdir(test_dir_path):
        print(f"Invalid directory: {test_dir_path}")
        return False
    
    try:
        # Load images - with output suppression
        print(f"Loading images from: {test_dir_path}")
        with suppress_stdout():
            image_results = image_loader.load_from_directory(test_dir_path)
        
        if not image_results:
            print("No images found in directory.")
            return False
        
        # Limit to 10 images for demonstration
        max_images = min(10, len(image_results))
        print(f"Using {max_images} images for preference learning demonstration.")
        
        # Analyze images - with output suppression
        analysis_results = {}
        
        for i, (path, image_data, metadata) in enumerate(image_results[:max_images]):
            image_id = os.path.basename(path)
            print(f"Analyzing image {i+1}/{max_images}: {image_id}", end='\r')
            with suppress_stdout():
                analysis_result = analysis_engine.analyze_image(image_data, metadata)
            analysis_results[image_id] = analysis_result
        
        print("\nGenerating initial ratings...")
        ratings_before = rating_system.batch_rate_images(
            analysis_results, apply_preferences=False
        )
        
        # Simulate user feedback: mark images with high exposure as favorites
        # and images with low sharpness as rejected
        print("\nSimulating user preferences:")
        print("- Favoring images with good exposure")
        print("- Rejecting images with poor sharpness")
        
        modified_ratings = {}
        
        for image_id, rating in ratings_before.items():
            result = analysis_results[image_id]
            
            # Simulate user explicit rating for some images
            if result.exposure_score > 90:
                # User prefers well-exposed images
                print(f"  Marking {image_id} as EXCELLENT (high exposure: {result.exposure_score:.1f})")
                modified_ratings[image_id] = rating_system.provide_feedback(
                    rating, user_rating=RatingCategory.EXCELLENT, is_favorite=True
                )
            elif result.sharpness_score < 40:
                # User doesn't like soft images
                print(f"  Marking {image_id} as REJECT (low sharpness: {result.sharpness_score:.1f})")
                modified_ratings[image_id] = rating_system.provide_feedback(
                    rating, user_rating=RatingCategory.REJECT, is_rejected=True
                )
            else:
                modified_ratings[image_id] = rating
        
        # Update preferences based on ratings
        print("\nUpdating preference model...")
        rating_system.update_preferences_from_ratings(analysis_results, modified_ratings)
        
        # Show the learned preferences
        print("\nLearned preference factors:")
        preference_model = rating_system.preference_model
        print(f"  Sharpness: {preference_model.sharpness_factor:.2f}")
        print(f"  Exposure: {preference_model.exposure_factor:.2f}")
        print(f"  Contrast: {preference_model.contrast_factor:.2f}")
        print(f"  Noise: {preference_model.noise_factor:.2f}")
        print(f"  Rule of Thirds: {preference_model.rule_of_thirds_factor:.2f}")
        print(f"  Symmetry: {preference_model.symmetry_factor:.2f}")
        print(f"  Subject Position: {preference_model.subject_position_factor:.2f}")
        
        # Re-rate images with updated preferences
        print("\nRe-rating images with learned preferences...")
        ratings_after = rating_system.batch_rate_images(
            analysis_results, apply_preferences=True
        )
        
        # Compare before and after
        print("\nRating changes after learning:")
        for image_id in analysis_results.keys():
            before = ratings_before[image_id]
            after = ratings_after[image_id]
            
            if before.rating_category != after.rating_category:
                print(f"  {image_id}: {before.rating_category.name} -> {after.rating_category.name}")
                print(f"    Score change: {before.overall_score:.1f} -> {after.overall_score:.1f}")
        
        # Ask if user wants to see visualization
        if input("\nShow preference learning visualization? (y/n): ").lower() == 'y':
            plot_learning_factors(preference_model)
            plot_score_comparison(ratings_before, ratings_after)
        
        # Test saving and loading the preference model
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp_path = temp.name
            
        print(f"\nSaving preference model to temporary file: {temp_path}")
        rating_system.save_preference_model(temp_path)
        
        print("Loading preference model from file...")
        new_rating_system = RatingSystem()
        new_rating_system.load_preference_model(temp_path)
        
        # Clean up
        os.unlink(temp_path)
        
        print("Preference model successfully saved and loaded.")
        
        return True
    except Exception as e:
        print(f"Error in preference learning test: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_tests():
    """Run all tests."""
    print("=== Testing Rating System ===\n")
    
    print("1. Testing basic rating functionality...")
    test_basic_rating()
    
    print("\n2. Testing custom rating profiles...")
    test_custom_profile()
    
    print("\n3. Testing batch rating...")
    test_batch_rating()
    
    print("\n4. Testing preference learning...")
    test_preference_learning()

if __name__ == "__main__":
    run_tests()