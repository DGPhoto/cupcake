# tests/test_analysis_engine.py

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.image_loader import ImageLoader
from src.analysis_engine import AnalysisEngine, ImageAnalysisResult

def display_analysis_results(image_path, image_data, result):
    """Display image with analysis results."""
    # Create a figure with 2 rows and 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Analysis Results: {os.path.basename(image_path)}', fontsize=16)
    
    # Original image
    axes[0, 0].imshow(image_data)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Edge map
    if result.edge_map is not None:
        axes[0, 1].imshow(result.edge_map, cmap='gray')
        axes[0, 1].set_title('Edge Detection')
        axes[0, 1].axis('off')
    
    # Face detection
    if result.face_locations:
        # Create a copy of the image for drawing
        img_with_faces = image_data.copy()
        for (x, y, w, h) in result.face_locations:
            cv2.rectangle(img_with_faces, (x, y), (x+w, y+h), (0, 255, 0), 2)
        axes[0, 2].imshow(img_with_faces)
        axes[0, 2].set_title(f'Face Detection ({result.face_count} faces)')
    else:
        axes[0, 2].imshow(image_data)
        axes[0, 2].set_title('No Faces Detected')
    axes[0, 2].axis('off')
    
    # Rule of thirds grid
    img_thirds = image_data.copy()
    h, w = img_thirds.shape[0], img_thirds.shape[1]
    
    # Draw rule of thirds lines
    for i in range(1, 3):
        # Vertical lines
        cv2.line(img_thirds, (w * i // 3, 0), (w * i // 3, h), (255, 255, 255), 1)
        # Horizontal lines
        cv2.line(img_thirds, (0, h * i // 3), (w, h * i // 3), (255, 255, 255), 1)
    
    axes[1, 0].imshow(img_thirds)
    axes[1, 0].set_title(f'Rule of Thirds: {result.rule_of_thirds_score:.1f}')
    axes[1, 0].axis('off')
    
    # Histogram
    if result.histogram_data is not None:
        # If we have separate RGB histograms
        if len(result.histogram_data) > 256:
            # Extract individual channel histograms
            hist_r = result.histogram_data[:256].flatten()
            hist_g = result.histogram_data[256:512].flatten()
            hist_b = result.histogram_data[512:].flatten()
            
            # Plot RGB histograms
            axes[1, 1].plot(hist_r, color='r', alpha=0.7)
            axes[1, 1].plot(hist_g, color='g', alpha=0.7)
            axes[1, 1].plot(hist_b, color='b', alpha=0.7)
        else:
            # Grayscale histogram
            axes[1, 1].plot(result.histogram_data, color='gray')
            
        axes[1, 1].set_title(f'Histogram (Exposure: {result.exposure_score:.1f})')
        axes[1, 1].set_xlim([0, 255])
    
    # Score summary as text
    axes[1, 2].axis('off')
    score_text = (
        f"TECHNICAL SCORES\n"
        f"Sharpness: {result.sharpness_score:.1f}\n"
        f"Exposure: {result.exposure_score:.1f}\n"
        f"Contrast: {result.contrast_score:.1f}\n"
        f"Noise: {result.noise_score:.1f}\n\n"
        f"COMPOSITION SCORES\n"
        f"Rule of Thirds: {result.rule_of_thirds_score:.1f}\n"
        f"Symmetry: {result.symmetry_score:.1f}\n"
        f"Subject Position: {result.subject_position_score:.1f}\n\n"
        f"OVERALL SCORES\n"
        f"Technical: {result.overall_technical_score:.1f}\n"
        f"Composition: {result.overall_composition_score:.1f}\n"
        f"Final Score: {result.overall_score:.1f}"
    )
    axes[1, 2].text(0, 0.5, score_text, fontsize=10, verticalalignment='center')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def test_analysis_single_image():
    """Test analyzing a single image."""
    
    image_loader = ImageLoader()
    analysis_engine = AnalysisEngine()
    
    # Request a test image path
    while True:
        test_image_path = input("\nEnter path to a test image (or 'skip' to skip): ")
        
        if test_image_path.lower() == 'skip':
            print("Skipping single image analysis test.")
            return True
        
        if not os.path.exists(test_image_path):
            print(f"File doesn't exist: {test_image_path}")
            continue
        
        try:
            # Load image
            print(f"Loading image: {test_image_path}")
            image_data, metadata = image_loader.load_from_path(test_image_path)
            
            # Analyze image
            print("Analyzing image...")
            result = analysis_engine.analyze_image(image_data, metadata)
            
            # Print analysis results
            print("\nAnalysis Results:")
            print(f"Image Dimensions: {image_data.shape}")
            print(f"Technical Scores: Sharpness={result.sharpness_score:.1f}, " 
                  f"Exposure={result.exposure_score:.1f}, "
                  f"Contrast={result.contrast_score:.1f}, "
                  f"Noise={result.noise_score:.1f}")
            print(f"Composition Scores: Rule of Thirds={result.rule_of_thirds_score:.1f}, "
                  f"Symmetry={result.symmetry_score:.1f}")
            if result.face_count > 0:
                print(f"Detected {result.face_count} faces with qualities: {[f'{q:.1f}' for q in result.face_qualities]}")
            print(f"Subject Position Score: {result.subject_position_score:.1f}")
            print(f"Overall Scores: Technical={result.overall_technical_score:.1f}, "
                  f"Composition={result.overall_composition_score:.1f}, "
                  f"Final={result.overall_score:.1f}")
            
            # Display visual results if matplotlib is available
            try:
                display_analysis_results(test_image_path, image_data, result)
            except Exception as e:
                print(f"Error displaying results: {e}")
            
            # Ask to try another image
            if input("\nAnalyze another image? (y/n): ").lower() != 'y':
                break
                
        except Exception as e:
            print(f"Error analyzing image: {e}")
            if input("Try another image? (y/n): ").lower() != 'y':
                break
    
    return True

def test_batch_analysis():
    """Test analyzing a batch of images from a directory."""
    
    image_loader = ImageLoader()
    analysis_engine = AnalysisEngine()
    
    # Request a test directory path
    while True:
        test_dir_path = input("\nEnter path to a directory with images (or 'skip' to skip): ")
        
        if test_dir_path.lower() == 'skip':
            print("Skipping batch analysis test.")
            return True
        
        if not os.path.exists(test_dir_path):
            print(f"Directory doesn't exist: {test_dir_path}")
            continue
            
        if not os.path.isdir(test_dir_path):
            print(f"Not a directory: {test_dir_path}")
            continue
        
        try:
            print(f"Loading images from: {test_dir_path}")
            results = image_loader.load_from_directory(test_dir_path)
            
            print(f"\nSuccessfully loaded {len(results)} images from directory")
            
            # Ask how many images to analyze
            max_images = len(results)
            try:
                num_images = int(input(f"How many images to analyze (1-{max_images}, default=3): ") or "3")
                num_images = min(max(1, num_images), max_images)
            except ValueError:
                num_images = min(3, max_images)
            
            # Analyze selected number of images
            print(f"\nAnalyzing {num_images} images...")
            
            for i, (path, image_data, metadata) in enumerate(results[:num_images]):
                filename = os.path.basename(path)
                print(f"\nAnalyzing image {i+1}/{num_images}: {filename}")
                
                # Analyze image
                result = analysis_engine.analyze_image(image_data, metadata)
                
                # Print summary
                print(f"  Technical: {result.overall_technical_score:.1f}, "
                      f"Composition: {result.overall_composition_score:.1f}, "
                      f"Overall: {result.overall_score:.1f}")
                
                # Display results for each image if requested
                if input("  Show detailed analysis? (y/n): ").lower() == 'y':
                    try:
                        display_analysis_results(path, image_data, result)
                    except Exception as e:
                        print(f"  Error displaying results: {e}")
            
            return True
            
        except Exception as e:
            print(f"Error analyzing batch: {e}")
            if input("Try another directory? (y/n): ").lower() != 'y':
                return False

def test_score_distribution():
    """Test the distribution of scores across images in a directory."""
    
    image_loader = ImageLoader()
    analysis_engine = AnalysisEngine()
    
    # Request a test directory path
    while True:
        test_dir_path = input("\nEnter path to a directory with images (or 'skip' to skip): ")
        
        if test_dir_path.lower() == 'skip':
            print("Skipping score distribution test.")
            return True
        
        if not os.path.exists(test_dir_path):
            print(f"Directory doesn't exist: {test_dir_path}")
            continue
            
        if not os.path.isdir(test_dir_path):
            print(f"Not a directory: {test_dir_path}")
            continue
        
        try:
            print(f"Loading images from: {test_dir_path}")
            results = image_loader.load_from_directory(test_dir_path)
            
            num_images = len(results)
            print(f"\nSuccessfully loaded {num_images} images from directory")
            
            if num_images == 0:
                print("No images found in directory.")
                continue
                
            # Analyze all images
            print("Analyzing all images... (this may take a while)")
            
            # Store scores
            technical_scores = []
            composition_scores = []
            overall_scores = []
            filenames = []
            
            for i, (path, image_data, metadata) in enumerate(results):
                filename = os.path.basename(path)
                print(f"  Analyzing {i+1}/{num_images}: {filename}", end='\r')
                
                # Analyze image
                result = analysis_engine.analyze_image(image_data, metadata)
                
                # Store scores
                technical_scores.append(result.overall_technical_score)
                composition_scores.append(result.overall_composition_score)
                overall_scores.append(result.overall_score)
                filenames.append(filename)
            
            print("\nAnalysis complete!")
            
            # Create histogram of scores
            plt.figure(figsize=(15, 10))
            
            # Technical scores histogram
            plt.subplot(2, 2, 1)
            plt.hist(technical_scores, bins=10, alpha=0.7, color='blue')
            plt.title('Technical Scores Distribution')
            plt.xlabel('Score')
            plt.ylabel('Number of Images')
            
            # Composition scores histogram
            plt.subplot(2, 2, 2)
            plt.hist(composition_scores, bins=10, alpha=0.7, color='green')
            plt.title('Composition Scores Distribution')
            plt.xlabel('Score')
            plt.ylabel('Number of Images')
            
            # Overall scores histogram
            plt.subplot(2, 2, 3)
            plt.hist(overall_scores, bins=10, alpha=0.7, color='red')
            plt.title('Overall Scores Distribution')
            plt.xlabel('Score')
            plt.ylabel('Number of Images')
            
            # Top and bottom 5 images
            plt.subplot(2, 2, 4)
            plt.axis('off')
            
            # Sort images by overall score
            sorted_indices = np.argsort(overall_scores)
            bottom_indices = sorted_indices[:5]
            top_indices = sorted_indices[-5:]
            
            # Create text for display
            text = "TOP 5 IMAGES:\n"
            for i, idx in enumerate(reversed(top_indices)):
                text += f"{i+1}. {filenames[idx]}: {overall_scores[idx]:.1f}\n"
            
            text += "\nBOTTOM 5 IMAGES:\n"
            for i, idx in enumerate(bottom_indices):
                text += f"{i+1}. {filenames[idx]}: {overall_scores[idx]:.1f}\n"
            
            plt.text(0, 0.5, text, fontsize=10, verticalalignment='center')
            
            plt.tight_layout()
            plt.show()
            
            return True
            
        except Exception as e:
            print(f"Error in score distribution test: {e}")
            if input("Try another directory? (y/n): ").lower() != 'y':
                return False

def run_tests():
    """Run all tests."""
    print("=== Testing Analysis Engine ===\n")
    
    print("1. Testing single image analysis...")
    test_analysis_single_image()
    
    print("\n2. Testing batch image analysis...")
    test_batch_analysis()
    
    print("\n3. Testing score distribution...")
    test_score_distribution()

if __name__ == "__main__":
    run_tests()