"""
Module for visualizing image analysis results.
Can generate visual reports like the one previously available with rawpy.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, Any, Optional, Tuple

class AnalysisVisualizer:
    """Class to visualize image analysis results similar to the previous rawpy visualization."""
    
    def __init__(self):
        """Initialize the visualizer."""
        pass
    
    def create_analysis_report(self, 
                             image_path: str, 
                             analysis_result: Any, 
                             output_path: Optional[str] = None) -> str:
        """
        Create a visual analysis report for an image.
        
        Args:
            image_path: Path to the original image
            analysis_result: Analysis result from AnalysisEngine
            output_path: Path to save the report image (if None, generates a path)
            
        Returns:
            Path to the generated report image
        """
        # Load the image
        try:
            from src.image_loader import ImageLoader
            loader = ImageLoader()
            image_data, metadata = loader.load_from_path(image_path)
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
        
        # Get the filename for display
        filename = os.path.basename(image_path)
        
        # Create the figure with a grid layout
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(f"Analysis Results: {filename}", fontsize=16)
        
        gs = GridSpec(3, 3, figure=fig)
        
        # Original image (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # Edge detection (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        edge_map = self._get_edge_map(image_data)
        ax2.imshow(edge_map, cmap='gray')
        ax2.set_title("Edge Detection")
        ax2.axis('off')
        
        # Face detection (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        face_img = self._highlight_faces(image_data.copy(), analysis_result.face_locations)
        ax3.imshow(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        face_count = len(analysis_result.face_locations) if hasattr(analysis_result, 'face_locations') else 0
        ax3.set_title(f"Face Detection ({face_count} faces)")
        ax3.axis('off')
        
        # Rule of thirds grid (bottom left)
        ax4 = fig.add_subplot(gs[1, 0])
        thirds_img = self._draw_rule_of_thirds(image_data.copy())
        ax4.imshow(cv2.cvtColor(thirds_img, cv2.COLOR_BGR2RGB))
        ax4.set_title(f"Rule of Thirds: {analysis_result.rule_of_thirds_score:.1f}")
        ax4.axis('off')
        
        # Histogram (bottom middle and right)
        ax5 = fig.add_subplot(gs[1, 1:])
        self._plot_histogram(image_data, ax5)
        ax5.set_title(f"Histogram (Exposure: {analysis_result.exposure_score:.1f})")
        
        # Technical scores text (bottom left)
        ax6 = fig.add_subplot(gs[2, 0])
        ax6.axis('off')
        tech_text = (
            "TECHNICAL SCORES\n"
            f"Sharpness: {analysis_result.sharpness_score:.1f}\n"
            f"Exposure: {analysis_result.exposure_score:.1f}\n"
            f"Contrast: {analysis_result.contrast_score:.1f}\n"
            f"Noise: {analysis_result.noise_score:.1f}\n"
        )
        ax6.text(0.1, 0.5, tech_text, fontsize=12, verticalalignment='center')
        
        # Composition scores text (bottom middle)
        ax7 = fig.add_subplot(gs[2, 1])
        ax7.axis('off')
        comp_text = (
            "COMPOSITION SCORES\n"
            f"Rule of Thirds: {analysis_result.rule_of_thirds_score:.1f}\n"
            f"Symmetry: {analysis_result.symmetry_score:.1f}\n"
            f"Subject Position: {analysis_result.subject_position_score:.1f}\n"
        )
        ax7.text(0.1, 0.5, comp_text, fontsize=12, verticalalignment='center')
        
        # Overall scores text (bottom right)
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        overall_text = (
            "OVERALL SCORES\n"
            f"Technical: {analysis_result.overall_technical_score:.1f}\n"
            f"Composition: {analysis_result.overall_composition_score:.1f}\n"
            f"Final Score: {analysis_result.overall_score:.1f}\n"
        )
        ax8.text(0.1, 0.5, overall_text, fontsize=12, verticalalignment='center')
        
        # Generate output path if not provided
        if output_path is None:
            base_dir = os.path.dirname(image_path)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(base_dir, f"{base_name}_analysis.png")
        
        # Save the figure
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for title
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        
        return output_path
    
    def _get_edge_map(self, image_data: np.ndarray) -> np.ndarray:
        """Generate edge map for visualization."""
        if len(image_data.shape) == 3 and image_data.shape[2] >= 3:
            # Convert to grayscale if color image
            gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_data
            
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        return edges
    
    def _highlight_faces(self, image: np.ndarray, face_locations: list) -> np.ndarray:
        """Draw rectangles around detected faces."""
        img_copy = image.copy()
        
        if not face_locations:
            return img_copy
            
        for (x, y, w, h) in face_locations:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        return img_copy
    
    def _draw_rule_of_thirds(self, image: np.ndarray) -> np.ndarray:
        """Draw rule of thirds grid on image."""
        img_copy = image.copy()
        h, w = img_copy.shape[:2]
        
        # Calculate grid line positions
        h1, h2 = h // 3, 2 * h // 3
        w1, w2 = w // 3, 2 * w // 3
        
        # Draw horizontal lines
        cv2.line(img_copy, (0, h1), (w, h1), (255, 255, 255), 1)
        cv2.line(img_copy, (0, h2), (w, h2), (255, 255, 255), 1)
        
        # Draw vertical lines
        cv2.line(img_copy, (w1, 0), (w1, h), (255, 255, 255), 1)
        cv2.line(img_copy, (w2, 0), (w2, h), (255, 255, 255), 1)
        
        # Draw intersection points
        points = [(w1, h1), (w1, h2), (w2, h1), (w2, h2)]
        for pt in points:
            cv2.circle(img_copy, pt, 5, (0, 255, 255), -1)
            
        return img_copy
    
    def _plot_histogram(self, image: np.ndarray, ax) -> None:
        """Plot RGB histogram."""
        colors = ('b', 'g', 'r')
        
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            ax.plot(hist, color=color)
            
        ax.set_xlim([0, 256])
        ax.grid(True)


def analyze_and_visualize(image_path: str, analysis_engine, output_path: Optional[str] = None) -> str:
    """
    Analyze an image and create a visualization of the results.
    
    Args:
        image_path: Path to the image
        analysis_engine: Instance of AnalysisEngine
        output_path: Optional path to save visualization
        
    Returns:
        Path to the generated visualization
    """
    # Load the image
    from src.image_loader import ImageLoader
    loader = ImageLoader()
    image_data, metadata = loader.load_from_path(image_path)
    
    # Analyze the image
    analysis_result = analysis_engine.analyze_image(image_data, metadata)
    
    # Create visualization
    visualizer = AnalysisVisualizer()
    return visualizer.create_analysis_report(image_path, analysis_result, output_path)