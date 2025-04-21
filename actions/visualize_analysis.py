from typing import Dict, Any
import os
import sys
import logging

# Import modules from the Cupcake library
from src import initialize_application
from src.analysis_visualizer import analyze_and_visualize

def run(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a visual analysis report for an image.

    Parameters:
        --image <path>      Path to the image to analyze (required)
        --output <path>     Path to save the analysis report (optional)
        --profile <name>    Rating profile to use (default: default)
        --use-gpu           Enable GPU acceleration if available
        --no-gpu            Force CPU fallback

    Example:
        cupcake visualize-analysis --image ./photos/image.jpg --output ./reports/analysis.png
    """
    # Validate required parameters
    if "image" not in params:
        raise ValueError("Required parameter --image is missing")
    
    image_path = params["image"]
    if not os.path.exists(image_path):
        raise ValueError(f"Image file does not exist: {image_path}")
    
    # Optional output path
    output_path = params.get("output", None)
    if output_path:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    
    # Initialize application components
    app = initialize_application()
    
    # Extract parameters
    rating_profile = params.get("profile", "default")
    
    # Configure GPU usage
    use_gpu = True  # Default to True
    if "no_gpu" in params and params["no_gpu"]:
        use_gpu = False
    elif "use_gpu" in params:
        use_gpu = bool(params["use_gpu"])
    else:
        # Get from settings
        use_gpu = app["settings"].get_setting("use_gpu", True)
    
    # Setup logging
    logger = logging.getLogger("cupcake.visualize_analysis")
    
    # Get analysis engine from the application
    analysis_engine = app["analysis_engine"]
    
    # Override analysis engine GPU setting
    analysis_engine.use_gpu = use_gpu
    
    # Process the image
    logger.info(f"Analyzing image: {image_path}")
    logger.info(f"Using profile: {rating_profile}")
    logger.info(f"GPU acceleration: {'enabled' if use_gpu else 'disabled'}")
    
    try:
        # Generate the analysis visualization
        output_file = analyze_and_visualize(image_path, analysis_engine, output_path)
        
        logger.info(f"Analysis report generated: {output_file}")
        
        return {
            "status": "completed",
            "input_image": image_path,
            "output_file": output_file,
            "profile": rating_profile
        }
        
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    # This allows the module to be run directly for testing
    if len(sys.argv) < 2:
        print("Usage: python visualize_analysis.py --image <path> [options]")
        sys.exit(1)
    
    # Parse command line arguments
    args = {}
    key = None
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            key = arg[2:].replace("-", "_")
            args[key] = True
        elif key:
            args[key] = arg
    
    # Run the action
    result = run(args)
    
    # Print results
    print("\n== Results ==")
    for k, v in result.items():
        print(f"{k}: {v}")