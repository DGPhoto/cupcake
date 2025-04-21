from typing import Dict, Any
from src import initialize_application
from src.storage_manager import ExportFormat
from complete_workflow import process_directory

def run(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze a directory of images and perform rating and selection.

    Parameters:
        --input-dir <path>     Path to directory with images (required)
        --output-dir <path>    Output path for selected images
        --profile <name>       Rating profile to use (default: default)
        --format <type>        Export format: original, jpeg, png, tiff (default: original)
        --threshold <float>    Culling threshold (default: 75.0)
        --batch-size <int>     Number of images to process at a time (default: 10)
        --use-gpu              Enable GPU acceleration if available
        --no-gpu               Force CPU fallback

    Example:
        cupcake analyze-directory --input-dir ./photos --output-dir ./selected --profile portrait --threshold 85
    """
    app = initialize_application()
    return process_directory(
        directory_path=params["input_dir"],
        output_dir=params["output_dir"],
        rating_profile=params.get("profile", "default"),
        export_format=ExportFormat[params.get("format", "ORIGINAL").upper()],
        use_gpu=params.get("use_gpu", True),
        culling_threshold=params.get("threshold", 75.0),
    )
