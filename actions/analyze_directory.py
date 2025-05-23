from typing import Dict, Any
import os
import sys
import logging
import gc
from tqdm import tqdm
import numpy as np

# Import modules from the Cupcake library
from src import initialize_application
from src.storage_manager import ExportFormat, NamingPattern, FolderStructure

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
        --verbose              Show detailed progress
        --skip-export          Skip exporting selected images
        --debug-scores         Print detailed score information

    Example:
        cupcake analyze-directory --input-dir ./photos --output-dir ./selected --profile portrait --threshold 85
    """
    # Validate required parameters
    if "input_dir" not in params:
        raise ValueError("Required parameter --input-dir is missing")
    
    input_dir = params["input_dir"]
    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Initialize application components
    app = initialize_application()
    
    # Extract parameters with defaults
    rating_profile = params.get("profile", "default")
    threshold = float(params.get("threshold", 75.0))
    verbose = params.get("verbose", False)
    batch_size = int(params.get("batch_size", 10))
    debug_scores = params.get("debug_scores", False) or "debug" in params
    
    # Configure GPU usage
    use_gpu = True  # Default to True
    if "no_gpu" in params and params["no_gpu"]:
        use_gpu = False
    elif "use_gpu" in params:
        use_gpu = bool(params["use_gpu"])
    else:
        # Get from settings
        use_gpu = app["settings"].get_setting("use_gpu", True)
    
    # Configure export format if output is requested
    export_format = ExportFormat.ORIGINAL
    if "format" in params:
        format_str = params.get("format", "original").upper()
        if hasattr(ExportFormat, format_str):
            export_format = ExportFormat[format_str]
    
    # Setup logging
    logger = logging.getLogger("cupcake.analyze_directory")
    logger.setLevel(logging.INFO if verbose or debug_scores else logging.WARNING)
    
    # Get components from the application
    image_loader = app["image_loader"]
    analysis_engine = app["analysis_engine"]
    rating_system = app["rating_system"]
    selection_manager = app["selection_manager"]
    storage_manager = app["storage_manager"]
    
    # Override analysis engine GPU setting
    analysis_engine.use_gpu = use_gpu
    
    # Process directory
    logger.info(f"Processing directory: {input_dir}")
    logger.info(f"Using profile: {rating_profile}")
    logger.info(f"GPU acceleration: {'enabled' if use_gpu else 'disabled'}")
    
    # Start processing
    stats = {
        "total_images": 0,
        "processed": 0,
        "selected": 0,
        "rejected": 0,
        "errors": 0,
        "exported": 0,
        "detailed_scores": {}  # For storing detailed scores
    }
    
    # Find all images in the directory
    image_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower().lstrip('.')
            
            from src.image_formats import ImageFormats
            if ImageFormats.is_supported_format(ext):
                image_files.append(file_path)
    
    stats["total_images"] = len(image_files)
    
    if stats["total_images"] == 0:
        logger.warning(f"No supported images found in {input_dir}")
        return {
            "status": "completed",
            "message": f"No supported images found in {input_dir}",
            "stats": stats
        }
    
    # Process images in batches to manage memory
    for batch_start in range(0, len(image_files), batch_size):
        batch_end = min(batch_start + batch_size, len(image_files))
        batch = image_files[batch_start:batch_end]
        
        # Process each image in the batch
        for file_path in tqdm(batch, desc="Analyzing images", disable=not verbose):
            try:
                # Load image
                image_data, metadata = image_loader.load_from_path(file_path)
                
                # Register with selection manager
                image_id = file_path
                selection_manager.register_image(image_id, metadata)
                
                # Debug info about image
                if debug_scores:
                    print(f"\n{'='*50}")
                    print(f"Image: {os.path.basename(file_path)}")
                    print(f"Size: {image_data.shape}")
                    print(f"Type: {image_data.dtype}")
                    print(f"Format: {metadata.get('extension', 'unknown')}")
                    print(f"Camera: {metadata.get('camera_make', 'unknown')} {metadata.get('camera_model', '')}")
                
                # Add a hook to detect empty arrays during analysis
                original_np_mean = np.mean
                
                def debug_mean(a, *args, **kwargs):
                    if a.size == 0:
                        print(f"WARNING: Empty array passed to np.mean() during analysis")
                        print(f"Array shape: {a.shape}, dtype: {a.dtype}")
                        print(f"Caller: {sys._getframe().f_back.f_code.co_name}")
                        # Return 0 for empty arrays instead of raising a warning
                        return 0.0
                    return original_np_mean(a, *args, **kwargs)
                
                # Replace np.mean temporarily for debugging
                if debug_scores:
                    np.mean = debug_mean
                
                # Analyze image
                try:
                    analysis_result = analysis_engine.analyze_image(image_data, metadata)
                finally:
                    # Restore original np.mean
                    if debug_scores:
                        np.mean = original_np_mean
                
                # Rate image
                rating = rating_system.rate_image(
                    analysis_result, 
                    image_id, 
                    profile_name=rating_profile
                )
                
                # Store detailed scores for reporting
                if debug_scores:
                    # Print detailed scores
                    print(f"\nValutazione dettagliata per {os.path.basename(file_path)}:")
                    print(f"  Punteggio tecnico: {rating.technical_score:.1f}")
                    print(f"    - Nitidezza: {rating.sharpness_score:.1f}")
                    print(f"    - Esposizione: {rating.exposure_score:.1f}")
                    print(f"    - Contrasto: {rating.contrast_score:.1f}")
                    print(f"    - Rumore: {rating.noise_score:.1f}")
                    print(f"  Punteggio composizione: {rating.composition_score:.1f}")
                    print(f"    - Regola dei terzi: {rating.rule_of_thirds_score:.1f}")
                    print(f"    - Simmetria: {rating.symmetry_score:.1f}")
                    print(f"    - Posizione soggetto: {rating.subject_position_score:.1f}")
                    print(f"  Punteggio complessivo: {rating.overall_score:.1f}")
                    print(f"  Soglia di selezione: {threshold}")
                    print(f"  Risultato: {'Selezionato' if rating.overall_score >= threshold else 'Rifiutato'}")
                    
                    # Store scores in stats
                    stats["detailed_scores"][os.path.basename(file_path)] = {
                        "technical_score": rating.technical_score,
                        "sharpness_score": rating.sharpness_score,
                        "exposure_score": rating.exposure_score,
                        "contrast_score": rating.contrast_score,
                        "noise_score": rating.noise_score,
                        "composition_score": rating.composition_score,
                        "rule_of_thirds_score": rating.rule_of_thirds_score,
                        "symmetry_score": rating.symmetry_score,
                        "subject_position_score": rating.subject_position_score,
                        "overall_score": rating.overall_score,
                        "result": "Selected" if rating.overall_score >= threshold else "Rejected"
                    }
                
                # Apply selection based on threshold
                if rating.overall_score >= threshold:
                    selection_manager.mark_as_selected(image_id)
                    stats["selected"] += 1
                else:
                    selection_manager.mark_as_rejected(image_id)
                    stats["rejected"] += 1
                
                stats["processed"] += 1
                
                # Free memory
                del image_data
                del analysis_result
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                if debug_scores:
                    import traceback
                    traceback.print_exc()
                stats["errors"] += 1
    
    # Export selected images if requested
    if "output_dir" in params and not params.get("skip_export", False):
        output_dir = params["output_dir"]
        
        # Make sure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Export selected images
        logger.info(f"Exporting selected images to {output_dir}")
        export_results = storage_manager.export_selected(
            selection_manager=selection_manager,
            output_dir=output_dir,
            export_format=export_format,
            naming_pattern=NamingPattern.ORIGINAL,
            folder_structure=FolderStructure.FLAT,
            jpeg_quality=int(params.get("quality", 95))
        )
        
        stats["exported"] = export_results.get("exported", 0)
    
    # Save session if requested
    if "save_session" in params:
        session_file = params.get("save_session")
        selection_manager.save_to_json(session_file)
        logger.info(f"Selection session saved to {session_file}")
    
    # Print summary of scores if debug_scores
    if debug_scores and stats["detailed_scores"]:
        print("\n" + "="*50)
        print("RIEPILOGO PUNTEGGI:")
        print("="*50)
        
        # Calculate averages
        avg_technical = sum(data["technical_score"] for data in stats["detailed_scores"].values()) / len(stats["detailed_scores"])
        avg_composition = sum(data["composition_score"] for data in stats["detailed_scores"].values()) / len(stats["detailed_scores"])
        avg_overall = sum(data["overall_score"] for data in stats["detailed_scores"].values()) / len(stats["detailed_scores"])
        
        print(f"Punteggio tecnico medio: {avg_technical:.1f}")
        print(f"Punteggio composizione medio: {avg_composition:.1f}")
        print(f"Punteggio complessivo medio: {avg_overall:.1f}")
        print(f"Soglia di selezione: {threshold}")
        print(f"Selezionate: {stats['selected']}/{stats['total_images']} ({stats['selected']/stats['total_images']*100:.1f}%)")
        print("="*50)
    
    # Remove detailed scores before returning
    if "detailed_scores" in stats:
        del stats["detailed_scores"]
    
    return {
        "status": "completed",
        "input_directory": input_dir,
        "output_directory": params.get("output_dir", "None"),
        "profile": rating_profile,
        "total_images": stats["total_images"],
        "processed": stats["processed"],
        "selected": stats["selected"],
        "rejected": stats["rejected"],
        "exported": stats["exported"],
        "errors": stats["errors"]
    }

if __name__ == "__main__":
    # This allows the module to be run directly for testing
    if len(sys.argv) < 2:
        print("Usage: python analyze_directory.py --input-dir <path> [options]")
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