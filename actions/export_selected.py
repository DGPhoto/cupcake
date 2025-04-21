# actions/export_selected.py
from typing import Dict, Any
from src import initialize_application
from src.storage_manager import ExportFormat, NamingPattern, FolderStructure
from .base import ActionBase

class ExportSelectedAction(ActionBase):
    @staticmethod
    def run(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Export selected images to disk.
        
        Parameters:
            --session <path>       Path to selection session file (required)
            --output-dir <path>    Output directory for exports (required)
            --format <type>        Export format: original, jpeg, png, tiff (default: original)
            --naming <pattern>     Naming pattern: original, sequence, datetime, custom (default: original)
            --structure <type>     Folder structure: flat, date, rating, collection (default: flat)
            --quality <int>        JPEG quality 1-100 (default: 95)
            --overwrite            Overwrite existing files
            
        Example:
            cupcake export-selected --session ./session.json --output-dir ./exports --format jpeg --naming datetime
        """
        app = initialize_application()
        
        # Load the selection manager
        from src.selection_manager import SelectionManager
        selection_manager = SelectionManager.load_from_json(params["session"])
        
        # Configure the storage manager
        storage_manager = app["storage_manager"]
        
        # Export the images
        export_results = storage_manager.export_selected(
            selection_manager=selection_manager,
            output_dir=params["output_dir"],
            export_format=ExportFormat[params.get("format", "ORIGINAL").upper()],
            naming_pattern=NamingPattern[params.get("naming", "ORIGINAL").upper()],
            folder_structure=FolderStructure[params.get("structure", "FLAT").upper()],
            jpeg_quality=int(params.get("quality", 95)),
            overwrite=params.get("overwrite", False),
        )
        
        return export_results