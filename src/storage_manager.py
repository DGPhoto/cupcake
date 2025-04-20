# src/storage_manager.py

import os
import shutil
import datetime
import pathlib
from enum import Enum
from typing import Dict, List, Set, Optional, Tuple, Any, Union, Callable
import logging
from tqdm import tqdm
from PIL import Image
import hashlib

from .selection_manager import SelectionManager, SelectionStatus

class ExportFormat(Enum):
    """Enum representing supported export formats."""
    ORIGINAL = "original"  # Keep original format
    JPEG = "jpeg"
    TIFF = "tiff"
    PNG = "png"
    

class NamingPattern(Enum):
    """Enum representing file naming patterns for exports."""
    ORIGINAL = "original"  # Keep original filename
    SEQUENCE = "sequence"  # Sequence number (e.g., 001, 002)
    DATETIME = "datetime"  # Based on capture date (YYYYMMDD_HHMMSS)
    CUSTOM = "custom"      # Custom pattern with placeholders


class FolderStructure(Enum):
    """Enum representing folder structure options for exports."""
    FLAT = "flat"              # All files in one directory
    DATE = "date"              # Organize by date (YYYY/MM/DD)
    RATING = "rating"          # Organize by rating
    COLLECTION = "collection"  # Organize by collection
    CUSTOM = "custom"          # Custom folder structure


class StorageManager:
    """
    Manages storage and export operations for selected images.
    Handles file operations, naming patterns, and folder organization.
    """
    
    def __init__(self, base_output_dir: str = "./output"):
        """
        Initialize the StorageManager.
        
        Args:
            base_output_dir: Base directory for export operations
        """
        self.base_output_dir = base_output_dir
        self.logger = logging.getLogger("cupcake.storage")
        
        # Create base output directory if it doesn't exist
        os.makedirs(base_output_dir, exist_ok=True)
    
    def export_selected(self, 
                      selection_manager: SelectionManager,
                      output_dir: Optional[str] = None,
                      export_format: ExportFormat = ExportFormat.ORIGINAL,
                      naming_pattern: NamingPattern = NamingPattern.ORIGINAL,
                      folder_structure: FolderStructure = FolderStructure.FLAT,
                      custom_name_pattern: Optional[str] = None,
                      custom_folder_pattern: Optional[str] = None,
                      jpeg_quality: int = 95,
                      tiff_compression: Optional[str] = None,
                      include_xmp: bool = True,
                      overwrite: bool = False) -> Dict[str, Any]:
        """
        Export selected images according to specified parameters.
        
        Args:
            selection_manager: Selection manager instance
            output_dir: Directory to export to (defaults to base_output_dir)
            export_format: Format to export images in
            naming_pattern: Pattern for naming exported files
            folder_structure: Structure for organizing exported files
            custom_name_pattern: Custom naming pattern (if pattern is CUSTOM)
            custom_folder_pattern: Custom folder pattern (if structure is CUSTOM)
            jpeg_quality: JPEG quality (0-100) if export_format is JPEG
            tiff_compression: TIFF compression type if export_format is TIFF
            include_xmp: Whether to include XMP sidecars for metadata
            overwrite: Whether to overwrite existing files
            
        Returns:
            Dictionary with export statistics
        """
        if output_dir is None:
            # Generate output directory with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(self.base_output_dir, f"export_{timestamp}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get selected images
        selected_images = selection_manager.get_selected_images()
        
        if not selected_images:
            self.logger.warning("No selected images to export")
            return {"exported": 0, "skipped": 0, "errors": 0}
        
        # Set up export tracking
        stats = {
            "total": len(selected_images),
            "exported": 0,
            "skipped": 0,
            "errors": 0,
            "output_dir": output_dir
        }
        
        # Create progress bar
        with tqdm(total=len(selected_images), desc="Exporting images") as progress:
            # Process each selected image
            for image_id in selected_images:
                try:
                    # Get image info
                    image_info = selection_manager.get_image_info(image_id)
                    if not image_info:
                        self.logger.warning(f"Image not found: {image_id}")
                        stats["errors"] += 1
                        progress.update(1)
                        continue
                    
                    # Determine destination path
                    dest_path = self._get_destination_path(
                        image_id,
                        image_info,
                        output_dir,
                        naming_pattern,
                        folder_structure,
                        export_format,
                        custom_name_pattern,
                        custom_folder_pattern
                    )
                    
                    # Create destination directory if it doesn't exist
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    
                    # Check if destination file already exists
                    if os.path.exists(dest_path) and not overwrite:
                        self.logger.info(f"File already exists, skipping: {dest_path}")
                        stats["skipped"] += 1
                        progress.update(1)
                        continue
                    
                    # Perform the export
                    success = self._export_image(
                        image_id,
                        dest_path,
                        export_format,
                        jpeg_quality,
                        tiff_compression
                    )
                    
                    if success:
                        stats["exported"] += 1
                        
                        # Export XMP sidecar if requested
                        if include_xmp:
                            xmp_path = os.path.splitext(dest_path)[0] + ".xmp"
                            self._export_xmp_sidecar(image_info, xmp_path)
                    else:
                        stats["errors"] += 1
                    
                    progress.update(1)
                    
                except Exception as e:
                    self.logger.error(f"Error exporting {image_id}: {str(e)}")
                    stats["errors"] += 1
                    progress.update(1)
        
        return stats
    
    def _get_destination_path(self,
                            image_id: str,
                            image_info: Dict[str, Any],
                            output_dir: str,
                            naming_pattern: NamingPattern,
                            folder_structure: FolderStructure,
                            export_format: ExportFormat,
                            custom_name_pattern: Optional[str] = None,
                            custom_folder_pattern: Optional[str] = None) -> str:
        """
        Determine the destination path for an exported image.
        
        Args:
            image_id: Image identifier
            image_info: Image information from selection manager
            output_dir: Base output directory
            naming_pattern: Naming pattern to use
            folder_structure: Folder structure to use
            export_format: Export format
            custom_name_pattern: Custom naming pattern
            custom_folder_pattern: Custom folder pattern
            
        Returns:
            Full destination path for the exported image
        """
        # Get metadata
        metadata = image_info.get("metadata", {})
        
        # Determine file extension based on export format
        original_ext = os.path.splitext(image_id)[1].lower()
        
        if export_format == ExportFormat.ORIGINAL:
            extension = original_ext
        elif export_format == ExportFormat.JPEG:
            extension = ".jpg"
        elif export_format == ExportFormat.TIFF:
            extension = ".tiff"
        elif export_format == ExportFormat.PNG:
            extension = ".png"
        else:
            extension = original_ext
        
        # Determine filename based on naming pattern
        if naming_pattern == NamingPattern.ORIGINAL:
            filename = os.path.basename(image_id)
            filename = os.path.splitext(filename)[0] + extension
        
        elif naming_pattern == NamingPattern.SEQUENCE:
            # Generate hash from image_id to create a stable sequence number
            hash_obj = hashlib.md5(image_id.encode())
            hash_int = int(hash_obj.hexdigest(), 16)
            filename = f"{hash_int % 10000:04d}{extension}"
        
        elif naming_pattern == NamingPattern.DATETIME:
            # Use capture date if available, otherwise use current time
            if "datetime" in metadata:
                try:
                    dt = datetime.datetime.strptime(metadata["datetime"], "%Y:%m:%d %H:%M:%S")
                    timestamp = dt.strftime("%Y%m%d_%H%M%S")
                except (ValueError, TypeError):
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            else:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            filename = f"{timestamp}{extension}"
        
        elif naming_pattern == NamingPattern.CUSTOM and custom_name_pattern:
            # Replace placeholders in custom pattern
            filename = self._apply_custom_pattern(custom_name_pattern, image_info, metadata)
            filename += extension
        
        else:
            # Fallback to original name
            filename = os.path.basename(image_id)
            filename = os.path.splitext(filename)[0] + extension
        
        # Determine subfolder based on folder structure
        if folder_structure == FolderStructure.FLAT:
            subfolder = ""
        
        elif folder_structure == FolderStructure.DATE:
            # Use capture date if available, otherwise use current date
            if "datetime" in metadata:
                try:
                    dt = datetime.datetime.strptime(metadata["datetime"], "%Y:%m:%d %H:%M:%S")
                    subfolder = dt.strftime("%Y/%m/%d")
                except (ValueError, TypeError):
                    subfolder = datetime.datetime.now().strftime("%Y/%m/%d")
            else:
                subfolder = datetime.datetime.now().strftime("%Y/%m/%d")
        
        elif folder_structure == FolderStructure.RATING:
            # Use rating if available, otherwise use "unrated"
            rating = image_info.get("rating", 0)
            subfolder = f"{rating}_stars"
        
        elif folder_structure == FolderStructure.COLLECTION:
            # Use first collection other than "All Images", or "uncategorized"
            collections = image_info.get("collections", set())
            collections = [c for c in collections if c != "All Images"]
            
            if collections:
                subfolder = collections[0]
            else:
                subfolder = "uncategorized"
        
        elif folder_structure == FolderStructure.CUSTOM and custom_folder_pattern:
            # Replace placeholders in custom pattern
            subfolder = self._apply_custom_pattern(custom_folder_pattern, image_info, metadata)
        
        else:
            subfolder = ""
        
        # Combine output directory, subfolder, and filename
        if subfolder:
            return os.path.join(output_dir, subfolder, filename)
        else:
            return os.path.join(output_dir, filename)
    
    def _apply_custom_pattern(self, 
                            pattern: str, 
                            image_info: Dict[str, Any],
                            metadata: Dict[str, Any]) -> str:
        """
        Apply a custom pattern with placeholders.
        
        Args:
            pattern: Pattern string with placeholders
            image_info: Image information from selection manager
            metadata: Image metadata
            
        Returns:
            Pattern with placeholders replaced
        """
        result = pattern
        
        # Image attributes
        result = result.replace("{filename}", os.path.basename(os.path.splitext(image_info.get("filepath", ""))[0]))
        result = result.replace("{rating}", str(image_info.get("rating", 0)))
        
        # Date placeholders
        if "datetime" in metadata:
            try:
                dt = datetime.datetime.strptime(metadata["datetime"], "%Y:%m:%d %H:%M:%S")
                result = result.replace("{year}", dt.strftime("%Y"))
                result = result.replace("{month}", dt.strftime("%m"))
                result = result.replace("{day}", dt.strftime("%d"))
                result = result.replace("{hour}", dt.strftime("%H"))
                result = result.replace("{minute}", dt.strftime("%M"))
                result = result.replace("{second}", dt.strftime("%S"))
            except (ValueError, TypeError):
                pass
        
        # Camera/lens placeholders
        result = result.replace("{camera_make}", metadata.get("camera_make", "unknown"))
        result = result.replace("{camera_model}", metadata.get("camera_model", "unknown"))
        result = result.replace("{lens_model}", metadata.get("lens_model", "unknown"))
        
        # Exposure placeholders
        if "focal_length" in metadata:
            result = result.replace("{focal_length}", f"{metadata['focal_length']:.0f}mm")
        
        if "f_number" in metadata:
            result = result.replace("{aperture}", f"f{metadata['f_number']:.1f}")
        
        if "exposure_time" in metadata:
            exp_time = metadata["exposure_time"]
            if exp_time >= 1:
                exp_str = f"{exp_time:.0f}s"
            else:
                exp_str = f"1_{int(1/exp_time)}"
            result = result.replace("{shutter}", exp_str)
        
        if "iso" in metadata:
            result = result.replace("{iso}", str(metadata["iso"]))
        
        # Replace invalid characters for filenames/paths
        invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        for char in invalid_chars:
            result = result.replace(char, '_')
        
        return result
    
    def _export_image(self,
                    image_id: str,
                    dest_path: str,
                    export_format: ExportFormat,
                    jpeg_quality: int = 95,
                    tiff_compression: Optional[str] = None) -> bool:
        """
        Export an image to the destination path.
        
        Args:
            image_id: Image identifier
            dest_path: Destination path
            export_format: Format to export in
            jpeg_quality: JPEG quality if exporting as JPEG
            tiff_compression: TIFF compression if exporting as TIFF
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # If keeping original format, just copy the file
            if export_format == ExportFormat.ORIGINAL:
                shutil.copy2(image_id, dest_path)
                return True
            
            # Otherwise, convert using PIL
            with Image.open(image_id) as img:
                if export_format == ExportFormat.JPEG:
                    img.save(dest_path, format="JPEG", quality=jpeg_quality)
                elif export_format == ExportFormat.TIFF:
                    compression = tiff_compression if tiff_compression else None
                    img.save(dest_path, format="TIFF", compression=compression)
                elif export_format == ExportFormat.PNG:
                    img.save(dest_path, format="PNG")
                else:
                    # Fallback to copy
                    shutil.copy2(image_id, dest_path)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error exporting {image_id} to {dest_path}: {str(e)}")
            return False
    
    def _export_xmp_sidecar(self, image_info: Dict[str, Any], xmp_path: str) -> bool:
        """
        Export an XMP sidecar file with image metadata.
        
        Args:
            image_info: Image information from selection manager
            xmp_path: Destination path for XMP file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import xml.etree.ElementTree as ET
            
            # Create XML structure
            root = ET.Element("x:xmpmeta", {
                "xmlns:x": "adobe:ns:meta/",
                "x:xmptk": "Cupcake Photo Culling Library"
            })
            
            rdf = ET.SubElement(root, "rdf:RDF", {
                "xmlns:rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                "xmlns:xmp": "http://ns.adobe.com/xap/1.0/",
                "xmlns:xmpMM": "http://ns.adobe.com/xap/1.0/mm/",
                "xmlns:dc": "http://purl.org/dc/elements/1.1/",
                "xmlns:lr": "http://ns.adobe.com/lightroom/1.0/"
            })
            
            description = ET.SubElement(rdf, "rdf:Description", {
                "rdf:about": image_info.get("filepath", "")
            })
            
            # Add rating
            rating = image_info.get("rating", 0)
            if rating > 0:
                ET.SubElement(description, "xmp:Rating").text = str(rating)
            
            # Add flag status
            status = image_info.get("status")
            if status and hasattr(status, "value"):
                if status.value == 1:  # SelectionStatus.SELECTED
                    ET.SubElement(description, "lr:pick").text = "1"
                elif status.value == 2:  # SelectionStatus.REJECTED
                    ET.SubElement(description, "lr:pick").text = "-1"
            
            # Add color label
            color_label = image_info.get("color_label")
            if color_label and hasattr(color_label, "value") and color_label.value != 0:
                color_map = {
                    1: "Red",
                    2: "Yellow",
                    3: "Green",
                    4: "Blue",
                    5: "Purple"
                }
                if color_label.value in color_map:
                    ET.SubElement(description, "lr:colorLabel").text = color_map[color_label.value]
            
            # Write XMP file
            tree = ET.ElementTree(root)
            tree.write(xmp_path, encoding="UTF-8", xml_declaration=True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting XMP sidecar to {xmp_path}: {str(e)}")
            return False
    
    def export_as_catalog(self, 
                        selection_manager: SelectionManager,
                        output_dir: str,
                        catalog_format: str = "lightroom") -> Dict[str, Any]:
        """
        Export a catalog file for use with external applications.
        
        Args:
            selection_manager: Selection manager instance
            output_dir: Directory to export catalog to
            catalog_format: Format of catalog ('lightroom' or 'capture_one')
            
        Returns:
            Dictionary with export statistics
        """
        os.makedirs(output_dir, exist_ok=True)
        
        stats = {
            "format": catalog_format,
            "output_dir": output_dir
        }
        
        if catalog_format.lower() == "lightroom":
            # Export Lightroom XMP sidecar files
            selection_manager.export_lightroom_metadata(output_dir)
            stats["exported"] = len(os.listdir(output_dir))
            
        elif catalog_format.lower() == "capture_one":
            # Export Capture One CSV file
            csv_path = os.path.join(output_dir, "capture_one_selections.csv")
            selection_manager.export_capture_one_metadata(csv_path)
            stats["exported"] = 1
            
        else:
            self.logger.error(f"Unsupported catalog format: {catalog_format}")
            stats["exported"] = 0
            
        return stats
    
    def organize_files(self,
                    source_dir: str,
                    output_dir: str,
                    folder_structure: FolderStructure,
                    naming_pattern: NamingPattern = NamingPattern.ORIGINAL,
                    custom_folder_pattern: Optional[str] = None,
                    custom_name_pattern: Optional[str] = None,
                    simulate: bool = False) -> Dict[str, Any]:
        """
        Organize image files from a source directory according to a folder structure.
        This is useful for organizing images without using the selection manager.
        
        Args:
            source_dir: Source directory containing images
            output_dir: Output directory for organized images
            folder_structure: Folder structure to use
            naming_pattern: Naming pattern to use
            custom_folder_pattern: Custom folder pattern if using custom structure
            custom_name_pattern: Custom name pattern if using custom naming
            simulate: If True, simulate organization without copying files
            
        Returns:
            Dictionary with organization statistics
        """
        from .image_loader import ImageLoader
        
        stats = {
            "total": 0,
            "organized": 0,
            "skipped": 0,
            "errors": 0,
            "source_dir": source_dir,
            "output_dir": output_dir
        }
        
        # Create loader to get metadata
        loader = ImageLoader()
        
        # Create output directory if not simulating
        if not simulate:
            os.makedirs(output_dir, exist_ok=True)
        
        # Find all image files in source directory
        image_files = []
        for root, _, files in os.walk(source_dir):
            for file in files:
                if self._is_image_file(file):
                    image_files.append(os.path.join(root, file))
        
        stats["total"] = len(image_files)
        
        # Process each image
        with tqdm(total=len(image_files), desc="Organizing images") as progress:
            for image_path in image_files:
                try:
                    # Load image to get metadata
                    _, metadata = loader.load_from_path(image_path)
                    
                    # Fake image_info for _get_destination_path
                    image_info = {
                        "filepath": image_path,
                        "metadata": metadata,
                        "rating": 0,
                        "collections": set(["All Images"])
                    }
                    
                    # Determine destination path
                    dest_path = self._get_destination_path(
                        image_path,
                        image_info,
                        output_dir,
                        naming_pattern,
                        folder_structure,
                        ExportFormat.ORIGINAL,
                        custom_name_pattern,
                        custom_folder_pattern
                    )
                    
                    # If not simulating, copy the file
                    if not simulate:
                        # Create destination directory
                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                        
                        # Copy the file
                        if os.path.exists(dest_path):
                            stats["skipped"] += 1
                        else:
                            shutil.copy2(image_path, dest_path)
                            stats["organized"] += 1
                    else:
                        stats["organized"] += 1
                    
                    progress.update(1)
                    
                except Exception as e:
                    self.logger.error(f"Error organizing {image_path}: {str(e)}")
                    stats["errors"] += 1
                    progress.update(1)
        
        return stats
    
    def export_with_metadata(self,
                          selection_manager: SelectionManager,
                          metadata_fields: List[str],
                          output_file: str,
                          format: str = "csv") -> Dict[str, Any]:
        """
        Export a list of selected images with their metadata.
        
        Args:
            selection_manager: Selection manager instance
            metadata_fields: List of metadata fields to include
            output_file: Path to output file
            format: Output format ('csv' or 'json')
            
        Returns:
            Dictionary with export statistics
        """
        import csv
        import json
        
        # Get selected images
        selected_images = selection_manager.get_selected_images()
        
        stats = {
            "total": len(selected_images),
            "exported": 0,
            "output_file": output_file
        }
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            if format.lower() == "csv":
                with open(output_file, 'w', newline='') as csvfile:
                    # Add basic fields
                    fieldnames = ["image_id", "rating", "status", "color_label"] + metadata_fields
                    
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for image_id in selected_images:
                        image_info = selection_manager.get_image_info(image_id)
                        if not image_info:
                            continue
                        
                        row = {
                            "image_id": image_id,
                            "rating": image_info.get("rating", 0),
                            "status": str(image_info.get("status", "UNRATED")),
                            "color_label": str(image_info.get("color_label", "NONE"))
                        }
                        
                        # Add requested metadata fields
                        metadata = image_info.get("metadata", {})
                        for field in metadata_fields:
                            row[field] = metadata.get(field, "")
                        
                        writer.writerow(row)
                        stats["exported"] += 1
            
            elif format.lower() == "json":
                data = []
                
                for image_id in selected_images:
                    image_info = selection_manager.get_image_info(image_id)
                    if not image_info:
                        continue
                    
                    item = {
                        "image_id": image_id,
                        "rating": image_info.get("rating", 0),
                        "status": str(image_info.get("status", "UNRATED")),
                        "color_label": str(image_info.get("color_label", "NONE")),
                        "metadata": {}
                    }
                    
                    # Add requested metadata fields
                    metadata = image_info.get("metadata", {})
                    for field in metadata_fields:
                        item["metadata"][field] = metadata.get(field, "")
                    
                    data.append(item)
                    stats["exported"] += 1
                
                with open(output_file, 'w') as jsonfile:
                    json.dump(data, jsonfile, indent=2)
            
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return {"total": len(selected_images), "exported": 0, "output_file": output_file}
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error exporting metadata: {str(e)}")
            return {"total": len(selected_images), "exported": 0, "error": str(e)}
    
    def _is_image_file(self, filename: str) -> bool:
        """
        Check if a file is an image file based on extension.
        
        Args:
            filename: Filename to check
            
        Returns:
            True if file is an image, False otherwise
        """
        from .image_formats import ImageFormats
        
        ext = os.path.splitext(filename)[1].lower().lstrip('.')
        return ImageFormats.is_supported_format(ext)
    
    def cleanup_temp_files(self, older_than_days: int = 7) -> int:
        """
        Clean up temporary files in the base output directory.
        
        Args:
            older_than_days: Remove files older than this many days
            
        Returns:
            Number of files removed
        """
        removed_count = 0
        cutoff_time = datetime.datetime.now() - datetime.timedelta(days=older_than_days)
        
        for root, dirs, files in os.walk(self.base_output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Check if file is old enough to delete
                file_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_time < cutoff_time:
                    try:
                        os.remove(file_path)
                        removed_count += 1
                    except Exception as e:
                        self.logger.error(f"Error removing {file_path}: {str(e)}")
            
            # Remove empty directories
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                if not os.listdir(dir_path):
                    try:
                        os.rmdir(dir_path)
                    except Exception as e:
                        self.logger.error(f"Error removing directory {dir_path}: {str(e)}")
        
        return removed_count