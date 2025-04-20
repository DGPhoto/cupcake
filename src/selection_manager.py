# src/selection_manager.py

import os
import json
import datetime
from enum import Enum
from typing import Dict, List, Set, Optional, Tuple, Any, Union
import xml.etree.ElementTree as ET
import csv

class SelectionStatus(Enum):
    """Enum representing the selection status of an image."""
    UNRATED = 0
    SELECTED = 1
    REJECTED = 2


class ColorLabel(Enum):
    """Color labels compatible with Lightroom and Capture One."""
    NONE = 0
    RED = 1
    YELLOW = 2
    GREEN = 3
    BLUE = 4
    PURPLE = 5


class SelectionManager:
    """
    Manages the selection and organization of images in a culling workflow.
    Provides compatibility with Lightroom and Capture One selection systems.
    """
    
    def __init__(self, project_name: str = "Untitled"):
        self.project_name = project_name
        self.creation_date = datetime.datetime.now()
        
        # Core selection tracking
        self._images: Dict[str, Dict[str, Any]] = {}  # Key: image_id, Value: image data
        
        # Collections (like albums or sets)
        self._collections: Dict[str, Set[str]] = {"All Images": set()}
        self._active_collection = "All Images"
        
        # History tracking
        self._history: List[Dict[str, Any]] = []
        
        # Filter state
        self._active_filters = {}
        
        # Statistics
        self._stats = {
            "total": 0,
            "selected": 0,
            "rejected": 0,
            "unrated": 0
        }
    
    def register_image(self, image_id: str, metadata: Dict[str, Any]) -> None:
        """
        Register a new image with the selection manager.
        
        Args:
            image_id: Unique identifier for the image
            metadata: Image metadata from the ImageLoader
        """
        if image_id in self._images:
            return  # Already registered
        
        # Initialize image record
        self._images[image_id] = {
            "status": SelectionStatus.UNRATED,
            "rating": 0,  # 0-5 star rating
            "color_label": ColorLabel.NONE,
            "flags": set(),  # Custom flags
            "metadata": metadata,
            "collections": {"All Images"},
            "history": [],
            "custom_metadata": {},
            "added_date": datetime.datetime.now()
        }
        
        # Add to default collection
        self._collections["All Images"].add(image_id)
        
        # Update stats
        self._stats["total"] += 1
        self._stats["unrated"] += 1
    
    def register_images_from_loader(self, loaded_images: List[Tuple[str, Any, Dict[str, Any]]]) -> None:
        """
        Register multiple images from ImageLoader results.
        
        Args:
            loaded_images: List of (path, image_data, metadata) tuples from ImageLoader
        """
        for path, _, metadata in loaded_images:
            image_id = metadata.get('filepath', path)
            self.register_image(image_id, metadata)
    
    # ----- Selection Operations -----
    
    def mark_as_selected(self, image_id: str) -> bool:
        """
        Mark an image as selected.
        
        Args:
            image_id: Image identifier
            
        Returns:
            True if the operation was successful
        """
        if image_id not in self._images:
            return False
        
        old_status = self._images[image_id]["status"]
        
        # Update status
        self._images[image_id]["status"] = SelectionStatus.SELECTED
        
        # Add to history
        self._add_to_history(image_id, "status_change", 
                            old_value=old_status, 
                            new_value=SelectionStatus.SELECTED)
        
        # Update stats
        if old_status == SelectionStatus.UNRATED:
            self._stats["unrated"] -= 1
            self._stats["selected"] += 1
        elif old_status == SelectionStatus.REJECTED:
            self._stats["rejected"] -= 1
            self._stats["selected"] += 1
            
        return True
    
    def mark_as_rejected(self, image_id: str) -> bool:
        """
        Mark an image as rejected.
        
        Args:
            image_id: Image identifier
            
        Returns:
            True if the operation was successful
        """
        if image_id not in self._images:
            return False
        
        old_status = self._images[image_id]["status"]
        
        # Update status
        self._images[image_id]["status"] = SelectionStatus.REJECTED
        
        # Add to history
        self._add_to_history(image_id, "status_change", 
                            old_value=old_status, 
                            new_value=SelectionStatus.REJECTED)
        
        # Update stats
        if old_status == SelectionStatus.UNRATED:
            self._stats["unrated"] -= 1
            self._stats["rejected"] += 1
        elif old_status == SelectionStatus.SELECTED:
            self._stats["selected"] -= 1
            self._stats["rejected"] += 1
            
        return True
    
    def mark_as_unrated(self, image_id: str) -> bool:
        """
        Reset an image to unrated status.
        
        Args:
            image_id: Image identifier
            
        Returns:
            True if the operation was successful
        """
        if image_id not in self._images:
            return False
        
        old_status = self._images[image_id]["status"]
        
        # Update status
        self._images[image_id]["status"] = SelectionStatus.UNRATED
        
        # Add to history
        self._add_to_history(image_id, "status_change", 
                            old_value=old_status, 
                            new_value=SelectionStatus.UNRATED)
        
        # Update stats
        if old_status == SelectionStatus.SELECTED:
            self._stats["selected"] -= 1
            self._stats["unrated"] += 1
        elif old_status == SelectionStatus.REJECTED:
            self._stats["rejected"] -= 1
            self._stats["unrated"] += 1
            
        return True
    
    def set_rating(self, image_id: str, rating: int) -> bool:
        """
        Set the star rating for an image (0-5 scale).
        
        Args:
            image_id: Image identifier
            rating: Rating value (0-5)
            
        Returns:
            True if the operation was successful
        """
        if image_id not in self._images:
            return False
            
        if not (0 <= rating <= 5):
            raise ValueError("Rating must be between 0 and 5")
        
        old_rating = self._images[image_id]["rating"]
        
        # Update rating
        self._images[image_id]["rating"] = rating
        
        # Add to history
        self._add_to_history(image_id, "rating_change", 
                            old_value=old_rating, 
                            new_value=rating)
            
        return True
    
    def set_color_label(self, image_id: str, color: ColorLabel) -> bool:
        """
        Set the color label for an image.
        
        Args:
            image_id: Image identifier
            color: ColorLabel enum value
            
        Returns:
            True if the operation was successful
        """
        if image_id not in self._images:
            return False
        
        old_color = self._images[image_id]["color_label"]
        
        # Update color
        self._images[image_id]["color_label"] = color
        
        # Add to history
        self._add_to_history(image_id, "color_change", 
                            old_value=old_color, 
                            new_value=color)
            
        return True
    
    def toggle_flag(self, image_id: str, flag_name: str) -> bool:
        """
        Toggle a custom flag on an image.
        
        Args:
            image_id: Image identifier
            flag_name: Name of the flag to toggle
            
        Returns:
            True if the flag was added, False if it was removed
        """
        if image_id not in self._images:
            raise KeyError(f"Image {image_id} not found")
        
        flags = self._images[image_id]["flags"]
        
        if flag_name in flags:
            flags.remove(flag_name)
            self._add_to_history(image_id, "flag_removed", flag=flag_name)
            return False
        else:
            flags.add(flag_name)
            self._add_to_history(image_id, "flag_added", flag=flag_name)
            return True
    
    # ----- Collections Management -----
    
    def create_collection(self, collection_name: str) -> bool:
        """
        Create a new collection.
        
        Args:
            collection_name: Name of the new collection
            
        Returns:
            True if the collection was created
        """
        if collection_name in self._collections:
            return False
            
        self._collections[collection_name] = set()
        return True
    
    def add_to_collection(self, image_id: str, collection_name: str) -> bool:
        """
        Add an image to a collection.
        
        Args:
            image_id: Image identifier
            collection_name: Name of the collection
            
        Returns:
            True if the image was added
        """
        if image_id not in self._images:
            return False
            
        if collection_name not in self._collections:
            self.create_collection(collection_name)
            
        self._collections[collection_name].add(image_id)
        self._images[image_id]["collections"].add(collection_name)
        
        self._add_to_history(image_id, "added_to_collection", collection=collection_name)
        return True
    
    def remove_from_collection(self, image_id: str, collection_name: str) -> bool:
        """
        Remove an image from a collection.
        
        Args:
            image_id: Image identifier
            collection_name: Name of the collection
            
        Returns:
            True if the image was removed
        """
        if (image_id not in self._images or 
            collection_name not in self._collections or
            collection_name == "All Images"):
            return False
            
        self._collections[collection_name].discard(image_id)
        self._images[image_id]["collections"].discard(collection_name)
        
        self._add_to_history(image_id, "removed_from_collection", collection=collection_name)
        return True
    
    def set_active_collection(self, collection_name: str) -> bool:
        """
        Set the active collection for operations.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            True if the active collection was changed
        """
        if collection_name not in self._collections:
            return False
            
        self._active_collection = collection_name
        return True
    
    def get_collection_images(self, collection_name: str = None) -> List[str]:
        """
        Get all image IDs in a collection.
        
        Args:
            collection_name: Name of the collection, or None for active collection
            
        Returns:
            List of image IDs in the collection
        """
        if collection_name is None:
            collection_name = self._active_collection
            
        if collection_name not in self._collections:
            return []
            
        return list(self._collections[collection_name])
    
    # ----- Filtering and Querying -----
    
    def filter_images(self, **filters) -> List[str]:
        """
        Filter images based on multiple criteria.
        
        Args:
            **filters: Keyword arguments for filtering
                status: SelectionStatus enum
                rating_min: Minimum rating
                rating_max: Maximum rating
                color: ColorLabel enum
                date_from: Datetime object
                date_to: Datetime object
                has_flag: Flag name that must be present
                
        Returns:
            List of image IDs that match the filters
        """
        collection = self._collections[self._active_collection]
        results = set(collection)
        
        # Status filter
        if "status" in filters:
            status = filters["status"]
            results = {img_id for img_id in results 
                      if self._images[img_id]["status"] == status}
        
        # Rating filter
        if "rating_min" in filters:
            min_rating = filters["rating_min"]
            results = {img_id for img_id in results 
                      if self._images[img_id]["rating"] >= min_rating}
                      
        if "rating_max" in filters:
            max_rating = filters["rating_max"]
            results = {img_id for img_id in results 
                      if self._images[img_id]["rating"] <= max_rating}
        
        # Color filter
        if "color" in filters:
            color = filters["color"]
            results = {img_id for img_id in results 
                      if self._images[img_id]["color_label"] == color}
        
        # Date filter
        if "date_from" in filters:
            date_from = filters["date_from"]
            results = {img_id for img_id in results 
                      if self._images[img_id]["added_date"] >= date_from}
                      
        if "date_to" in filters:
            date_to = filters["date_to"]
            results = {img_id for img_id in results 
                      if self._images[img_id]["added_date"] <= date_to}
        
        # Flag filter
        if "has_flag" in filters:
            flag = filters["has_flag"]
            results = {img_id for img_id in results 
                      if flag in self._images[img_id]["flags"]}
        
        # Save the active filters
        self._active_filters = filters
        
        return list(results)
    
    def get_selected_images(self) -> List[str]:
        """
        Get all selected images in the active collection.
        
        Returns:
            List of selected image IDs
        """
        return self.filter_images(status=SelectionStatus.SELECTED)
    
    def get_rejected_images(self) -> List[str]:
        """
        Get all rejected images in the active collection.
        
        Returns:
            List of rejected image IDs
        """
        return self.filter_images(status=SelectionStatus.REJECTED)
    
    def get_unrated_images(self) -> List[str]:
        """
        Get all unrated images in the active collection.
        
        Returns:
            List of unrated image IDs
        """
        return self.filter_images(status=SelectionStatus.UNRATED)
    
    def get_image_info(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        Get all information about an image.
        
        Args:
            image_id: Image identifier
            
        Returns:
            Dictionary with image information or None if not found
        """
        if image_id not in self._images:
            return None
            
        return self._images[image_id]
    
    # ----- Import/Export -----
    
    def export_lightroom_metadata(self, output_path: str) -> None:
        """
        Export selection data to a format compatible with Lightroom.
        Generates XMP sidecar files for each image.
        
        Args:
            output_path: Directory to output XMP files
        """
        os.makedirs(output_path, exist_ok=True)
        
        for image_id, info in self._images.items():
            # Create XMP file name from image path
            base_name = os.path.basename(image_id)
            name, ext = os.path.splitext(base_name)
            xmp_path = os.path.join(output_path, f"{name}.xmp")
            
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
                "rdf:about": image_id
            })
            
            # Add rating
            rating = info["rating"]
            if rating > 0:
                ET.SubElement(description, "xmp:Rating").text = str(rating)
            
            # Add flag status
            if info["status"] == SelectionStatus.SELECTED:
                ET.SubElement(description, "lr:pick").text = "1"
            elif info["status"] == SelectionStatus.REJECTED:
                ET.SubElement(description, "lr:pick").text = "-1"
            
            # Add color label
            if info["color_label"] != ColorLabel.NONE:
                color_map = {
                    ColorLabel.RED: "Red",
                    ColorLabel.YELLOW: "Yellow",
                    ColorLabel.GREEN: "Green",
                    ColorLabel.BLUE: "Blue",
                    ColorLabel.PURPLE: "Purple"
                }
                ET.SubElement(description, "lr:colorLabel").text = color_map[info["color_label"]]
            
            # Write XMP file
            tree = ET.ElementTree(root)
            tree.write(xmp_path, encoding="UTF-8", xml_declaration=True)
    
    def export_capture_one_metadata(self, output_path: str) -> None:
        """
        Export selection data to a format compatible with Capture One.
        
        Args:
            output_path: Path to output CSV file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['IMAGE_PATH', 'RATING', 'COLOR_TAG', 'PICK_STATUS']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            
            for image_id, info in self._images.items():
                row = {
                    'IMAGE_PATH': image_id,
                    'RATING': info["rating"],
                    'COLOR_TAG': info["color_label"].name,
                    'PICK_STATUS': info["status"].name
                }
                writer.writerow(row)
    
    def import_lightroom_metadata(self, xmp_directory: str) -> int:
        """
        Import selection data from Lightroom XMP sidecar files.
        
        Args:
            xmp_directory: Directory containing XMP files
            
        Returns:
            Number of images updated
        """
        updated_count = 0
        
        for filename in os.listdir(xmp_directory):
            if not filename.endswith('.xmp'):
                continue
                
            xmp_path = os.path.join(xmp_directory, filename)
            
            try:
                tree = ET.parse(xmp_path)
                root = tree.getroot()
                
                # Find the rdf:Description element
                ns = {
                    'x': 'adobe:ns:meta/',
                    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                    'xmp': 'http://ns.adobe.com/xap/1.0/',
                    'lr': 'http://ns.adobe.com/lightroom/1.0/'
                }
                
                desc = root.find('.//rdf:Description', ns)
                if desc is None:
                    continue
                
                # Get image path from rdf:about attribute
                image_path = desc.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about')
                
                # If we can't find this image, try matching by basename
                image_id = None
                if image_path in self._images:
                    image_id = image_path
                else:
                    base_name = os.path.splitext(os.path.basename(filename))[0]
                    for img_id in self._images:
                        if os.path.splitext(os.path.basename(img_id))[0] == base_name:
                            image_id = img_id
                            break
                
                if image_id is None:
                    continue
                
                # Extract rating
                rating_elem = desc.find('./xmp:Rating', ns)
                if rating_elem is not None and rating_elem.text:
                    self.set_rating(image_id, int(rating_elem.text))
                
                # Extract pick status
                pick_elem = desc.find('./lr:pick', ns)
                if pick_elem is not None and pick_elem.text:
                    pick_value = int(pick_elem.text)
                    if pick_value == 1:
                        self.mark_as_selected(image_id)
                    elif pick_value == -1:
                        self.mark_as_rejected(image_id)
                
                # Extract color label
                color_elem = desc.find('./lr:colorLabel', ns)
                if color_elem is not None and color_elem.text:
                    color_map = {
                        "Red": ColorLabel.RED,
                        "Yellow": ColorLabel.YELLOW,
                        "Green": ColorLabel.GREEN,
                        "Blue": ColorLabel.BLUE,
                        "Purple": ColorLabel.PURPLE
                    }
                    if color_elem.text in color_map:
                        self.set_color_label(image_id, color_map[color_elem.text])
                
                updated_count += 1
                
            except Exception as e:
                print(f"Error processing XMP file {xmp_path}: {e}")
        
        return updated_count
    
    def import_capture_one_metadata(self, csv_path: str) -> int:
        """
        Import selection data from a Capture One metadata CSV file.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Number of images updated
        """
        updated_count = 0
        
        try:
            with open(csv_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row in reader:
                    image_path = row.get('IMAGE_PATH')
                    
                    # If we can't find this image, try matching by basename
                    image_id = None
                    if image_path in self._images:
                        image_id = image_path
                    else:
                        base_name = os.path.basename(image_path)
                        for img_id in self._images:
                            if os.path.basename(img_id) == base_name:
                                image_id = img_id
                                break
                    
                    if image_id is None:
                        continue
                    
                    # Extract rating
                    rating = row.get('RATING')
                    if rating and rating.isdigit():
                        self.set_rating(image_id, int(rating))
                    
                    # Extract pick status
                    pick_status = row.get('PICK_STATUS')
                    if pick_status:
                        if pick_status == 'SELECTED':
                            self.mark_as_selected(image_id)
                        elif pick_status == 'REJECTED':
                            self.mark_as_rejected(image_id)
                        elif pick_status == 'UNRATED':
                            self.mark_as_unrated(image_id)
                    
                    # Extract color tag
                    color_tag = row.get('COLOR_TAG')
                    if color_tag:
                        try:
                            color = ColorLabel[color_tag]
                            self.set_color_label(image_id, color)
                        except KeyError:
                            pass
                    
                    updated_count += 1
                    
        except Exception as e:
            print(f"Error processing CSV file {csv_path}: {e}")
        
        return updated_count
    
    # ----- History and Undo/Redo -----
    
    def _add_to_history(self, image_id: str, action: str, **details) -> None:
        """
        Add an action to the history.
        
        Args:
            image_id: Image identifier
            action: Type of action performed
            **details: Additional details about the action
        """
        timestamp = datetime.datetime.now()
        
        history_entry = {
            "timestamp": timestamp,
            "image_id": image_id,
            "action": action,
            "details": details
        }
        
        # Add to global history
        self._history.append(history_entry)
        
        # Add to image-specific history
        if image_id in self._images:
            self._images[image_id]["history"].append(history_entry)
    
    def undo_last_action(self) -> bool:
        """
        Undo the last action in the history.
        
        Returns:
            True if an action was undone
        """
        if not self._history:
            return False
        
        # Get last action
        last_action = self._history.pop()
        image_id = last_action["image_id"]
        action = last_action["action"]
        details = last_action["details"]
        
        # Undo based on action type
        if action == "status_change":
            old_status = details["old_value"]
            self._images[image_id]["status"] = old_status
            
            # Update stats
            new_status = details["new_value"]
            if new_status == SelectionStatus.SELECTED:
                self._stats["selected"] -= 1
            elif new_status == SelectionStatus.REJECTED:
                self._stats["rejected"] -= 1
            elif new_status == SelectionStatus.UNRATED:
                self._stats["unrated"] -= 1
                
            if old_status == SelectionStatus.SELECTED:
                self._stats["selected"] += 1
            elif old_status == SelectionStatus.REJECTED:
                self._stats["rejected"] += 1
            elif old_status == SelectionStatus.UNRATED:
                self._stats["unrated"] += 1
                
        elif action == "rating_change":
            old_rating = details["old_value"]
            self._images[image_id]["rating"] = old_rating
            
        elif action == "color_change":
            old_color = details["old_value"]
            self._images[image_id]["color_label"] = old_color
            
        elif action == "flag_added":
            flag = details["flag"]
            self._images[image_id]["flags"].discard(flag)
            
        elif action == "flag_removed":
            flag = details["flag"]
            self._images[image_id]["flags"].add(flag)
            
        elif action == "added_to_collection":
            collection = details["collection"]
            self._collections[collection].discard(image_id)
            self._images[image_id]["collections"].discard(collection)
            
        elif action == "removed_from_collection":
            collection = details["collection"]
            self._collections[collection].add(image_id)
            self._images[image_id]["collections"].add(collection)
        
        # Remove from image history if it exists
        if image_id in self._images:
            if self._images[image_id]["history"]:
                self._images[image_id]["history"].pop()
        
        return True
    
    # ----- Statistics -----
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the current selections.
        
        Returns:
            Dictionary with selection statistics
        """
        # Calculate additional stats
        selected_pct = 0
        rejected_pct = 0
        unrated_pct = 0
        
        if self._stats["total"] > 0:
            selected_pct = (self._stats["selected"] / self._stats["total"]) * 100
            rejected_pct = (self._stats["rejected"] / self._stats["total"]) * 100
            unrated_pct = (self._stats["unrated"] / self._stats["total"]) * 100
        
        stats = {
            **self._stats,
            "selected_percent": selected_pct,
            "rejected_percent": rejected_pct,
            "unrated_percent": unrated_pct,
            "collections": len(self._collections),
            "history_actions": len(self._history)
        }
        
        return stats
    
    # ----- Serialization -----
    
    def save_to_json(self, filepath: str) -> None:
        """
        Save the selection state to a JSON file.
        
        Args:
            filepath: Path to save the JSON file
        """
        # Prepare data for serialization
        serialized = {
            "project_name": self.project_name,
            "creation_date": self.creation_date.isoformat(),
            "active_collection": self._active_collection,
            "stats": self._stats,
            "collections": {k: list(v) for k, v in self._collections.items()},
            "images": {}
        }
        
        # Serialize image data
        for image_id, info in self._images.items():
            serialized["images"][image_id] = {
                "status": info["status"].value,
                "rating": info["rating"],
                "color_label": info["color_label"].value,
                "flags": list(info["flags"]),
                "collections": list(info["collections"]),
                "added_date": info["added_date"].isoformat(),
                "custom_metadata": info["custom_metadata"]
            }
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(serialized, f, indent=2)
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'SelectionManager':
        """
        Load a SelectionManager from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            New SelectionManager instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Create new instance
        manager = cls(project_name=data["project_name"])
        manager.creation_date = datetime.datetime.fromisoformat(data["creation_date"])
        manager._active_collection = data["active_collection"]
        manager._stats = data["stats"]
        
        # Load collections
        manager._collections = {k: set(v) for k, v in data["collections"].items()}
        
        # Load image data
        for image_id, info in data["images"].items():
            manager._images[image_id] = {
                "status": SelectionStatus(info["status"]),
                "rating": info["rating"],
                "color_label": ColorLabel(info["color_label"]),
                "flags": set(info["flags"]),
                "collections": set(info["collections"]),
                "added_date": datetime.datetime.fromisoformat(info["added_date"]),
                "custom_metadata": info["custom_metadata"],
                "metadata": {},  # Will need to be reloaded
                "history": []    # History isn't preserved
            }
        
        return manager