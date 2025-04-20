# src/plugin_system.py

import os
import sys
import importlib
import inspect
import pkgutil
import logging
from enum import Enum
from typing import Dict, List, Set, Optional, Tuple, Any, Union, Callable, Type
import json

class PluginType(Enum):
    """Enum representing types of plugins."""
    ANALYSIS = "analysis"      # Image analysis plugins
    RATING = "rating"          # Rating algorithm plugins
    SELECTION = "selection"    # Selection helper plugins
    EXPORT = "export"          # Export enhancement plugins
    UTILITY = "utility"        # General utility plugins
    INTERFACE = "interface"    # Interface plugins
    ML = "ml"                  # Machine learning plugins


class PluginHook(Enum):
    """Enum representing the hooks where plugins can attach."""
    # Image loading hooks
    PRE_LOAD = "pre_load"                  # Before image loading
    POST_LOAD = "post_load"                # After image loading
    
    # Analysis hooks
    PRE_ANALYSIS = "pre_analysis"          # Before image analysis
    POST_ANALYSIS = "post_analysis"        # After image analysis
    
    # Rating hooks
    PRE_RATING = "pre_rating"              # Before rating calculation
    POST_RATING = "post_rating"            # After rating calculation
    LEARN_PREFERENCES = "learn_preferences" # During preference learning
    
    # Selection hooks
    PRE_SELECT = "pre_select"              # Before selection changes
    POST_SELECT = "post_select"            # After selection changes
    
    # Export hooks
    PRE_EXPORT = "pre_export"              # Before export
    POST_EXPORT = "post_export"            # After export
    
    # General hooks
    STARTUP = "startup"                    # On application startup
    SHUTDOWN = "shutdown"                  # On application shutdown


class CupcakePlugin:
    """Base class for Cupcake plugins."""
    
    plugin_name = "Unnamed Plugin"
    plugin_type = PluginType.UTILITY
    plugin_description = "No description provided"
    plugin_version = "0.1.0"
    plugin_author = "Unknown"
    plugin_hooks = []  # List of PluginHook values
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the plugin.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"cupcake.plugin.{self.plugin_name}")
    
    def initialize(self) -> bool:
        """
        Initialize the plugin. Called when the plugin is loaded.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        return True
    
    def shutdown(self) -> bool:
        """
        Shutdown the plugin. Called when the plugin is unloaded.
        
        Returns:
            True if shutdown was successful, False otherwise
        """
        return True
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get the configuration schema for this plugin.
        
        Returns:
            Dictionary with configuration schema
        """
        return {}
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate the configuration for this plugin.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Default implementation doesn't validate anything
        return True, None
    
    def update_config(self, config: Dict[str, Any]) -> bool:
        """
        Update the plugin configuration.
        
        Args:
            config: New configuration
            
        Returns:
            True if configuration was updated successfully
        """
        is_valid, error = self.validate_config(config)
        if not is_valid:
            self.logger.error(f"Invalid configuration: {error}")
            return False
            
        self.config.update(config)
        return True
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get plugin metadata.
        
        Returns:
            Dictionary with plugin metadata
        """
        return {
            "name": self.plugin_name,
            "type": self.plugin_type.value,
            "description": self.plugin_description,
            "version": self.plugin_version,
            "author": self.plugin_author,
            "hooks": [hook.value for hook in self.plugin_hooks]
        }


class PluginManager:
    """Manages the discovery, loading, and execution of plugins."""
    
    def __init__(self, plugin_dirs: Optional[List[str]] = None):
        """
        Initialize the plugin manager.
        
        Args:
            plugin_dirs: List of directories to search for plugins
        """
        self.plugin_dirs = plugin_dirs or ["plugins"]
        self.logger = logging.getLogger("cupcake.plugins")
        
        # Add plugin directories to Python path
        for plugin_dir in self.plugin_dirs:
            if os.path.exists(plugin_dir) and plugin_dir not in sys.path:
                sys.path.append(os.path.abspath(plugin_dir))
        
        # Plugin storage
        self.available_plugins: Dict[str, Type[CupcakePlugin]] = {}
        self.loaded_plugins: Dict[str, CupcakePlugin] = {}
        
        # Hook registrations
        self.hook_registrations: Dict[PluginHook, List[Tuple[str, Callable]]] = {
            hook: [] for hook in PluginHook
        }
        
        # Plugin configurations
        self.config_dir = os.path.join(os.path.expanduser("~"), ".cupcake", "plugin_configs")
        os.makedirs(self.config_dir, exist_ok=True)
    
    def discover_plugins(self) -> Dict[str, Type[CupcakePlugin]]:
        """
        Discover available plugins in plugin directories.
        
        Returns:
            Dictionary of plugin names to plugin classes
        """
        self.available_plugins = {}
        
        for plugin_dir in self.plugin_dirs:
            if not os.path.exists(plugin_dir):
                self.logger.warning(f"Plugin directory not found: {plugin_dir}")
                continue
                
            self.logger.info(f"Searching for plugins in {plugin_dir}")
            
            # Get all Python modules in the plugin directory
            for _, module_name, is_pkg in pkgutil.iter_modules([plugin_dir]):
                if is_pkg:
                    # If it's a package, search for plugins inside
                    try:
                        package = importlib.import_module(module_name)
                        
                        for _, obj in inspect.getmembers(package, inspect.isclass):
                            if (issubclass(obj, CupcakePlugin) and 
                                obj is not CupcakePlugin):
                                self.available_plugins[obj.plugin_name] = obj
                                self.logger.info(f"Discovered plugin: {obj.plugin_name}")
                    except Exception as e:
                        self.logger.error(f"Error discovering plugins in package {module_name}: {e}")
                else:
                    # If it's a module, look for plugin classes directly
                    try:
                        module = importlib.import_module(module_name)
                        
                        for _, obj in inspect.getmembers(module, inspect.isclass):
                            if (issubclass(obj, CupcakePlugin) and 
                                obj is not CupcakePlugin):
                                self.available_plugins[obj.plugin_name] = obj
                                self.logger.info(f"Discovered plugin: {obj.plugin_name}")
                    except Exception as e:
                        self.logger.error(f"Error discovering plugins in module {module_name}: {e}")
        
        return self.available_plugins
    
    def load_plugin(self, plugin_name: str) -> Optional[CupcakePlugin]:
        """
        Load a specific plugin by name.
        
        Args:
            plugin_name: Name of the plugin to load
            
        Returns:
            Loaded plugin instance or None if loading failed
        """
        if plugin_name not in self.available_plugins:
            self.logger.error(f"Plugin not found: {plugin_name}")
            return None
        
        if plugin_name in self.loaded_plugins:
            self.logger.warning(f"Plugin already loaded: {plugin_name}")
            return self.loaded_plugins[plugin_name]
        
        try:
            # Load plugin configuration
            config = self._load_plugin_config(plugin_name)
            
            # Instantiate the plugin
            plugin_class = self.available_plugins[plugin_name]
            plugin = plugin_class(config)
            
            # Initialize the plugin
            if not plugin.initialize():
                self.logger.error(f"Failed to initialize plugin: {plugin_name}")
                return None
            
            # Register hooks
            for hook in plugin.plugin_hooks:
                if hasattr(plugin, hook.value):
                    hook_method = getattr(plugin, hook.value)
                    self.hook_registrations[hook].append((plugin_name, hook_method))
                    self.logger.debug(f"Registered hook {hook.value} for plugin {plugin_name}")
            
            # Store the loaded plugin
            self.loaded_plugins[plugin_name] = plugin
            self.logger.info(f"Loaded plugin: {plugin_name}")
            
            return plugin
            
        except Exception as e:
            self.logger.error(f"Error loading plugin {plugin_name}: {e}")
            return None
    
    def load_all_plugins(self) -> Dict[str, CupcakePlugin]:
        """
        Load all discovered plugins.
        
        Returns:
            Dictionary of loaded plugin names to plugin instances
        """
        # Discover plugins if we haven't already
        if not self.available_plugins:
            self.discover_plugins()
        
        # Load each plugin
        for plugin_name in self.available_plugins:
            self.load_plugin(plugin_name)
        
        return self.loaded_plugins
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a specific plugin by name.
        
        Args:
            plugin_name: Name of the plugin to unload
            
        Returns:
            True if the plugin was unloaded successfully
        """
        if plugin_name not in self.loaded_plugins:
            self.logger.warning(f"Plugin not loaded: {plugin_name}")
            return False
        
        try:
            # Get the plugin
            plugin = self.loaded_plugins[plugin_name]
            
            # Shutdown the plugin
            if not plugin.shutdown():
                self.logger.warning(f"Plugin {plugin_name} reported shutdown failure")
            
            # Unregister hooks
            for hook in PluginHook:
                self.hook_registrations[hook] = [
                    (name, method) for name, method in self.hook_registrations[hook]
                    if name != plugin_name
                ]
            
            # Remove from loaded plugins
            del self.loaded_plugins[plugin_name]
            
            self.logger.info(f"Unloaded plugin: {plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return False
    
    def unload_all_plugins(self) -> bool:
        """
        Unload all loaded plugins.
        
        Returns:
            True if all plugins were unloaded successfully
        """
        success = True
        
        # Make a copy of the keys since we'll be modifying the dictionary
        plugin_names = list(self.loaded_plugins.keys())
        
        for plugin_name in plugin_names:
            if not self.unload_plugin(plugin_name):
                success = False
        
        return success
    
    def execute_hook(self, hook: PluginHook, *args, **kwargs) -> Dict[str, Any]:
        """
        Execute all plugin methods registered for a specific hook.
        
        Args:
            hook: Hook to execute
            *args: Positional arguments to pass to hook methods
            **kwargs: Keyword arguments to pass to hook methods
            
        Returns:
            Dictionary mapping plugin names to hook results
        """
        results = {}
        
        if hook not in self.hook_registrations:
            self.logger.error(f"Invalid hook: {hook}")
            return results
        
        # Execute each registered method
        for plugin_name, method in self.hook_registrations[hook]:
            try:
                results[plugin_name] = method(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Error executing hook {hook.value} for plugin {plugin_name}: {e}")
                results[plugin_name] = None
        
        return results
    
    def get_loaded_plugins(self, plugin_type: Optional[PluginType] = None) -> Dict[str, CupcakePlugin]:
        """
        Get all loaded plugins, optionally filtered by type.
        
        Args:
            plugin_type: Optional type to filter by
            
        Returns:
            Dictionary of plugin names to plugin instances
        """
        if plugin_type is None:
            return self.loaded_plugins
        
        return {
            name: plugin for name, plugin in self.loaded_plugins.items()
            if plugin.plugin_type == plugin_type
        }
    
    def get_plugin_metadata(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific plugin.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Dictionary with plugin metadata or None if not found
        """
        if plugin_name in self.loaded_plugins:
            return self.loaded_plugins[plugin_name].get_metadata()
        
        if plugin_name in self.available_plugins:
            # Create a temporary instance to get metadata
            try:
                plugin = self.available_plugins[plugin_name]()
                return plugin.get_metadata()
            except Exception as e:
                self.logger.error(f"Error getting metadata for plugin {plugin_name}: {e}")
        
        return None
    
    def get_all_plugin_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata for all available plugins.
        
        Returns:
            Dictionary mapping plugin names to metadata dictionaries
        """
        metadata = {}
        
        # Include loaded plugins
        for name, plugin in self.loaded_plugins.items():
            metadata[name] = plugin.get_metadata()
        
        # Include available but not loaded plugins
        for name, plugin_class in self.available_plugins.items():
            if name not in metadata:
                try:
                    plugin = plugin_class()
                    metadata[name] = plugin.get_metadata()
                except Exception as e:
                    self.logger.error(f"Error getting metadata for plugin {name}: {e}")
                    metadata[name] = {
                        "name": name,
                        "error": str(e)
                    }
        
        return metadata
    
    def update_plugin_config(self, plugin_name: str, config: Dict[str, Any]) -> bool:
        """
        Update the configuration for a specific plugin.
        
        Args:
            plugin_name: Name of the plugin
            config: New configuration dictionary
            
        Returns:
            True if the configuration was updated successfully
        """
        if plugin_name in self.loaded_plugins:
            plugin = self.loaded_plugins[plugin_name]
            if plugin.update_config(config):
                # Save the updated configuration
                self._save_plugin_config(plugin_name, plugin.config)
                return True
        
        return False
    
    def _load_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """
        Load configuration for a specific plugin.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Configuration dictionary
        """
        config_path = os.path.join(self.config_dir, f"{plugin_name}.json")
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading config for plugin {plugin_name}: {e}")
        
        return {}
    
    def _save_plugin_config(self, plugin_name: str, config: Dict[str, Any]) -> bool:
        """
        Save configuration for a specific plugin.
        
        Args:
            plugin_name: Name of the plugin
            config: Configuration to save
            
        Returns:
            True if the configuration was saved successfully
        """
        config_path = os.path.join(self.config_dir, f"{plugin_name}.json")
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Error saving config for plugin {plugin_name}: {e}")
            return False
    
    def create_plugin_template(self, plugin_name: str, output_dir: str, plugin_type: PluginType) -> bool:
        """
        Create a template for a new plugin.
        
        Args:
            plugin_name: Name of the new plugin
            output_dir: Directory to create the plugin in
            plugin_type: Type of plugin to create
            
        Returns:
            True if the template was created successfully
        """
        # Clean up the plugin name
        cleaned_name = ''.join(c for c in plugin_name if c.isalnum() or c == '_')
        if not cleaned_name:
            self.logger.error(f"Invalid plugin name: {plugin_name}")
            return False
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create the plugin file
        file_path = os.path.join(output_dir, f"{cleaned_name}.py")
        
        if os.path.exists(file_path):
            self.logger.error(f"Plugin file already exists: {file_path}")
            return False
        
        try:
            with open(file_path, 'w') as f:
                # Write the initial comments and imports
                f.write("# " + cleaned_name + ".py\n")
                f.write("# Cupcake Photo Culling Library Plugin\n\n")
                f.write("from src.plugin_system import CupcakePlugin, PluginType, PluginHook\n")
                f.write("from typing import Dict, Any, Optional, Tuple\n\n")
                
                # Write the class definition
                f.write("class " + cleaned_name.capitalize() + "Plugin(CupcakePlugin):\n")
                f.write("    \"\"\"\n")
                f.write("    " + plugin_name + " - A plugin for Cupcake Photo Culling Library\n")
                f.write("    \"\"\"\n\n")
                
                # Write class attributes
                f.write("    plugin_name = \"" + plugin_name + "\"\n")
                f.write("    plugin_type = PluginType." + plugin_type.name + "\n")
                f.write("    plugin_description = \"Your plugin description here\"\n")
                f.write("    plugin_version = \"0.1.0\"\n")
                f.write("    plugin_author = \"Your Name\"\n")
                f.write("    plugin_hooks = [\n")
                f.write("        # Add the hooks you want to implement\n")
                f.write("        # PluginHook.STARTUP,\n")
                f.write("        # PluginHook.SHUTDOWN,\n")
                f.write("        # ...\n")
                f.write("    ]\n\n")
                
                # Write __init__ method
                f.write("    def __init__(self, config: Optional[Dict[str, Any]] = None):\n")
                f.write("        super().__init__(config)\n")
                f.write("        # Add your initialization code here\n\n")
                
                # Write initialize method
                f.write("    def initialize(self) -> bool:\n")
                f.write("        \"\"\"\n")
                f.write("        Initialize the plugin. Called when the plugin is loaded.\n")
                f.write("        \"\"\"\n")
                f.write("        self.logger.info(f\"Initializing {self.plugin_name}\")\n")
                f.write("        # Add your initialization code here\n")
                f.write("        return True\n\n")
                
                # Write shutdown method
                f.write("    def shutdown(self) -> bool:\n")
                f.write("        \"\"\"\n")
                f.write("        Shutdown the plugin. Called when the plugin is unloaded.\n")
                f.write("        \"\"\"\n")
                f.write("        self.logger.info(f\"Shutting down {self.plugin_name}\")\n")
                f.write("        # Add your cleanup code here\n")
                f.write("        return True\n\n")
                
                # Write config schema method
                f.write("    def get_config_schema(self) -> Dict[str, Any]:\n")
                f.write("        \"\"\"\n")
                f.write("        Get the configuration schema for this plugin.\n")
                f.write("        \"\"\"\n")
                f.write("        return {\n")
                f.write("            \"option1\": {\n")
                f.write("                \"type\": \"string\",\n")
                f.write("                \"default\": \"default value\",\n")
                f.write("                \"description\": \"Description of option1\"\n")
                f.write("            },\n")
                f.write("            \"option2\": {\n")
                f.write("                \"type\": \"integer\",\n")
                f.write("                \"default\": 42,\n")
                f.write("                \"description\": \"Description of option2\"\n")
                f.write("            }\n")
                f.write("            # Add more configuration options as needed\n")
                f.write("        }\n\n")
                
                # Write validate config method
                f.write("    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:\n")
                f.write("        \"\"\"\n")
                f.write("        Validate the configuration for this plugin.\n")
                f.write("        \"\"\"\n")
                f.write("        # Add your configuration validation code here\n")
                f.write("        # Return (True, None) if the configuration is valid\n")
                f.write("        # Return (False, \"Error message\") if the configuration is invalid\n")
                f.write("        return True, None\n\n")
                
                # Write hook implementation examples
                f.write("    # Add your hook implementations here\n")
                f.write("    # def startup(self):\n")
                f.write("    #     self.logger.info(\"Plugin startup\")\n")
                f.write("    #     return True\n")
                f.write("    #\n")
                f.write("    # def shutdown(self):\n")
                f.write("    #     self.logger.info(\"Plugin shutdown\")\n")
                f.write("    #     return True\n")
            
            self.logger.info(f"Created plugin template at {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating plugin template: {e}")
            return False