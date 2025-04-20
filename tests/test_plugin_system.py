# tests/test_plugin_system.py

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.plugin_system import PluginManager, CupcakePlugin, PluginType, PluginHook

def test_plugin_discovery():
    """Test plugin discovery functionality."""
    print("\n=== Testing Plugin Discovery ===")
    
    # Create a plugin manager
    plugin_dirs = ["plugins"]  # Default plugin directory
    
    # Check if the plugins directory exists
    if not os.path.exists("plugins"):
        print("Plugins directory not found, creating a test directory")
        os.makedirs("plugins", exist_ok=True)
    
    plugin_manager = PluginManager(plugin_dirs)
    
    # Discover plugins
    print("\nDiscovering plugins...")
    available_plugins = plugin_manager.discover_plugins()
    
    print(f"Found {len(available_plugins)} available plugins:")
    for name, plugin_class in available_plugins.items():
        print(f"  - {name} ({plugin_class.plugin_type.value})")
    
    return True

def test_plugin_loading():
    """Test plugin loading functionality."""
    print("\n=== Testing Plugin Loading ===")
    
    # Create a plugin manager
    plugin_manager = PluginManager(["plugins"])
    
    # Discover plugins
    available_plugins = plugin_manager.discover_plugins()
    
    if not available_plugins:
        print("No plugins available for testing")
        return True
    
    # Try to load each plugin
    print("\nLoading plugins...")
    for plugin_name in available_plugins:
        print(f"Loading plugin: {plugin_name}")
        plugin = plugin_manager.load_plugin(plugin_name)
        
        if plugin:
            print(f"  Successfully loaded: {plugin.plugin_name}")
            print(f"  Type: {plugin.plugin_type.value}")
            print(f"  Version: {plugin.plugin_version}")
            print(f"  Author: {plugin.plugin_author}")
            print(f"  Hooks: {[hook.value for hook in plugin.plugin_hooks]}")
        else:
            print(f"  Failed to load plugin: {plugin_name}")
    
    # Get loaded plugins
    loaded_plugins = plugin_manager.get_loaded_plugins()
    print(f"\nLoaded {len(loaded_plugins)} plugins")
    
    # Test unloading
    print("\nUnloading plugins...")
    for plugin_name in list(loaded_plugins.keys()):
        success = plugin_manager.unload_plugin(plugin_name)
        print(f"  Unloaded {plugin_name}: {success}")
    
    # Verify all plugins are unloaded
    loaded_plugins = plugin_manager.get_loaded_plugins()
    print(f"Remaining loaded plugins: {len(loaded_plugins)}")
    
    return True

def test_plugin_hooks():
    """Test plugin hook execution."""
    print("\n=== Testing Plugin Hooks ===")
    
    # Create a plugin manager
    plugin_manager = PluginManager(["plugins"])
    
    # Discover and load plugins
    plugin_manager.discover_plugins()
    plugin_manager.load_all_plugins()
    
    # Get loaded plugins
    loaded_plugins = plugin_manager.get_loaded_plugins()
    
    if not loaded_plugins:
        print("No plugins loaded for testing hooks")
        return True
    
    # Test startup hook
    print("\nExecuting STARTUP hook...")
    results = plugin_manager.execute_hook(PluginHook.STARTUP)
    
    print(f"Hook results:")
    for plugin_name, result in results.items():
        print(f"  {plugin_name}: {result}")
    
    # Test shutdown hook
    print("\nExecuting SHUTDOWN hook...")
    results = plugin_manager.execute_hook(PluginHook.SHUTDOWN)
    
    print(f"Hook results:")
    for plugin_name, result in results.items():
        print(f"  {plugin_name}: {result}")
    
    # Unload all plugins
    plugin_manager.unload_all_plugins()
    
    return True

def test_plugin_template_creation():
    """Test creating a plugin template."""
    print("\n=== Testing Plugin Template Creation ===")
    
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create plugin manager
        plugin_manager = PluginManager()
        
        # Create a template plugin
        print(f"\nCreating template plugin in {temp_dir}...")
        success = plugin_manager.create_plugin_template(
            "Test Plugin",
            temp_dir,
            PluginType.UTILITY
        )
        
        if success:
            print("Successfully created plugin template")
            
            # Check what files were created
            files = os.listdir(temp_dir)
            print(f"Files created: {files}")
            
            # Read and display part of the template
            if files:
                with open(os.path.join(temp_dir, files[0]), 'r') as f:
                    content = f.read()
                    print("\nTemplate preview:")
                    # Show first few lines
                    preview_lines = content.split('\n')[:10]
                    for line in preview_lines:
                        print(f"  {line}")
                    print("  ...")
        else:
            print("Failed to create plugin template")
    
    return True

def test_plugin_configuration():
    """Test plugin configuration management."""
    print("\n=== Testing Plugin Configuration ===")
    
    # Create a plugin manager
    plugin_manager = PluginManager(["plugins"])
    
    # Discover plugins
    plugin_manager.discover_plugins()
    
    # Get all plugin metadata
    metadata = plugin_manager.get_all_plugin_metadata()
    
    print(f"Found {len(metadata)} plugins with metadata:")
    for name, plugin_meta in metadata.items():
        print(f"  - {name} ({plugin_meta.get('type', 'unknown')})")
    
    # Select a plugin to test configuration (if any available)
    if not metadata:
        print("No plugins available for configuration testing")
        return True
    
    # Choose first plugin for testing
    test_plugin_name = list(metadata.keys())[0]
    print(f"\nTesting configuration for plugin: {test_plugin_name}")
    
    # Load the plugin
    plugin = plugin_manager.load_plugin(test_plugin_name)
    
    if not plugin:
        print(f"Failed to load plugin: {test_plugin_name}")
        return False
    
    # Get current configuration
    print("\nCurrent configuration:")
    for key, value in plugin.config.items():
        print(f"  {key}: {value}")
    
    # Get configuration schema
    schema = plugin.get_config_schema()
    print("\nConfiguration schema:")
    for key, details in schema.items():
        print(f"  {key}: {details.get('type', 'unknown')} (default: {details.get('default', 'none')})")
        if 'description' in details:
            print(f"    {details['description']}")
    
    # Test updating configuration
    if schema:
        print("\nTesting configuration update:")
        test_key = list(schema.keys())[0]
        test_value = schema[test_key].get('default')
        
        print(f"Updating {test_key} to {test_value}")
        config_update = {test_key: test_value}
        result = plugin_manager.update_plugin_config(test_plugin_name, config_update)
        
        print(f"Update result: {result}")
        
        # Verify updated configuration
        print("\nVerified configuration:")
        for key, value in plugin.config.items():
            print(f"  {key}: {value}")
    
    # Unload the plugin
    plugin_manager.unload_plugin(test_plugin_name)
    
    return True

def test_ml_style_predictor():
    """Test the LLM Style Predictor plugin if available."""
    print("\n=== Testing LLM Style Predictor Plugin ===")
    
    # Create a plugin manager
    plugin_manager = PluginManager(["plugins"])
    
    # Discover plugins
    available_plugins = plugin_manager.discover_plugins()
    
    # Check if the style predictor plugin is available
    if "LLM Style Predictor" not in available_plugins:
        print("LLM Style Predictor plugin not found, skipping this test")
        return True
    
    print("\nLoading the LLM Style Predictor plugin...")
    plugin = plugin_manager.load_plugin("LLM Style Predictor")
    
    if not plugin:
        print("Failed to load the LLM Style Predictor plugin")
        return False
    
    # Test basic functionality
    print("\nTesting plugin functionality:")
    
    # Load a test image
    from src.image_loader import ImageLoader
    loader = ImageLoader()
    
    test_image_path = input("\nEnter path to a test image (or 'skip' to skip): ")
    
    if test_image_path.lower() == 'skip':
        print("Skipping image style prediction test.")
        plugin_manager.unload_plugin("LLM Style Predictor")
        return True
    
    if not os.path.exists(test_image_path):
        print(f"File doesn't exist: {test_image_path}")
        plugin_manager.unload_plugin("LLM Style Predictor")
        return False
    
    try:
        print(f"Loading test image: {test_image_path}")
        image_data, metadata = loader.load_from_path(test_image_path)
        
        # Perform style prediction
        print("Predicting image style...")
        analysis_results = {}
        style_results = plugin.post_analysis(image_data, analysis_results, metadata)
        
        if style_results:
            print("\nStyle prediction results:")
            
            if "style_predictions" in style_results:
                print("\nStyle scores:")
                for style, score in sorted(style_results["style_predictions"].items(), 
                                          key=lambda x: x[1], reverse=True):
                    print(f"  {style}: {score:.4f}")
            
            if "dominant_styles" in style_results:
                print("\nDominant styles:")
                for style in style_results["dominant_styles"]:
                    print(f"  - {style}")
            
            if "preference_score" in style_results:
                print(f"\nUser preference score: {style_results['preference_score']:.4f}")
        else:
            print("No style prediction results returned")
        
        # Test preference learning
        print("\nTesting preference learning...")
        # Simulate user selecting the image
        learning_result = plugin.learn_preferences(test_image_path, True, style_results)
        print(f"Learning result: {learning_result}")
        
        # Get user preference report
        if hasattr(plugin, "get_user_preference_report"):
            print("\nUser preference report:")
            report = plugin.get_user_preference_report()
            
            if "top_styles" in report:
                print(f"Top styles: {', '.join(report['top_styles'])}")
            
            if "preference_strength" in report:
                print(f"Preference strength: {report['preference_strength']:.4f}")
            
            if "preferences" in report:
                print("\nDetailed preferences:")
                for pref in report["preferences"][:3]:  # Show top 3
                    print(f"  {pref['style']}: {pref['preference']:.4f}")
    
    except Exception as e:
        print(f"Error testing plugin: {e}")
    
    # Unload the plugin
    plugin_manager.unload_plugin("LLM Style Predictor")
    
    return True

def run_tests():
    """Run all tests."""
    print("=== Testing Plugin System ===\n")
    
    tests = [
        ("Plugin Discovery", test_plugin_discovery),
        ("Plugin Loading", test_plugin_loading),
        ("Plugin Hooks", test_plugin_hooks),
        ("Plugin Template Creation", test_plugin_template_creation),
        ("Plugin Configuration", test_plugin_configuration),
        ("LLM Style Predictor", test_ml_style_predictor)
    ]
    
    results = []
    
    for name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running Test: {name}")
        print(f"{'='*50}")
        
        try:
            success = test_func()
            results.append((name, success))
            print(f"\nTest '{name}' {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            print(f"\nTest '{name}' FAILED with exception: {e}")
            results.append((name, False))
    
    # Print summary
    print("\n\n=== Test Summary ===")
    for name, success in results:
        print(f"{name}: {'PASSED' if success else 'FAILED'}")

if __name__ == "__main__":
    run_tests()