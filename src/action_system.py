"""
Action system for Cupcake Photo Culling Library.
Provides base classes and utilities for implementing actions.
"""
import os
import importlib
import inspect
import json
from typing import Dict, Any, List, Callable, Type, Optional, Union

class ActionBase:
    """Base class for all Cupcake actions."""
    
    @classmethod
    def run(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the action with the given parameters.
        
        Args:
            params: Dictionary of parameters
            
        Returns:
            Dictionary of results
        """
        raise NotImplementedError("Subclasses must implement run()")
    
    @classmethod
    def get_help(cls) -> str:
        """Get help text for this action."""
        return cls.run.__doc__ or "No help available."


def get_actions_dir() -> str:
    """Get the directory containing action modules."""
    # Assuming the actions directory is at the same level as src
    src_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(src_dir), "actions")


def list_action_modules() -> List[str]:
    """List all action module names without file extensions."""
    actions_dir = get_actions_dir()
    modules = []
    
    for filename in os.listdir(actions_dir):
        if (filename.endswith('.py') and 
            not filename.startswith('_') and 
            filename != '__init__.py'):
            modules.append(filename[:-3])
    
    return modules


def get_action_runner(action_name: str) -> Optional[Callable]:
    """Get the run function for an action."""
    module_name = action_name.replace('-', '_')
    
    try:
        module = importlib.import_module(f"actions.{module_name}")
        
        # Look for action class
        for name, obj in inspect.getmembers(module):
            if (isinstance(obj, type) and 
                issubclass(obj, ActionBase) and
                obj is not ActionBase):
                return obj.run
        
        # Fall back to traditional function
        if hasattr(module, "run"):
            return module.run
        
        return None
    except ModuleNotFoundError:
        return None


def run_pipeline(pipeline_config: Union[str, Dict]) -> Dict[str, Any]:
    """
    Run a pipeline of actions in sequence.
    
    Args:
        pipeline_config: Either a path to a JSON file or a dictionary with
                         'actions' and 'params' keys
    
    Returns:
        Combined results from all actions
    """
    # Load pipeline configuration
    if isinstance(pipeline_config, str):
        # It's a file path
        with open(pipeline_config, 'r') as f:
            pipeline_data = json.load(f)
    else:
        # It's already a dictionary
        pipeline_data = pipeline_config
    
    if not isinstance(pipeline_data, dict):
        raise ValueError("Pipeline config must be a dictionary")
    
    if "actions" not in pipeline_data:
        raise ValueError("Pipeline config must contain 'actions' key")
    
    # Get action list and params
    actions = pipeline_data["actions"]
    action_params = pipeline_data.get("params", {})
    
    # Run each action in sequence
    results = {}
    for action_name in actions:
        # Get parameters for this action
        params = action_params.get(action_name, {})
        
        # Merge with previous results if requested
        if params.get("use_previous_results", False):
            params.update(results)
        
        # Get and run the action
        action_runner = get_action_runner(action_name)
        if action_runner:
            try:
                action_result = action_runner(params)
                results[action_name] = action_result
            except Exception as e:
                results[action_name] = {"error": f"Error in action {action_name}: {str(e)}"}
        else:
            results[action_name] = {"error": f"Action {action_name} not found"}
    
    return results