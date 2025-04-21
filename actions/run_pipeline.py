"""
Pipeline action to run a sequence of actions.
"""
from typing import Dict, Any
import json

from src.action_system import ActionBase, run_pipeline

def run(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a pipeline of actions in sequence.
    
    Parameters:
        --pipeline <file>       Path to pipeline JSON file
        --actions <action1,action2,...>  Comma-separated list of actions
        --params <json>         JSON string with parameters for each action
        
    Example:
        cupcake run-pipeline --actions "analyze-directory,export-selected" --params '{"analyze-directory": {"input-dir": "./photos"}, "export-selected": {"output-dir": "./selected"}}'
    """
    # If pipeline file is specified, use it directly
    if "pipeline" in params and params["pipeline"]:
        return run_pipeline(params["pipeline"])
    
    # Otherwise, build pipeline config from parameters
    actions_str = params.get("actions", "")
    if not actions_str:
        return {"error": "No actions specified"}
        
    actions = actions_str.split(",")
    
    # Parse parameters for each action
    try:
        if "params" in params and params["params"]:
            action_params = json.loads(params["params"])
        else:
            action_params = {}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON in params"}
        
    pipeline_config = {
        "actions": actions,
        "params": action_params
    }
    
    return run_pipeline(pipeline_config)