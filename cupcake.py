import sys
import importlib
import logging
import os
import inspect
import argparse

# Ensure current directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("cupcake")

ACTIONS_DIR = os.path.join(os.path.dirname(__file__), "actions")


def list_available_actions():
    """List all available actions by scanning the actions directory."""
    return [f[:-3].replace('_', '-') for f in os.listdir(ACTIONS_DIR)
            if f.endswith(".py") and not f.startswith("_") and f != "__init__.py"]


def load_action_module(action_name):
    """Load the module for the specified action."""
    module_name = action_name.replace('-', '_')
    try:
        return importlib.import_module(f"actions.{module_name}")
    except ModuleNotFoundError as e:
        logger.error(f"Failed to import action module: {e}")
        return None


def print_action_help(module):
    """Print help for the specified action module."""
    if hasattr(module, "run"):
        doc = inspect.getdoc(module.run)
    else:
        logger.warning(f"No 'run' function found in module '{module.__name__}'")
        doc = None

    print("\nSyntax:")
    if doc:
        print(doc)
    else:
        print("No help available for this action.")


def list_all_action_help():
    """List all available actions with their help text."""
    print("\nAvailable actions with help:\n")
    for action in list_available_actions():
        module = load_action_module(action)
        if module:
            print(f"-- {action} --")
            print_action_help(module)
            print("\n")


def main():
    """Main entry point for the Cupcake CLI."""
    if len(sys.argv) < 2:
        print("Usage: cupcake <action> [options]\n")
        print("Available actions:")
        for action in list_available_actions():
            print(f"  {action}")
        print("\nFor help on a specific action, run: cupcake <action> --help")
        print("For help on all actions, run: cupcake help")
        sys.exit(1)

    action = sys.argv[1].replace('_', '-')

    if action in ["list", "help", "--help", "-h"]:
        list_all_action_help()
        sys.exit(0)

    action_module = load_action_module(action)

    if not action_module:
        print(f"Unknown action: '{action}'\n")
        print("Available actions:")
        for a in list_available_actions():
            print(f"  {a}")
        sys.exit(1)

    if len(sys.argv) == 2 or sys.argv[2] in ["--help", "-h"]:
        print(f"== Help for action '{action}' ==")
        print_action_help(action_module)
        sys.exit(0)

    # Parse CLI args as key-value pairs
    args_dict = {}
    key = None
    for item in sys.argv[2:]:
        if item.startswith("--"):
            key = item[2:].replace('-', '_')
            args_dict[key] = True
        elif item.startswith("-"):
            # Handle short options
            key = item[1:].replace('-', '_')
            args_dict[key] = True
        else:
            if key:
                if args_dict[key] is True:
                    args_dict[key] = item
                else:
                    args_dict[key] += f" {item}"

    logger.info(f"Running action '{action}' with params: {args_dict}")

    try:
        if not hasattr(action_module, "run"):
            raise AttributeError(f"Module {action_module.__name__} has no 'run' function")
            
        result = action_module.run(args_dict)
        
        if not result:
            print("\n✅ Action completed successfully with no results to display.")
        elif isinstance(result, dict):
            print("\n== Results ==")
            for k, v in result.items():
                print(f"{k}: {v}")
        else:
            print(f"\n✅ Result: {result}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print_action_help(action_module)
        sys.exit(1)


if __name__ == "__main__":
    main()