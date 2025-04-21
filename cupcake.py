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
    return [f[:-3].replace('_', '-') for f in os.listdir(ACTIONS_DIR)
            if f.endswith(".py") and not f.startswith("_") and f != "__init__.py"]


def load_action_module(action_name):
    module_name = action_name.replace('-', '_')
    try:
        return importlib.import_module(f"actions.{module_name}")
    except ModuleNotFoundError as e:
        logger.error(f"Failed to import action module: {e}")
        return None


def print_action_help(module):
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
    print("\nAvailable actions with help:\n")
    for action in list_available_actions():
        module = load_action_module(action)
        if module:
            print(f"-- {action} --")
            print_action_help(module)
            print("\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: cupcake <action> [options]\n")
        print("Available actions:")
        for action in list_available_actions():
            print(f"  {action}")
        sys.exit(1)

    action = sys.argv[1].replace('_', '-')

    if action == "list" or action == "help" or action == "--help":
        list_all_action_help()
        sys.exit(0)

    action_module = load_action_module(action)

    if not action_module:
        print(f"Unknown action: '{action}'\n")
        print("Available actions:")
        for a in list_available_actions():
            print(f"  {a}")
        sys.exit(1)

    if len(sys.argv) == 2 or sys.argv[2].startswith("--help"):
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
        else:
            if key:
                if args_dict[key] is True:
                    args_dict[key] = item
                else:
                    args_dict[key] += f" {item}"

    logger.info(f"Running action '{action}' with params: {args_dict}")

    try:
        result = action_module.run(args_dict)
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        print_action_help(action_module)
        sys.exit(1)

    print("\n== Results ==")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
