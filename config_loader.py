import yaml
import os

CONFIG_PATH = "config.yaml"

def load_config(path: str) -> dict:
    """Load and parse YAML config with environment variable substitution."""
    with open(path, "r") as file:
        raw_content = file.read()

    # Substitute environment variables like ${VAR_NAME}
    expanded_content = os.path.expandvars(raw_content)
    config = yaml.safe_load(expanded_content)
    return config

def load_default_config() -> dict:
    return load_config(CONFIG_PATH)