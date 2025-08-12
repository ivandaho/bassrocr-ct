import yaml
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yml')

def load_config():
    """Loads the YAML configuration file."""
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Configuration file not found at: {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def get_statement_config(statement_type):
    """
    Gets the configuration for a specific statement type, falling back to defaults.
    """
    config = load_config()
    default_config = config.get('default', {})
    statement_config = config.get(statement_type, {})

    # Merge the statement-specific config into the default config
    final_config = default_config.copy()
    final_config.update(statement_config)
    
    return final_config
