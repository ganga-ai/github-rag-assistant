import tomli
from pathlib import Path
from typing import Dict, Any


def load_config() -> Dict[str, Any]:
    """Load configuration from config.toml file."""
    config_path = Path(__file__).parent.parent.parent.parent / "config.toml"
    
    with open(config_path, "rb") as f:
        config = tomli.load(f)
    
    return config


def get_model_config() -> Dict[str, str]:
    """Get model configuration."""
    config = load_config()
    return config.get("models", {})


def get_chunking_config() -> Dict[str, int]:
    """Get chunking configuration."""
    config = load_config()
    return config.get("chunking", {})


def get_filtering_config() -> Dict[str, Any]:
    """Get file filtering configuration."""
    config = load_config()
    return config.get("filtering", {})


def get_vector_store_config() -> Dict[str, Any]:
    """Get vector store configuration."""
    config = load_config()
    return config.get("vector_store", {})