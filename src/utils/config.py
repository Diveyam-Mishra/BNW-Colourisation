"""
Configuration management utilities for the image colorization system.

This module provides functions to load, validate, and manage configuration
files in JSON and YAML formats for model and training configurations.
"""

import json
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Union, Type, TypeVar
from src.data.models import ModelConfig, TrainingConfig

# Type variable for configuration classes
ConfigType = TypeVar('ConfigType', ModelConfig, TrainingConfig)


class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass


def load_json_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        file_path: Path to the JSON configuration file
        
    Returns:
        Dictionary containing configuration data
        
    Raises:
        ConfigurationError: If file cannot be loaded or parsed
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ConfigurationError(f"Configuration file not found: {file_path}")
        
        if not file_path.suffix.lower() == '.json':
            raise ConfigurationError(f"Expected JSON file, got: {file_path.suffix}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            
        return config_data
        
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON format in {file_path}: {e}")
    except IOError as e:
        raise ConfigurationError(f"Error reading file {file_path}: {e}")


def load_yaml_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        file_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration data
        
    Raises:
        ConfigurationError: If file cannot be loaded or parsed
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ConfigurationError(f"Configuration file not found: {file_path}")
        
        if not file_path.suffix.lower() in ['.yaml', '.yml']:
            raise ConfigurationError(f"Expected YAML file, got: {file_path.suffix}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            
        if config_data is None:
            config_data = {}
            
        return config_data
        
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML format in {file_path}: {e}")
    except IOError as e:
        raise ConfigurationError(f"Error reading file {file_path}: {e}")


def load_config_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a file, automatically detecting JSON or YAML format.
    
    Args:
        file_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration data
        
    Raises:
        ConfigurationError: If file format is unsupported or cannot be loaded
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    
    if suffix == '.json':
        return load_json_config(file_path)
    elif suffix in ['.yaml', '.yml']:
        return load_yaml_config(file_path)
    else:
        raise ConfigurationError(f"Unsupported configuration file format: {suffix}")


def save_json_config(config_data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Save configuration data to a JSON file.
    
    Args:
        config_data: Configuration dictionary to save
        file_path: Path where to save the JSON file
        
    Raises:
        ConfigurationError: If file cannot be written
    """
    try:
        file_path = Path(file_path)
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
            
    except IOError as e:
        raise ConfigurationError(f"Error writing file {file_path}: {e}")


def save_yaml_config(config_data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Save configuration data to a YAML file.
    
    Args:
        config_data: Configuration dictionary to save
        file_path: Path where to save the YAML file
        
    Raises:
        ConfigurationError: If file cannot be written
    """
    try:
        file_path = Path(file_path)
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
    except IOError as e:
        raise ConfigurationError(f"Error writing file {file_path}: {e}")


def create_model_config_from_dict(config_dict: Dict[str, Any]) -> ModelConfig:
    """
    Create a ModelConfig instance from a dictionary.
    
    Args:
        config_dict: Dictionary containing model configuration parameters
        
    Returns:
        ModelConfig instance
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    try:
        # Filter only valid ModelConfig parameters
        valid_params = {}
        model_config_fields = {field.name for field in ModelConfig.__dataclass_fields__.values()}
        
        for key, value in config_dict.items():
            if key in model_config_fields:
                # Convert input_size from list to tuple if needed
                if key == 'input_size' and isinstance(value, list):
                    value = tuple(value)
                valid_params[key] = value
        
        return ModelConfig(**valid_params)
        
    except (TypeError, ValueError) as e:
        raise ConfigurationError(f"Invalid model configuration: {e}")


def create_training_config_from_dict(config_dict: Dict[str, Any]) -> TrainingConfig:
    """
    Create a TrainingConfig instance from a dictionary.
    
    Args:
        config_dict: Dictionary containing training configuration parameters
        
    Returns:
        TrainingConfig instance
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    try:
        # Filter only valid TrainingConfig parameters
        valid_params = {}
        training_config_fields = {field.name for field in TrainingConfig.__dataclass_fields__.values()}
        
        for key, value in config_dict.items():
            if key in training_config_fields:
                valid_params[key] = value
        
        return TrainingConfig(**valid_params)
        
    except (TypeError, ValueError) as e:
        raise ConfigurationError(f"Invalid training configuration: {e}")


def load_model_config(file_path: Union[str, Path]) -> ModelConfig:
    """
    Load ModelConfig from a configuration file.
    
    Args:
        file_path: Path to the configuration file
        
    Returns:
        ModelConfig instance
        
    Raises:
        ConfigurationError: If file cannot be loaded or configuration is invalid
    """
    config_dict = load_config_file(file_path)
    return create_model_config_from_dict(config_dict)


def load_training_config(file_path: Union[str, Path]) -> TrainingConfig:
    """
    Load TrainingConfig from a configuration file.
    
    Args:
        file_path: Path to the configuration file
        
    Returns:
        TrainingConfig instance
        
    Raises:
        ConfigurationError: If file cannot be loaded or configuration is invalid
    """
    config_dict = load_config_file(file_path)
    return create_training_config_from_dict(config_dict)


def validate_config_file(file_path: Union[str, Path], config_type: Type[ConfigType]) -> bool:
    """
    Validate a configuration file against a specific configuration type.
    
    Args:
        file_path: Path to the configuration file
        config_type: Type of configuration to validate against (ModelConfig or TrainingConfig)
        
    Returns:
        True if configuration is valid
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    try:
        config_dict = load_config_file(file_path)
        
        if config_type == ModelConfig:
            create_model_config_from_dict(config_dict)
        elif config_type == TrainingConfig:
            create_training_config_from_dict(config_dict)
        else:
            raise ConfigurationError(f"Unsupported configuration type: {config_type}")
        
        return True
        
    except ConfigurationError:
        raise
    except Exception as e:
        raise ConfigurationError(f"Validation failed: {e}")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override_config taking precedence.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()
    merged.update(override_config)
    return merged