"""
Custom exception hierarchy for the image colorization system.

This module defines specific exception classes for different error types
that can occur during image processing, model operations, and training.
"""

import logging
from typing import Optional, Any, Dict


class ColorizationError(Exception):
    """Base exception class for all colorization-related errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        
        # Log the error when it's created
        logger = logging.getLogger(__name__)
        logger.error(f"ColorizationError: {message}", extra={
            'error_code': error_code,
            'context': context
        })


# Input Validation Errors
class InvalidImageFormatError(ColorizationError):
    """Raised when unsupported image formats are provided."""
    
    def __init__(self, format_type: str, supported_formats: list):
        message = f"Unsupported image format: {format_type}. Supported formats: {supported_formats}"
        super().__init__(message, error_code="INVALID_FORMAT")
        self.format_type = format_type
        self.supported_formats = supported_formats


class ImageCorruptedError(ColorizationError):
    """Raised when image files cannot be properly decoded."""
    
    def __init__(self, image_path: str, details: Optional[str] = None):
        message = f"Image file is corrupted or unreadable: {image_path}"
        if details:
            message += f". Details: {details}"
        super().__init__(message, error_code="IMAGE_CORRUPTED")
        self.image_path = image_path


class InvalidDimensionsError(ColorizationError):
    """Raised when image dimensions are outside acceptable ranges."""
    
    def __init__(self, dimensions: tuple, min_size: tuple, max_size: tuple):
        message = f"Image dimensions {dimensions} are outside acceptable range [{min_size}, {max_size}]"
        super().__init__(message, error_code="INVALID_DIMENSIONS")
        self.dimensions = dimensions
        self.min_size = min_size
        self.max_size = max_size


# Model Errors
class ModelLoadError(ColorizationError):
    """Raised when pre-trained weights cannot be loaded."""
    
    def __init__(self, model_path: str, reason: Optional[str] = None):
        message = f"Failed to load model from: {model_path}"
        if reason:
            message += f". Reason: {reason}"
        super().__init__(message, error_code="MODEL_LOAD_FAILED")
        self.model_path = model_path


class InsufficientMemoryError(ColorizationError):
    """Raised when GPU/CPU memory is insufficient for processing."""
    
    def __init__(self, required_memory: Optional[int] = None, 
                 available_memory: Optional[int] = None, device: str = "unknown"):
        message = f"Insufficient memory on {device}"
        if required_memory and available_memory:
            message += f". Required: {required_memory}MB, Available: {available_memory}MB"
        super().__init__(message, error_code="INSUFFICIENT_MEMORY")
        self.required_memory = required_memory
        self.available_memory = available_memory
        self.device = device


class TrainingConvergenceError(ColorizationError):
    """Raised when training fails to converge."""
    
    def __init__(self, epoch: int, loss_value: float, threshold: float):
        message = f"Training failed to converge at epoch {epoch}. Loss: {loss_value}, Threshold: {threshold}"
        super().__init__(message, error_code="TRAINING_CONVERGENCE_FAILED")
        self.epoch = epoch
        self.loss_value = loss_value
        self.threshold = threshold


# Processing Errors
class ColorSpaceConversionError(ColorizationError):
    """Raised when color space transformations fail."""
    
    def __init__(self, source_space: str, target_space: str, details: Optional[str] = None):
        message = f"Failed to convert from {source_space} to {target_space}"
        if details:
            message += f". Details: {details}"
        super().__init__(message, error_code="COLOR_CONVERSION_FAILED")
        self.source_space = source_space
        self.target_space = target_space


class PostProcessingError(ColorizationError):
    """Raised when output image generation fails."""
    
    def __init__(self, stage: str, details: Optional[str] = None):
        message = f"Post-processing failed at stage: {stage}"
        if details:
            message += f". Details: {details}"
        super().__init__(message, error_code="POST_PROCESSING_FAILED")
        self.stage = stage


# Configuration Errors
class ConfigurationError(ColorizationError):
    """Raised when configuration validation fails."""
    
    def __init__(self, config_key: str, value: Any, expected_type: type):
        message = f"Invalid configuration for '{config_key}': {value}. Expected type: {expected_type.__name__}"
        super().__init__(message, error_code="INVALID_CONFIG")
        self.config_key = config_key
        self.value = value
        self.expected_type = expected_type


# Recovery Strategy Functions
def handle_memory_error(func):
    """Decorator to handle memory errors with automatic retry strategies."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except InsufficientMemoryError as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Memory error in {func.__name__}: {e.message}")
            
            # Attempt recovery strategies
            if len(args) > 0 and hasattr(args[0], 'reduce_batch_size'):
                logger.info("Attempting batch size reduction...")
                args[0].reduce_batch_size()
                try:
                    return func(*args, **kwargs)
                except InsufficientMemoryError:
                    # If still failing, try CPU fallback
                    if hasattr(args[0], 'fallback_to_cpu'):
                        logger.info("Attempting CPU fallback...")
                        args[0].fallback_to_cpu()
                        return func(*args, **kwargs)
                    else:
                        raise
            elif len(args) > 0 and hasattr(args[0], 'fallback_to_cpu'):
                logger.info("Attempting CPU fallback...")
                args[0].fallback_to_cpu()
                return func(*args, **kwargs)
            else:
                raise
    return wrapper


def handle_model_load_error(func):
    """Decorator to handle model loading errors with fallback strategies."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ModelLoadError as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Model load error in {func.__name__}: {e.message}")
            
            # Attempt fallback to random initialization
            if len(args) > 0 and hasattr(args[0], 'initialize_random_weights'):
                logger.info("Falling back to random weight initialization...")
                args[0].initialize_random_weights()
                # Set a flag to indicate weights were loaded (for the retry)
                if hasattr(args[0], 'weights_loaded'):
                    args[0].weights_loaded = True
                return func(*args, **kwargs)
            else:
                raise
    return wrapper


def handle_image_processing_error(func):
    """Decorator to handle image processing errors with retry strategies."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (ImageCorruptedError, InvalidImageFormatError) as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Image processing error in {func.__name__}: {e.message}")
            
            # For batch processing, skip corrupted images and continue
            if hasattr(args[0], 'skip_corrupted_images') and args[0].skip_corrupted_images:
                logger.info("Skipping corrupted image and continuing...")
                return None
            else:
                raise
    return wrapper