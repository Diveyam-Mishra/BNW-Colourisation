"""
Data models for the image colorization system.

This module contains dataclasses for structured representation of images
and configuration parameters used throughout the colorization pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional
import numpy as np


@dataclass
class ImageData:
    """
    Structured representation of image data for colorization pipeline.
    
    Attributes:
        original_rgb: Original RGB image array
        lab_image: Full LAB color space representation
        l_channel: Lightness channel (model input)
        ab_channels: Chrominance channels (model target)
        metadata: Image metadata (size, format, etc.)
    """
    original_rgb: np.ndarray
    lab_image: np.ndarray
    l_channel: np.ndarray
    ab_channels: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate ImageData after initialization."""
        self._validate_arrays()
        self._validate_shapes()
    
    def _validate_arrays(self):
        """Validate that all required arrays are numpy arrays."""
        required_arrays = ['original_rgb', 'lab_image', 'l_channel', 'ab_channels']
        for attr_name in required_arrays:
            attr_value = getattr(self, attr_name)
            if not isinstance(attr_value, np.ndarray):
                raise TypeError(f"{attr_name} must be a numpy array, got {type(attr_value)}")
    
    def _validate_shapes(self):
        """Validate that array shapes are consistent."""
        height, width = self.original_rgb.shape[:2]
        
        # Validate original RGB shape
        if len(self.original_rgb.shape) != 3 or self.original_rgb.shape[2] != 3:
            raise ValueError("original_rgb must have shape (H, W, 3)")
        
        # Validate LAB image shape
        if self.lab_image.shape != (height, width, 3):
            raise ValueError(f"lab_image shape {self.lab_image.shape} doesn't match expected {(height, width, 3)}")
        
        # Validate L channel shape
        if self.l_channel.shape != (height, width) and self.l_channel.shape != (height, width, 1):
            raise ValueError(f"l_channel shape {self.l_channel.shape} doesn't match expected {(height, width)} or {(height, width, 1)}")
        
        # Validate AB channels shape
        if self.ab_channels.shape != (height, width, 2):
            raise ValueError(f"ab_channels shape {self.ab_channels.shape} doesn't match expected {(height, width, 2)}")


@dataclass
class ModelConfig:
    """
    Configuration parameters for the U-Net colorization model.
    
    Attributes:
        input_size: Target input image dimensions (height, width)
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        l1_loss_weight: Weight for L1 loss component
        perceptual_loss_weight: Weight for perceptual loss component
        pretrained_encoder: Whether to use pre-trained encoder weights
        checkpoint_frequency: Frequency of model checkpointing (epochs)
    """
    input_size: Tuple[int, int] = (256, 256)
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 100
    l1_loss_weight: float = 1.0
    perceptual_loss_weight: float = 0.1
    pretrained_encoder: bool = True
    checkpoint_frequency: int = 10
    
    def __post_init__(self):
        """Validate ModelConfig after initialization."""
        self.validate()
    
    def validate(self):
        """Validate configuration parameters."""
        # Validate input_size
        if not isinstance(self.input_size, tuple) or len(self.input_size) != 2:
            raise ValueError("input_size must be a tuple of (height, width)")
        
        if not all(isinstance(dim, int) and dim > 0 for dim in self.input_size):
            raise ValueError("input_size dimensions must be positive integers")
        
        # Validate batch_size
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        
        # Validate learning_rate
        if not isinstance(self.learning_rate, (int, float)) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be a positive number")
        
        # Validate num_epochs
        if not isinstance(self.num_epochs, int) or self.num_epochs <= 0:
            raise ValueError("num_epochs must be a positive integer")
        
        # Validate loss weights
        if not isinstance(self.l1_loss_weight, (int, float)) or self.l1_loss_weight < 0:
            raise ValueError("l1_loss_weight must be a non-negative number")
        
        if not isinstance(self.perceptual_loss_weight, (int, float)) or self.perceptual_loss_weight < 0:
            raise ValueError("perceptual_loss_weight must be a non-negative number")
        
        # Validate pretrained_encoder
        if not isinstance(self.pretrained_encoder, bool):
            raise ValueError("pretrained_encoder must be a boolean")
        
        # Validate checkpoint_frequency
        if not isinstance(self.checkpoint_frequency, int) or self.checkpoint_frequency <= 0:
            raise ValueError("checkpoint_frequency must be a positive integer")


@dataclass
class TrainingConfig:
    """
    Configuration parameters for the training pipeline.
    
    Attributes:
        dataset_path: Path to the training dataset
        validation_split: Fraction of data to use for validation
        augmentation_enabled: Whether to enable data augmentation
        early_stopping_patience: Number of epochs to wait before early stopping
        model_save_path: Path to save trained model
        log_frequency: Frequency of logging (batches)
    """
    dataset_path: str
    validation_split: float = 0.2
    augmentation_enabled: bool = True
    early_stopping_patience: int = 15
    model_save_path: str = "checkpoints/model.pth"
    log_frequency: int = 100
    
    def __post_init__(self):
        """Validate TrainingConfig after initialization."""
        self.validate()
    
    def validate(self):
        """Validate configuration parameters."""
        # Validate dataset_path
        if not isinstance(self.dataset_path, str) or not self.dataset_path.strip():
            raise ValueError("dataset_path must be a non-empty string")
        
        # Validate validation_split
        if not isinstance(self.validation_split, (int, float)) or not (0.0 <= self.validation_split <= 1.0):
            raise ValueError("validation_split must be a number between 0.0 and 1.0")
        
        # Validate augmentation_enabled
        if not isinstance(self.augmentation_enabled, bool):
            raise ValueError("augmentation_enabled must be a boolean")
        
        # Validate early_stopping_patience
        if not isinstance(self.early_stopping_patience, int) or self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be a positive integer")
        
        # Validate model_save_path
        if not isinstance(self.model_save_path, str) or not self.model_save_path.strip():
            raise ValueError("model_save_path must be a non-empty string")
        
        # Validate log_frequency
        if not isinstance(self.log_frequency, int) or self.log_frequency <= 0:
            raise ValueError("log_frequency must be a positive integer")