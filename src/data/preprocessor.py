"""
Image preprocessing module for the colorization system.

This module provides the ImagePreprocessor class that handles image loading,
validation, resizing, and normalization for the colorization pipeline.
"""

import os
from pathlib import Path
from typing import Tuple, Union, Optional
import numpy as np
import cv2
from PIL import Image
import logging

# Configure logging
logger = logging.getLogger(__name__)


class ImageProcessingError(Exception):
    """Base exception for image processing errors."""
    pass


class InvalidImageFormat(ImageProcessingError):
    """Raised when an unsupported image format is encountered."""
    pass


class ImageCorrupted(ImageProcessingError):
    """Raised when an image file is corrupted or cannot be decoded."""
    pass


class InvalidDimensions(ImageProcessingError):
    """Raised when image dimensions are outside acceptable ranges."""
    pass


class ImagePreprocessor:
    """
    Handles image preprocessing for the colorization pipeline.
    
    This class provides methods for loading, validating, resizing, and normalizing
    images according to the requirements of the colorization model.
    """
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
    MIN_DIMENSION = 32
    MAX_DIMENSION = 4096
    
    def __init__(self, target_size: Optional[Tuple[int, int]] = None):
        """
        Initialize the ImagePreprocessor.
        
        Args:
            target_size: Optional target size for resizing images (height, width).
                        If None, images will not be resized by default.
        """
        self.target_size = target_size
        
    def validate_image(self, image_path: Union[str, Path]) -> bool:
        """
        Validate if an image file is supported and accessible.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            bool: True if the image is valid and supported
            
        Raises:
            InvalidImageFormat: If the image format is not supported
            ImageCorrupted: If the image file is corrupted or inaccessible
        """
        image_path = Path(image_path)
        
        # Check if file exists
        if not image_path.exists():
            raise ImageCorrupted(f"Image file not found: {image_path}")
            
        # Check file extension
        if image_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise InvalidImageFormat(
                f"Unsupported image format: {image_path.suffix}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )
            
        # Try to open and validate the image
        try:
            with Image.open(image_path) as img:
                # Verify the image can be loaded
                img.verify()
                
            # Re-open to get dimensions (verify() closes the image)
            with Image.open(image_path) as img:
                width, height = img.size
                
                # Check dimensions
                if width < self.MIN_DIMENSION or height < self.MIN_DIMENSION:
                    raise InvalidDimensions(
                        f"Image dimensions too small: {width}x{height}. "
                        f"Minimum dimension: {self.MIN_DIMENSION}"
                    )
                    
                if width > self.MAX_DIMENSION or height > self.MAX_DIMENSION:
                    raise InvalidDimensions(
                        f"Image dimensions too large: {width}x{height}. "
                        f"Maximum dimension: {self.MAX_DIMENSION}"
                    )
                    
        except (IOError, OSError) as e:
            raise ImageCorrupted(f"Cannot read image file {image_path}: {str(e)}")
            
        return True
        
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load an image from file and convert to RGB format.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            np.ndarray: Image array in RGB format with shape (H, W, 3)
            
        Raises:
            InvalidImageFormat: If the image format is not supported
            ImageCorrupted: If the image file is corrupted
        """
        # Validate the image first
        self.validate_image(image_path)
        
        try:
            # Load image using OpenCV (loads as BGR)
            image = cv2.imread(str(image_path))
            
            if image is None:
                raise ImageCorrupted(f"OpenCV failed to load image: {image_path}")
                
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            logger.info(f"Successfully loaded image: {image_path}, shape: {image_rgb.shape}")
            return image_rgb
            
        except cv2.error as e:
            raise ImageCorrupted(f"OpenCV error loading {image_path}: {str(e)}")
        except Exception as e:
            raise ImageCorrupted(f"Unexpected error loading {image_path}: {str(e)}")
            
    def resize_image(self, image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Resize an image while preserving aspect ratio.
        
        Args:
            image: Input image array with shape (H, W, 3)
            target_size: Target size as (height, width). If None, uses self.target_size
            
        Returns:
            np.ndarray: Resized image array
            
        Raises:
            InvalidDimensions: If target_size is invalid
        """
        if target_size is None:
            target_size = self.target_size
            
        if target_size is None:
            # No resizing requested
            return image.copy()
            
        target_height, target_width = target_size
        
        if target_height <= 0 or target_width <= 0:
            raise InvalidDimensions(f"Invalid target size: {target_size}")
            
        current_height, current_width = image.shape[:2]
        
        # Calculate scaling factor to preserve aspect ratio
        scale_height = target_height / current_height
        scale_width = target_width / current_width
        scale = min(scale_height, scale_width)
        
        # Calculate new dimensions
        new_width = int(current_width * scale)
        new_height = int(current_height * scale)
        
        # Resize the image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Create a canvas with target size and center the resized image
        canvas = np.zeros((target_height, target_width, 3), dtype=image.dtype)
        
        # Calculate padding to center the image
        pad_y = (target_height - new_height) // 2
        pad_x = (target_width - new_width) // 2
        
        # Place the resized image on the canvas
        canvas[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized
        
        logger.debug(f"Resized image from {current_width}x{current_height} to {target_width}x{target_height}")
        return canvas
        
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image pixel values to the range [0, 1].
        
        Args:
            image: Input image array with values typically in range [0, 255]
            
        Returns:
            np.ndarray: Normalized image array with values in range [0, 1]
        """
        if image.dtype != np.uint8:
            logger.warning(f"Expected uint8 image, got {image.dtype}")
            
        # Convert to float32 and normalize to [0, 1]
        normalized = image.astype(np.float32) / 255.0
        
        # Ensure values are in valid range
        normalized = np.clip(normalized, 0.0, 1.0)
        
        logger.debug(f"Normalized image: min={normalized.min():.3f}, max={normalized.max():.3f}")
        return normalized
        
    def preprocess_image(self, image_path: Union[str, Path], 
                        target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Complete preprocessing pipeline: load, resize, and normalize an image.
        
        Args:
            image_path: Path to the image file
            target_size: Optional target size for resizing
            
        Returns:
            np.ndarray: Preprocessed image ready for model input
        """
        # Load the image
        image = self.load_image(image_path)
        
        # Resize if needed
        if target_size is not None or self.target_size is not None:
            image = self.resize_image(image, target_size)
            
        # Normalize
        image = self.normalize_image(image)
        
        return image