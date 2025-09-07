# -*- coding: utf-8 -*-
"""
Post-processing utilities for image colorization inference.

This module provides utilities for reconstructing output images from predicted
channels, color space conversion back to RGB, and output image saving with
various format handling options.
"""

import os
import logging
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any, List
import numpy as np
from PIL import Image, ImageEnhance
import torch

# Import dependencies with absolute imports
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.utils.color_converter import ColorSpaceConverter
from src.data.models import ImageData

logger = logging.getLogger(__name__)


class PostProcessingError(Exception):
    """Base exception for post-processing errors."""
    pass


class ImageReconstructionError(PostProcessingError):
    """Raised when image reconstruction fails."""
    pass


class ColorSpaceConversionError(PostProcessingError):
    """Raised when color space conversion fails."""
    pass


class ImageSaveError(PostProcessingError):
    """Raised when image saving fails."""
    pass


class ColorizationPostProcessor:
    """
    Post-processing utilities for image colorization inference.
    
    This class handles the reconstruction of colorized images from model predictions,
    color space conversions, and saving output images in various formats.
    """
    
    def __init__(self, 
                 color_converter: Optional[ColorSpaceConverter] = None,
                 output_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the post-processor.
        
        Args:
            color_converter: ColorSpaceConverter instance for color space operations
            output_config: Configuration for output processing
        """
        self.color_converter = color_converter or ColorSpaceConverter()
        self.output_config = output_config or self._get_default_config()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default output configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            'output_format': 'jpg',
            'output_quality': 95,
            'enable_enhancement': True,
            'saturation_boost': 1.1,
            'contrast_boost': 1.05,
            'brightness_adjustment': 1.0,
            'enable_color_correction': True,
            'gamma_correction': 1.0,
            'preserve_original_size': True,
            'interpolation_method': 'lanczos'
        }
    
    def reconstruct_image(self, 
                         l_channel: np.ndarray, 
                         ab_predictions: Union[np.ndarray, torch.Tensor],
                         original_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Reconstruct colorized image from L channel and predicted AB channels.
        
        Args:
            l_channel: Original lightness channel with shape (H, W) or (H, W, 1)
            ab_predictions: Predicted AB channels with shape (H, W, 2) or (2, H, W)
            original_size: Optional original image size (height, width) for resizing
            
        Returns:
            Reconstructed RGB image as numpy array
            
        Raises:
            ImageReconstructionError: If reconstruction fails
        """
        try:
            # Convert tensor to numpy if needed
            if isinstance(ab_predictions, torch.Tensor):
                ab_predictions = ab_predictions.detach().cpu().numpy()
            
            # Handle different input shapes
            l_channel = self._normalize_l_channel(l_channel)
            ab_predictions = self._normalize_ab_predictions(ab_predictions)
            
            # Validate shapes match
            if l_channel.shape[:2] != ab_predictions.shape[:2]:
                raise ImageReconstructionError(
                    f"Shape mismatch: L channel {l_channel.shape[:2]} vs AB predictions {ab_predictions.shape[:2]}"
                )
            
            # Combine L and AB channels
            lab_image = self.color_converter.combine_lab_channels(l_channel, ab_predictions)
            
            # Convert to RGB
            rgb_image = self.color_converter.lab_to_rgb(lab_image)
            
            # Resize to original size if specified
            if original_size and self.output_config['preserve_original_size']:
                rgb_image = self._resize_image(rgb_image, original_size)
            
            # Apply enhancements if enabled
            if self.output_config['enable_enhancement']:
                rgb_image = self._enhance_image(rgb_image)
            
            # Apply color correction if enabled
            if self.output_config['enable_color_correction']:
                rgb_image = self._apply_color_correction(rgb_image)
            
            return rgb_image
            
        except Exception as e:
            raise ImageReconstructionError(f"Failed to reconstruct image: {str(e)}")
    
    def _normalize_l_channel(self, l_channel: np.ndarray) -> np.ndarray:
        """
        Normalize L channel to expected shape and format.
        
        Args:
            l_channel: L channel array
            
        Returns:
            Normalized L channel with shape (H, W)
        """
        if len(l_channel.shape) == 3 and l_channel.shape[2] == 1:
            l_channel = l_channel[:, :, 0]
        elif len(l_channel.shape) != 2:
            raise ImageReconstructionError(f"Invalid L channel shape: {l_channel.shape}")
        
        return l_channel.astype(np.float32)
    
    def _normalize_ab_predictions(self, ab_predictions: np.ndarray) -> np.ndarray:
        """
        Normalize AB predictions to expected shape and format.
        
        Args:
            ab_predictions: AB channel predictions
            
        Returns:
            Normalized AB predictions with shape (H, W, 2)
        """
        # Handle different input shapes
        if len(ab_predictions.shape) == 3:
            if ab_predictions.shape[0] == 2:  # (2, H, W) -> (H, W, 2)
                ab_predictions = np.transpose(ab_predictions, (1, 2, 0))
            elif ab_predictions.shape[2] != 2:
                raise ImageReconstructionError(f"Invalid AB predictions shape: {ab_predictions.shape}")
        else:
            raise ImageReconstructionError(f"Invalid AB predictions shape: {ab_predictions.shape}")
        
        return ab_predictions.astype(np.float32)
    
    def convert_to_rgb(self, lab_image: np.ndarray) -> np.ndarray:
        """
        Convert LAB image to RGB color space for display.
        
        Args:
            lab_image: Input LAB image with shape (H, W, 3)
            
        Returns:
            RGB image as numpy array with values in [0, 255]
            
        Raises:
            ColorSpaceConversionError: If conversion fails
        """
        try:
            rgb_image = self.color_converter.lab_to_rgb(lab_image)
            
            # Ensure proper data type and range
            rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
            
            return rgb_image
            
        except Exception as e:
            raise ColorSpaceConversionError(f"Failed to convert LAB to RGB: {str(e)}")
    
    def save_image(self, 
                   image: np.ndarray, 
                   output_path: Union[str, Path],
                   format_override: Optional[str] = None,
                   quality_override: Optional[int] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save image to file with specified format and quality.
        
        Args:
            image: RGB image array to save
            output_path: Path to save the image
            format_override: Override output format ('jpg', 'png', 'tiff')
            quality_override: Override quality setting (1-100 for JPEG)
            metadata: Optional metadata to embed in the image
            
        Raises:
            ImageSaveError: If saving fails
        """
        try:
            output_path = Path(output_path)
            
            # Ensure image is in correct format
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            
            # Validate image shape
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ImageSaveError(f"Invalid image shape for saving: {image.shape}")
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image, mode='RGB')
            
            # Determine format
            output_format = format_override or self.output_config['output_format']
            if not format_override:
                # Infer from file extension if not overridden
                ext = output_path.suffix.lower()
                if ext in ['.jpg', '.jpeg']:
                    output_format = 'jpg'
                elif ext == '.png':
                    output_format = 'png'
                elif ext in ['.tiff', '.tif']:
                    output_format = 'tiff'
            
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with appropriate format and settings
            if output_format in ['jpg', 'jpeg']:
                quality = quality_override or self.output_config['output_quality']
                pil_image.save(output_path, 'JPEG', quality=quality, optimize=True)
            elif output_format == 'png':
                pil_image.save(output_path, 'PNG', optimize=True)
            elif output_format in ['tiff', 'tif']:
                pil_image.save(output_path, 'TIFF')
            else:
                # Default to JPEG
                quality = quality_override or self.output_config['output_quality']
                pil_image.save(output_path, 'JPEG', quality=quality, optimize=True)
            
            logger.info(f"Successfully saved image to {output_path}")
            
        except Exception as e:
            raise ImageSaveError(f"Failed to save image to {output_path}: {str(e)}")
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported output formats.
        
        Returns:
            List of supported format strings
        """
        return ['jpg', 'jpeg', 'png', 'tiff', 'tif']
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update output configuration.
        
        Args:
            new_config: New configuration parameters
        """
        self.output_config.update(new_config)
        logger.info("Post-processor configuration updated")