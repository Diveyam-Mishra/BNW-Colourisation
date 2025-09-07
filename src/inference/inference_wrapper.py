"""
Inference wrapper for image colorization model.

This module provides a high-level interface for performing inference with
the trained colorization model, supporting single images, batch processing,
and memory-efficient processing of large images.
"""

import os
import logging
from pathlib import Path
from typing import Union, List, Optional, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..models.unet import UNetColorizer
from ..data.preprocessor import ImagePreprocessor
from ..utils.color_converter import ColorSpaceConverter
from ..utils.config import load_config_file

logger = logging.getLogger(__name__)


class InferenceError(Exception):
    """Base exception for inference errors."""
    pass


class ModelLoadError(InferenceError):
    """Raised when model loading fails."""
    pass


class InsufficientMemory(InferenceError):
    """Raised when insufficient memory for processing."""
    pass


class ColorizationInference:
    """
    High-level inference wrapper for image colorization.
    
    This class provides methods for single image colorization, batch processing,
    and memory-efficient processing of large images using the trained U-Net model.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 config_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize the inference wrapper.
        
        Args:
            model_path: Path to the trained model checkpoint
            config_path: Path to inference configuration file
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        self.model = None
        self.device = self._setup_device(device)
        self.preprocessor = None
        self.color_converter = ColorSpaceConverter()
        
        # Load configuration
        self.config = self._load_inference_config(config_path)
        
        # Initialize preprocessor with config
        target_size = tuple(self.config.get('input_size', [256, 256]))
        self.preprocessor = ImagePreprocessor(target_size=target_size)
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
    
    def _setup_device(self, device: Optional[str]) -> torch.device:
        """
        Setup the computation device.
        
        Args:
            device: Device specification ('cpu', 'cuda', or 'auto')
            
        Returns:
            torch.device: Configured device
        """
        if device == 'auto' or device is None:
            if torch.cuda.is_available():
                device = 'cuda'
                logger.info("Using CUDA device for inference")
            else:
                device = 'cpu'
                logger.info("Using CPU device for inference")
        
        return torch.device(device)
    
    def _load_inference_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load inference configuration.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        if config_path and os.path.exists(config_path):
            try:
                config = load_config_file(config_path)
                return config.get('inference', {})
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        # Default configuration
        return {
            'input_size': [256, 256],
            'batch_size': 8,
            'max_image_size': [2048, 2048],
            'preserve_aspect_ratio': True,
            'enable_memory_efficient_mode': True,
            'patch_size': [512, 512],
            'output_format': 'jpg',
            'output_quality': 95
        }
    
    def load_model(self, model_path: str) -> None:
        """
        Load the trained colorization model.
        
        Args:
            model_path: Path to the model checkpoint
            
        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            if not os.path.exists(model_path):
                raise ModelLoadError(f"Model file not found: {model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract model configuration if available
            model_config = checkpoint.get('model_config', {})
            input_channels = model_config.get('input_channels', 1)
            output_channels = model_config.get('output_channels', 2)
            
            # Initialize model
            self.model = UNetColorizer(
                input_channels=input_channels,
                output_channels=output_channels
            )
            
            # Load state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Successfully loaded model from {model_path}")
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {str(e)}")
    
    def colorize_image(self, 
                      image_path: Union[str, Path],
                      output_path: Optional[Union[str, Path]] = None) -> np.ndarray:
        """
        Colorize a single grayscale image.
        
        Args:
            image_path: Path to the input grayscale image
            output_path: Optional path to save the colorized image
            
        Returns:
            Colorized image as RGB numpy array
            
        Raises:
            InferenceError: If inference fails
        """
        if self.model is None:
            raise InferenceError("Model not loaded. Call load_model() first.")
        
        try:
            # Load and preprocess image
            rgb_image = self.preprocessor.load_image(image_path)
            
            # Check if image needs memory-efficient processing
            height, width = rgb_image.shape[:2]
            max_h, max_w = self.config['max_image_size']
            
            if (height > max_h or width > max_w) and self.config['enable_memory_efficient_mode']:
                colorized = self._colorize_large_image(rgb_image)
            else:
                colorized = self._colorize_single_image(rgb_image)
            
            # Save if output path provided
            if output_path:
                self._save_image(colorized, output_path)
            
            return colorized
            
        except Exception as e:
            raise InferenceError(f"Failed to colorize image {image_path}: {str(e)}")
    
    def colorize_batch(self, 
                      image_paths: List[Union[str, Path]],
                      output_dir: Optional[Union[str, Path]] = None) -> List[np.ndarray]:
        """
        Colorize multiple images in batch.
        
        Args:
            image_paths: List of paths to input images
            output_dir: Optional directory to save colorized images
            
        Returns:
            List of colorized images as RGB numpy arrays
            
        Raises:
            InferenceError: If batch processing fails
        """
        if not image_paths:
            return []
        
        if self.model is None:
            raise InferenceError("Model not loaded. Call load_model() first.")
        
        colorized_images = []
        batch_size = self.config['batch_size']
        
        # Create output directory if needed
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process images in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            try:
                batch_results = self._process_batch(batch_paths)
                colorized_images.extend(batch_results)
                
                # Save batch results if output directory provided
                if output_dir:
                    for j, colorized in enumerate(batch_results):
                        input_path = Path(batch_paths[j])
                        output_path = output_dir / f"colorized_{input_path.stem}.{self.config['output_format']}"
                        self._save_image(colorized, output_path)
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(image_paths) + batch_size - 1)//batch_size}")
                
            except Exception as e:
                logger.error(f"Failed to process batch starting at index {i}: {e}")
                # Add None placeholders for failed batch
                colorized_images.extend([None] * len(batch_paths))
        
        return colorized_images
    
    def _colorize_single_image(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Colorize a single image using the model.
        
        Args:
            rgb_image: Input RGB image
            
        Returns:
            Colorized RGB image
        """
        # Convert to LAB and extract L channel
        lab_image = self.color_converter.rgb_to_lab(rgb_image)
        l_channel = self.color_converter.extract_l_channel(lab_image)
        
        # Normalize L channel for model input
        l_normalized = (l_channel / 50.0) - 1.0  # [0, 100] -> [-1, 1]
        
        # Prepare tensor for model
        l_tensor = torch.from_numpy(l_normalized).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        l_tensor = l_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            ab_pred = self.model(l_tensor)  # (1, 2, H, W)
        
        # Convert back to numpy
        ab_pred = ab_pred.cpu().numpy()[0]  # (2, H, W)
        ab_pred = np.transpose(ab_pred, (1, 2, 0))  # (H, W, 2)
        
        # Denormalize AB channels
        ab_pred = ab_pred * 127.0  # [-1, 1] -> [-127, 127]
        
        # Combine with original L channel
        lab_colorized = self.color_converter.combine_lab_channels(l_channel, ab_pred)
        
        # Convert back to RGB
        rgb_colorized = self.color_converter.lab_to_rgb(lab_colorized)
        
        return rgb_colorized
    
    def _colorize_large_image(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Colorize a large image using patch-based processing.
        
        Args:
            rgb_image: Input RGB image
            
        Returns:
            Colorized RGB image
        """
        height, width = rgb_image.shape[:2]
        patch_h, patch_w = self.config['patch_size']
        
        # Calculate overlap for smooth blending
        overlap = 32
        stride_h = patch_h - overlap
        stride_w = patch_w - overlap
        
        # Convert to LAB once
        lab_image = self.color_converter.rgb_to_lab(rgb_image)
        l_channel = self.color_converter.extract_l_channel(lab_image)
        
        # Initialize output AB channels
        ab_output = np.zeros((height, width, 2), dtype=np.float32)
        weight_map = np.zeros((height, width), dtype=np.float32)
        
        # Process patches
        for y in range(0, height, stride_h):
            for x in range(0, width, stride_w):
                # Calculate patch boundaries
                y_end = min(y + patch_h, height)
                x_end = min(x + patch_w, width)
                
                # Extract patch
                l_patch = l_channel[y:y_end, x:x_end]
                
                # Pad patch if necessary
                pad_h = patch_h - l_patch.shape[0]
                pad_w = patch_w - l_patch.shape[1]
                
                if pad_h > 0 or pad_w > 0:
                    l_patch = np.pad(l_patch, ((0, pad_h), (0, pad_w)), mode='reflect')
                
                # Process patch
                ab_patch = self._process_patch(l_patch)
                
                # Remove padding
                if pad_h > 0 or pad_w > 0:
                    ab_patch = ab_patch[:l_patch.shape[0]-pad_h, :l_patch.shape[1]-pad_w]
                
                # Create weight mask for blending
                patch_weight = self._create_patch_weight(ab_patch.shape[:2], overlap)
                
                # Accumulate results
                ab_output[y:y_end, x:x_end] += ab_patch * patch_weight[:, :, np.newaxis]
                weight_map[y:y_end, x:x_end] += patch_weight
        
        # Normalize by weights
        weight_map = np.maximum(weight_map, 1e-8)  # Avoid division by zero
        ab_output = ab_output / weight_map[:, :, np.newaxis]
        
        # Combine with original L channel
        lab_colorized = self.color_converter.combine_lab_channels(l_channel, ab_output)
        
        # Convert back to RGB
        rgb_colorized = self.color_converter.lab_to_rgb(lab_colorized)
        
        return rgb_colorized
    
    def _process_patch(self, l_patch: np.ndarray) -> np.ndarray:
        """
        Process a single patch through the model.
        
        Args:
            l_patch: L channel patch
            
        Returns:
            Predicted AB channels for the patch
        """
        # Normalize L channel
        l_normalized = (l_patch / 50.0) - 1.0
        
        # Prepare tensor
        l_tensor = torch.from_numpy(l_normalized).float().unsqueeze(0).unsqueeze(0)
        l_tensor = l_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            ab_pred = self.model(l_tensor)
        
        # Convert back to numpy
        ab_pred = ab_pred.cpu().numpy()[0]
        ab_pred = np.transpose(ab_pred, (1, 2, 0))
        
        # Denormalize
        ab_pred = ab_pred * 127.0
        
        return ab_pred
    
    def _create_patch_weight(self, patch_shape: Tuple[int, int], overlap: int) -> np.ndarray:
        """
        Create weight mask for patch blending.
        
        Args:
            patch_shape: Shape of the patch (height, width)
            overlap: Overlap size for blending
            
        Returns:
            Weight mask for the patch
        """
        height, width = patch_shape
        weight = np.ones((height, width), dtype=np.float32)
        
        # Create fade-in/fade-out at edges
        if overlap > 0:
            # Top edge
            for i in range(min(overlap, height)):
                weight[i, :] *= (i + 1) / overlap
            
            # Bottom edge
            for i in range(min(overlap, height)):
                weight[height - 1 - i, :] *= (i + 1) / overlap
            
            # Left edge
            for j in range(min(overlap, width)):
                weight[:, j] *= (j + 1) / overlap
            
            # Right edge
            for j in range(min(overlap, width)):
                weight[:, width - 1 - j] *= (j + 1) / overlap
        
        return weight
    
    def _process_batch(self, image_paths: List[Union[str, Path]]) -> List[np.ndarray]:
        """
        Process a batch of images.
        
        Args:
            image_paths: List of image paths to process
            
        Returns:
            List of colorized images
        """
        batch_results = []
        
        for image_path in image_paths:
            try:
                rgb_image = self.preprocessor.load_image(image_path)
                colorized = self._colorize_single_image(rgb_image)
                batch_results.append(colorized)
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                batch_results.append(None)
        
        return batch_results
    
    def _save_image(self, image: np.ndarray, output_path: Union[str, Path]) -> None:
        """
        Save colorized image to file.
        
        Args:
            image: RGB image array to save
            output_path: Path to save the image
        """
        try:
            # Ensure image is in correct format
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image, mode='RGB')
            
            # Save with appropriate format and quality
            output_path = Path(output_path)
            
            if output_path.suffix.lower() in ['.jpg', '.jpeg']:
                pil_image.save(output_path, 'JPEG', quality=self.config['output_quality'])
            elif output_path.suffix.lower() == '.png':
                pil_image.save(output_path, 'PNG')
            elif output_path.suffix.lower() in ['.tiff', '.tif']:
                pil_image.save(output_path, 'TIFF')
            else:
                # Default to JPEG
                pil_image.save(output_path, 'JPEG', quality=self.config['output_quality'])
            
            logger.debug(f"Saved colorized image to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save image to {output_path}: {e}")
            raise InferenceError(f"Failed to save image: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"status": "No model loaded"}
        
        info = self.model.get_model_info()
        info.update({
            "device": str(self.device),
            "config": self.config
        })
        
        return info
    
    def estimate_memory_usage(self, image_shape: Tuple[int, int]) -> Dict[str, float]:
        """
        Estimate memory usage for processing an image of given shape.
        
        Args:
            image_shape: Shape of the image (height, width)
            
        Returns:
            Dictionary with memory estimates in MB
        """
        height, width = image_shape
        
        # Estimate tensor memory usage
        input_tensor_mb = (height * width * 4) / (1024 * 1024)  # float32
        output_tensor_mb = (height * width * 2 * 4) / (1024 * 1024)  # 2 channels, float32
        
        # Model parameters (approximate)
        if self.model:
            model_params = sum(p.numel() for p in self.model.parameters())
            model_mb = (model_params * 4) / (1024 * 1024)  # float32
        else:
            model_mb = 0
        
        # Intermediate activations (rough estimate)
        activation_mb = input_tensor_mb * 10  # Rough multiplier for U-Net activations
        
        total_mb = input_tensor_mb + output_tensor_mb + model_mb + activation_mb
        
        return {
            "input_tensor_mb": input_tensor_mb,
            "output_tensor_mb": output_tensor_mb,
            "model_parameters_mb": model_mb,
            "estimated_activations_mb": activation_mb,
            "total_estimated_mb": total_mb
        }