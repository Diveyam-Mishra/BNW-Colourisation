"""
Color space conversion utilities for image colorization.

This module provides functionality to convert between RGB and CIELAB color spaces,
extract individual channels, and combine channels back into full color images.
"""

import numpy as np
from typing import Tuple
import cv2


class ColorSpaceConverter:
    """
    Handles conversions between RGB and CIELAB color spaces for image colorization.
    
    The CIELAB color space is used for training as it separates luminance (L) from
    chrominance (a, b) channels, making it ideal for colorization tasks where we
    predict color information from grayscale input.
    """
    
    def __init__(self):
        """Initialize the ColorSpaceConverter."""
        pass
    
    def rgb_to_lab(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Convert RGB image to CIELAB color space.
        
        Args:
            rgb_image: Input RGB image as numpy array with shape (H, W, 3)
                      and values in range [0, 255] or [0, 1]
        
        Returns:
            LAB image as numpy array with shape (H, W, 3)
            L channel: [0, 100], a and b channels: [-127, 127]
        
        Raises:
            ValueError: If input image has invalid shape or data type
        """
        if not isinstance(rgb_image, np.ndarray):
            raise ValueError("Input must be a numpy array")
        
        if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
            raise ValueError("Input image must have shape (H, W, 3)")
        
        # Ensure input is in [0, 255] range for OpenCV
        if rgb_image.dtype == np.float32 or rgb_image.dtype == np.float64:
            if rgb_image.max() <= 1.0:
                rgb_image = (rgb_image * 255).astype(np.uint8)
            else:
                rgb_image = rgb_image.astype(np.uint8)
        
        # Convert RGB to LAB using OpenCV
        lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
        
        # Convert to float and normalize to proper LAB ranges
        lab_image = lab_image.astype(np.float32)
        
        # OpenCV LAB conversion gives L in [0, 255], a and b in [0, 255]
        # We need to convert to standard LAB ranges: L [0, 100], a,b [-127, 127]
        lab_image[:, :, 0] = lab_image[:, :, 0] * (100.0 / 255.0)  # L channel: [0, 255] -> [0, 100]
        lab_image[:, :, 1] = lab_image[:, :, 1] - 128.0  # a channel: [0, 255] -> [-128, 127]
        lab_image[:, :, 2] = lab_image[:, :, 2] - 128.0  # b channel: [0, 255] -> [-128, 127]
        
        return lab_image
    
    def lab_to_rgb(self, lab_image: np.ndarray) -> np.ndarray:
        """
        Convert CIELAB image to RGB color space.
        
        Args:
            lab_image: Input LAB image as numpy array with shape (H, W, 3)
                      L channel: [0, 100], a and b channels: [-127, 127]
        
        Returns:
            RGB image as numpy array with shape (H, W, 3) and values in [0, 255]
        
        Raises:
            ValueError: If input image has invalid shape or data type
        """
        if not isinstance(lab_image, np.ndarray):
            raise ValueError("Input must be a numpy array")
        
        if len(lab_image.shape) != 3 or lab_image.shape[2] != 3:
            raise ValueError("Input image must have shape (H, W, 3)")
        
        # Convert from standard LAB ranges to OpenCV LAB ranges
        lab_opencv = lab_image.copy().astype(np.float32)
        lab_opencv[:, :, 0] = lab_opencv[:, :, 0] * (255.0 / 100.0)  # L: [0, 100] -> [0, 255]
        lab_opencv[:, :, 1] = lab_opencv[:, :, 1] + 128.0  # a: [-127, 127] -> [1, 255]
        lab_opencv[:, :, 2] = lab_opencv[:, :, 2] + 128.0  # b: [-127, 127] -> [1, 255]
        
        # Clip to valid range and convert to uint8
        lab_opencv = np.clip(lab_opencv, 0, 255).astype(np.uint8)
        
        # Convert LAB to RGB using OpenCV
        rgb_image = cv2.cvtColor(lab_opencv, cv2.COLOR_LAB2RGB)
        
        return rgb_image
    
    def extract_l_channel(self, lab_image: np.ndarray) -> np.ndarray:
        """
        Extract the L (lightness) channel from a LAB image.
        
        Args:
            lab_image: Input LAB image with shape (H, W, 3)
        
        Returns:
            L channel as numpy array with shape (H, W) and values in [0, 100]
        
        Raises:
            ValueError: If input image has invalid shape
        """
        if not isinstance(lab_image, np.ndarray):
            raise ValueError("Input must be a numpy array")
        
        if len(lab_image.shape) != 3 or lab_image.shape[2] != 3:
            raise ValueError("Input image must have shape (H, W, 3)")
        
        return lab_image[:, :, 0]
    
    def extract_ab_channels(self, lab_image: np.ndarray) -> np.ndarray:
        """
        Extract the a and b (chrominance) channels from a LAB image.
        
        Args:
            lab_image: Input LAB image with shape (H, W, 3)
        
        Returns:
            AB channels as numpy array with shape (H, W, 2) and values in [-127, 127]
        
        Raises:
            ValueError: If input image has invalid shape
        """
        if not isinstance(lab_image, np.ndarray):
            raise ValueError("Input must be a numpy array")
        
        if len(lab_image.shape) != 3 or lab_image.shape[2] != 3:
            raise ValueError("Input image must have shape (H, W, 3)")
        
        return lab_image[:, :, 1:3]
    
    def combine_lab_channels(self, l_channel: np.ndarray, ab_channels: np.ndarray) -> np.ndarray:
        """
        Combine L channel with predicted AB channels to create full LAB image.
        
        Args:
            l_channel: Lightness channel with shape (H, W) or (H, W, 1)
            ab_channels: Chrominance channels with shape (H, W, 2)
        
        Returns:
            Combined LAB image with shape (H, W, 3)
        
        Raises:
            ValueError: If input channels have incompatible shapes
        """
        if not isinstance(l_channel, np.ndarray) or not isinstance(ab_channels, np.ndarray):
            raise ValueError("Inputs must be numpy arrays")
        
        # Handle L channel shape
        if len(l_channel.shape) == 3 and l_channel.shape[2] == 1:
            l_channel = l_channel[:, :, 0]
        elif len(l_channel.shape) != 2:
            raise ValueError("L channel must have shape (H, W) or (H, W, 1)")
        
        # Validate AB channels shape
        if len(ab_channels.shape) != 3 or ab_channels.shape[2] != 2:
            raise ValueError("AB channels must have shape (H, W, 2)")
        
        # Check spatial dimensions match
        if l_channel.shape[:2] != ab_channels.shape[:2]:
            raise ValueError("L and AB channels must have matching spatial dimensions")
        
        # Combine channels
        lab_image = np.zeros((l_channel.shape[0], l_channel.shape[1], 3), dtype=l_channel.dtype)
        lab_image[:, :, 0] = l_channel
        lab_image[:, :, 1:3] = ab_channels
        
        return lab_image
    
    def normalize_lab_for_training(self, lab_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize LAB image for neural network training.
        
        Args:
            lab_image: Input LAB image with shape (H, W, 3)
        
        Returns:
            Tuple of (normalized_l_channel, normalized_ab_channels)
            L channel normalized to [-1, 1], AB channels normalized to [-1, 1]
        """
        if not isinstance(lab_image, np.ndarray):
            raise ValueError("Input must be a numpy array")
        
        if len(lab_image.shape) != 3 or lab_image.shape[2] != 3:
            raise ValueError("Input image must have shape (H, W, 3)")
        
        # Extract channels
        l_channel = lab_image[:, :, 0]
        ab_channels = lab_image[:, :, 1:3]
        
        # Normalize L channel from [0, 100] to [-1, 1]
        l_normalized = (l_channel / 50.0) - 1.0
        
        # Normalize AB channels from [-127, 127] to [-1, 1]
        ab_normalized = ab_channels / 127.0
        
        return l_normalized, ab_normalized
    
    def denormalize_lab_from_training(self, l_normalized: np.ndarray, ab_normalized: np.ndarray) -> np.ndarray:
        """
        Denormalize LAB channels from training format back to standard LAB ranges.
        
        Args:
            l_normalized: L channel normalized to [-1, 1]
            ab_normalized: AB channels normalized to [-1, 1]
        
        Returns:
            LAB image with standard ranges: L [0, 100], AB [-127, 127]
        """
        if not isinstance(l_normalized, np.ndarray) or not isinstance(ab_normalized, np.ndarray):
            raise ValueError("Inputs must be numpy arrays")
        
        # Denormalize L channel from [-1, 1] to [0, 100]
        l_channel = (l_normalized + 1.0) * 50.0
        
        # Denormalize AB channels from [-1, 1] to [-127, 127]
        ab_channels = ab_normalized * 127.0
        
        # Combine channels
        return self.combine_lab_channels(l_channel, ab_channels)
    
    def validate_lab_ranges(self, lab_image: np.ndarray) -> bool:
        """
        Validate that LAB image values are within expected ranges.
        
        Args:
            lab_image: LAB image to validate
        
        Returns:
            True if all values are within valid ranges, False otherwise
        """
        if not isinstance(lab_image, np.ndarray):
            return False
        
        if len(lab_image.shape) != 3 or lab_image.shape[2] != 3:
            return False
        
        l_channel = lab_image[:, :, 0]
        a_channel = lab_image[:, :, 1]
        b_channel = lab_image[:, :, 2]
        
        # Check L channel range [0, 100]
        if l_channel.min() < 0 or l_channel.max() > 100:
            return False
        
        # Check A and B channel ranges [-127, 127]
        if a_channel.min() < -127 or a_channel.max() > 127:
            return False
        
        if b_channel.min() < -127 or b_channel.max() > 127:
            return False
        
        return True
    
    def validate_rgb_ranges(self, rgb_image: np.ndarray) -> bool:
        """
        Validate that RGB image values are within expected ranges.
        
        Args:
            rgb_image: RGB image to validate
        
        Returns:
            True if all values are within valid ranges [0, 255], False otherwise
        """
        if not isinstance(rgb_image, np.ndarray):
            return False
        
        if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
            return False
        
        # Check RGB ranges [0, 255]
        if rgb_image.min() < 0 or rgb_image.max() > 255:
            return False
        
        return True
    
    def test_round_trip_accuracy(self, rgb_image: np.ndarray, tolerance: float = 15.0) -> Tuple[bool, float]:
        """
        Test the accuracy of RGB -> LAB -> RGB round-trip conversion.
        
        Args:
            rgb_image: Input RGB image to test
            tolerance: Maximum allowed difference in RGB values
        
        Returns:
            Tuple of (is_accurate, max_difference)
        """
        try:
            # Perform round-trip conversion
            lab_image = self.rgb_to_lab(rgb_image)
            rgb_reconstructed = self.lab_to_rgb(lab_image)
            
            # Calculate maximum absolute difference
            diff = np.abs(rgb_reconstructed.astype(np.float32) - rgb_image.astype(np.float32))
            max_diff = np.max(diff)
            
            # Check if within tolerance
            is_accurate = bool(max_diff <= tolerance)
            
            return is_accurate, float(max_diff)
            
        except Exception:
            return False, float('inf')
    
    def validate_color_space_boundaries(self, test_extreme_values: bool = True) -> dict:
        """
        Validate color space conversion behavior at boundary conditions.
        
        Args:
            test_extreme_values: Whether to test extreme color values
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'boundary_tests_passed': 0,
            'boundary_tests_total': 0,
            'extreme_tests_passed': 0,
            'extreme_tests_total': 0,
            'errors': []
        }
        
        # Test boundary conditions
        boundary_colors = [
            [0, 0, 0],        # Black
            [255, 255, 255],  # White
            [128, 128, 128],  # Gray
        ]
        
        for color in boundary_colors:
            results['boundary_tests_total'] += 1
            try:
                rgb_test = np.array([[color]], dtype=np.uint8)
                lab_result = self.rgb_to_lab(rgb_test)
                rgb_result = self.lab_to_rgb(lab_result)
                
                # Validate ranges
                if self.validate_lab_ranges(lab_result) and self.validate_rgb_ranges(rgb_result):
                    results['boundary_tests_passed'] += 1
                else:
                    results['errors'].append(f"Range validation failed for color {color}")
                    
            except Exception as e:
                results['errors'].append(f"Exception for color {color}: {str(e)}")
        
        # Test extreme values if requested
        if test_extreme_values:
            extreme_colors = [
                [255, 0, 0],    # Pure red
                [0, 255, 0],    # Pure green
                [0, 0, 255],    # Pure blue
                [255, 255, 0],  # Yellow
                [255, 0, 255],  # Magenta
                [0, 255, 255],  # Cyan
            ]
            
            for color in extreme_colors:
                results['extreme_tests_total'] += 1
                try:
                    rgb_test = np.array([[color]], dtype=np.uint8)
                    is_accurate, max_diff = self.test_round_trip_accuracy(rgb_test, tolerance=20.0)
                    
                    if is_accurate:
                        results['extreme_tests_passed'] += 1
                    else:
                        results['errors'].append(f"Round-trip accuracy failed for color {color}, max_diff: {max_diff}")
                        
                except Exception as e:
                    results['errors'].append(f"Exception for extreme color {color}: {str(e)}")
        
        return results