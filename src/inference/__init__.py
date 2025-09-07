"""
Inference module for image colorization.

This module provides inference capabilities for the colorization model,
including single image processing, batch processing, and memory-efficient
handling of large images.
"""

from .inference_wrapper import ColorizationInference

__all__ = ['ColorizationInference']