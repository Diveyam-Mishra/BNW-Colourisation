"""
Dataset handling for image colorization training.

This module provides dataset classes for loading and preprocessing training images,
implementing data augmentation pipelines, and handling train/validation splits.
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional, Union, Callable, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import logging

from src.data.models import ImageData, TrainingConfig
from src.data.preprocessor import ImagePreprocessor, ImageProcessingError
from src.utils.color_converter import ColorSpaceConverter

# Configure logging
logger = logging.getLogger(__name__)


class ColorizationDataset(Dataset):
    """
    Dataset class for loading and preprocessing images for colorization training.
    
    This dataset loads color images, converts them to CIELAB space, and provides
    the L channel as input and ab channels as target for training.
    """
    
    def __init__(self,
                 dataset_path: Union[str, Path],
                 target_size: Tuple[int, int] = (256, 256),
                 augmentation_enabled: bool = True,
                 augmentation_transforms: Optional[Callable] = None):
        """
        Initialize the colorization dataset.
        
        Args:
            dataset_path: Path to directory containing training images
            target_size: Target size for resizing images (height, width)
            augmentation_enabled: Whether to apply data augmentation
            augmentation_transforms: Custom augmentation transforms (if None, uses default)
        """
        self.dataset_path = Path(dataset_path)
        self.target_size = target_size
        self.augmentation_enabled = augmentation_enabled
        
        # Initialize preprocessor and color converter
        self.preprocessor = ImagePreprocessor(target_size=target_size)
        self.color_converter = ColorSpaceConverter()
        
        # Load image paths
        self.image_paths = self._load_image_paths()
        
        # Set up augmentation transforms
        if augmentation_transforms is None:
            self.augmentation_transforms = self._create_default_augmentations()
        else:
            self.augmentation_transforms = augmentation_transforms
            
        logger.info(f"Initialized dataset with {len(self.image_paths)} images from {dataset_path}")
        
    def _load_image_paths(self) -> List[Path]:
        """
        Load all valid image paths from the dataset directory.
        
        Returns:
            List of valid image file paths
            
        Raises:
            ValueError: If dataset path doesn't exist or contains no valid images
        """
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {self.dataset_path}")
            
        if not self.dataset_path.is_dir():
            raise ValueError(f"Dataset path is not a directory: {self.dataset_path}")
        
        # Supported image extensions
        supported_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        
        image_paths = []
        
        # Recursively find all image files
        for ext in supported_extensions:
            # Case insensitive search
            image_paths.extend(self.dataset_path.rglob(f"*{ext}"))
            image_paths.extend(self.dataset_path.rglob(f"*{ext.upper()}"))
        
        # Remove duplicates and sort
        image_paths = sorted(list(set(image_paths)))
        
        # Validate images
        valid_paths = []
        for path in image_paths:
            try:
                self.preprocessor.validate_image(path)
                valid_paths.append(path)
            except ImageProcessingError as e:
                logger.warning(f"Skipping invalid image {path}: {e}")
                
        if not valid_paths:
            raise ValueError(f"No valid images found in {self.dataset_path}")
            
        return valid_paths
        
    def _create_default_augmentations(self) -> transforms.Compose:
        """
        Create default data augmentation transforms.
        
        Returns:
            Composed transforms for data augmentation
        """
        augmentation_list = [
            # Random horizontal flip
            transforms.RandomHorizontalFlip(p=0.5),
            
            # Random rotation (small angles)
            transforms.RandomRotation(degrees=(-10, 10), fill=0),
            
            # Random brightness and contrast
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05
            ),
            
            # Random crop and resize (slight zoom)
            transforms.RandomResizedCrop(
                size=self.target_size,
                scale=(0.9, 1.0),
                ratio=(0.9, 1.1),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
        ]
        
        return transforms.Compose(augmentation_list)
        
    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_paths)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Dictionary containing:
                - 'l_channel': L channel tensor, shape (1, H, W)
                - 'ab_channels': ab channels tensor, shape (2, H, W)
                - 'lab_image': Full LAB image tensor, shape (3, H, W)
                - 'metadata': Dictionary with image metadata
                
        Raises:
            IndexError: If index is out of range
            ImageProcessingError: If image cannot be processed
        """
        if idx >= len(self.image_paths):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.image_paths)}")
            
        image_path = self.image_paths[idx]
        
        try:
            # Load and preprocess the image
            rgb_image = self.preprocessor.load_image(image_path)
            
            # Apply augmentation if enabled
            if self.augmentation_enabled and self.augmentation_transforms is not None:
                # Convert to PIL Image for transforms
                pil_image = Image.fromarray((rgb_image * 255).astype(np.uint8))
                pil_image = self.augmentation_transforms(pil_image)
                rgb_image = np.array(pil_image).astype(np.float32) / 255.0
            
            # Resize if needed
            rgb_image = self.preprocessor.resize_image(rgb_image, self.target_size)
            
            # Normalize
            rgb_image = self.preprocessor.normalize_image(rgb_image)
            
            # Convert to LAB color space
            lab_image = self.color_converter.rgb_to_lab(rgb_image)
            
            # Extract L and ab channels
            l_channel = self.color_converter.extract_l_channel(lab_image)
            ab_channels = lab_image[:, :, 1:3]  # a and b channels
            
            # Convert to tensors and rearrange dimensions to (C, H, W)
            l_tensor = torch.from_numpy(l_channel).unsqueeze(0).float()  # (1, H, W)
            ab_tensor = torch.from_numpy(ab_channels).permute(2, 0, 1).float()  # (2, H, W)
            lab_tensor = torch.from_numpy(lab_image).permute(2, 0, 1).float()  # (3, H, W)
            
            # Create metadata
            metadata = {
                'image_path': str(image_path),
                'original_size': rgb_image.shape[:2],
                'target_size': self.target_size,
                'augmented': self.augmentation_enabled
            }
            
            return {
                'l_channel': l_tensor,
                'ab_channels': ab_tensor,
                'lab_image': lab_tensor,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise ImageProcessingError(f"Failed to process image {image_path}: {e}")
            
    def get_image_paths(self) -> List[Path]:
        """
        Get list of all image paths in the dataset.
        
        Returns:
            List of image file paths
        """
        return self.image_paths.copy()
        
    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        return {
            'num_images': len(self.image_paths),
            'dataset_path': str(self.dataset_path),
            'target_size': self.target_size,
            'augmentation_enabled': self.augmentation_enabled
        }


class DatasetSplitter:
    """
    Utility class for splitting datasets into training and validation sets.
    """
    
    @staticmethod
    def split_dataset(dataset: ColorizationDataset, 
                     validation_split: float = 0.2,
                     random_seed: Optional[int] = None) -> Tuple[Dataset, Dataset]:
        """
        Split a dataset into training and validation sets.
        
        Args:
            dataset: Dataset to split
            validation_split: Fraction of data to use for validation (0.0 to 1.0)
            random_seed: Random seed for reproducible splits
            
        Returns:
            Tuple of (train_dataset, validation_dataset)
            
        Raises:
            ValueError: If validation_split is not in valid range
        """
        if not (0.0 <= validation_split <= 1.0):
            raise ValueError(f"validation_split must be between 0.0 and 1.0, got {validation_split}")
            
        if random_seed is not None:
            torch.manual_seed(random_seed)
            
        total_size = len(dataset)
        val_size = int(total_size * validation_split)
        train_size = total_size - val_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        logger.info(f"Split dataset: {train_size} training, {val_size} validation samples")
        
        return train_dataset, val_dataset
        
    @staticmethod
    def create_dataloaders(train_dataset: Dataset,
                          val_dataset: Dataset,
                          batch_size: int = 16,
                          num_workers: int = 4,
                          pin_memory: bool = True) -> Tuple[DataLoader, DataLoader]:
        """
        Create DataLoader instances for training and validation datasets.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            batch_size: Batch size for data loading
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            
        Returns:
            Tuple of (train_dataloader, val_dataloader)
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True  # Drop last incomplete batch for consistent training
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        
        logger.info(f"Created dataloaders: batch_size={batch_size}, num_workers={num_workers}")
        
        return train_loader, val_loader


def create_training_dataloaders(config: TrainingConfig,
                               target_size: Tuple[int, int] = (256, 256),
                               batch_size: int = 16,
                               num_workers: int = 4) -> Tuple[DataLoader, DataLoader, ColorizationDataset]:
    """
    Convenience function to create training and validation dataloaders from config.
    
    Args:
        config: Training configuration
        target_size: Target image size for resizing
        batch_size: Batch size for data loading
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_dataloader, val_dataloader, full_dataset)
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Create dataset
    dataset = ColorizationDataset(
        dataset_path=config.dataset_path,
        target_size=target_size,
        augmentation_enabled=config.augmentation_enabled
    )
    
    # Split dataset
    train_dataset, val_dataset = DatasetSplitter.split_dataset(
        dataset=dataset,
        validation_split=config.validation_split,
        random_seed=42  # Fixed seed for reproducibility
    )
    
    # Create dataloaders
    train_loader, val_loader = DatasetSplitter.create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, dataset


class AugmentationFactory:
    """
    Factory class for creating different types of augmentation transforms.
    """
    
    @staticmethod
    def create_light_augmentation(target_size: Tuple[int, int]) -> transforms.Compose:
        """
        Create light augmentation transforms for training.
        
        Args:
            target_size: Target image size
            
        Returns:
            Composed transforms with light augmentation
        """
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02),
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR)
        ])
        
    @staticmethod
    def create_medium_augmentation(target_size: Tuple[int, int]) -> transforms.Compose:
        """
        Create medium augmentation transforms for training.
        
        Args:
            target_size: Target image size
            
        Returns:
            Composed transforms with medium augmentation
        """
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-5, 5), fill=0),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.RandomResizedCrop(
                size=target_size,
                scale=(0.95, 1.0),
                ratio=(0.95, 1.05),
                interpolation=transforms.InterpolationMode.BILINEAR
            )
        ])
        
    @staticmethod
    def create_heavy_augmentation(target_size: Tuple[int, int]) -> transforms.Compose:
        """
        Create heavy augmentation transforms for training.
        
        Args:
            target_size: Target image size
            
        Returns:
            Composed transforms with heavy augmentation
        """
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-15, 15), fill=0),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(
                size=target_size,
                scale=(0.8, 1.0),
                ratio=(0.8, 1.2),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.2)
        ])