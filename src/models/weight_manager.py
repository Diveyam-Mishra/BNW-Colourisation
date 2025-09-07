"""
Model weight management utilities for U-Net colorization model.

This module provides comprehensive functionality for saving, loading, and managing
model weights and checkpoints, including validation and fallback mechanisms.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
import json
from pathlib import Path
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelWeightManager:
    """
    Comprehensive model weight management for U-Net colorization model.
    
    Handles model serialization, checkpoint management, pre-trained weight loading,
    and weight compatibility validation.
    """
    
    def __init__(self, model: nn.Module, checkpoint_dir: str = "checkpoints"):
        """
        Initialize weight manager.
        
        Args:
            model: PyTorch model to manage
            checkpoint_dir: Directory to store checkpoints
        """
        self.model = model
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(self, 
                       filepath: str,
                       epoch: Optional[int] = None,
                       loss: Optional[float] = None,
                       optimizer_state: Optional[Dict] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save model checkpoint with comprehensive information.
        
        Args:
            filepath: Path to save checkpoint
            epoch: Current training epoch
            loss: Current loss value
            optimizer_state: Optimizer state dictionary
            metadata: Additional metadata to save
        """
        try:
            # Prepare checkpoint data
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'model_config': self._get_model_config(),
                'timestamp': datetime.now().isoformat(),
                'pytorch_version': torch.__version__,
            }
            
            # Add optional training information
            if epoch is not None:
                checkpoint['epoch'] = epoch
            if loss is not None:
                checkpoint['loss'] = loss
            if optimizer_state is not None:
                checkpoint['optimizer_state_dict'] = optimizer_state
            if metadata is not None:
                checkpoint['metadata'] = metadata
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save checkpoint
            torch.save(checkpoint, filepath)
            logger.info(f"Checkpoint saved to {filepath}")
            
            # Save human-readable metadata
            self._save_checkpoint_metadata(filepath, checkpoint)
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            raise RuntimeError(f"Failed to save checkpoint: {str(e)}")
    
    def load_checkpoint(self, 
                       filepath: str,
                       load_optimizer: bool = False,
                       strict: bool = True) -> Dict[str, Any]:
        """
        Load model checkpoint with validation.
        
        Args:
            filepath: Path to checkpoint file
            load_optimizer: Whether to return optimizer state
            strict: Whether to strictly enforce key matching
            
        Returns:
            Dictionary containing loaded checkpoint information
        """
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
            
            # Load checkpoint
            checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
            
            # Validate checkpoint format
            self._validate_checkpoint_format(checkpoint)
            
            # Validate model compatibility
            self._validate_model_compatibility(checkpoint)
            
            # Load model state
            state_dict = checkpoint['model_state_dict']
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=strict)
            
            if missing_keys:
                logger.warning(f"Missing keys in checkpoint: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
            
            logger.info(f"Model weights loaded from {filepath}")
            
            # Prepare return information
            result = {
                'epoch': checkpoint.get('epoch'),
                'loss': checkpoint.get('loss'),
                'timestamp': checkpoint.get('timestamp'),
                'metadata': checkpoint.get('metadata', {}),
                'missing_keys': missing_keys,
                'unexpected_keys': unexpected_keys
            }
            
            if load_optimizer and 'optimizer_state_dict' in checkpoint:
                result['optimizer_state_dict'] = checkpoint['optimizer_state_dict']
            
            return result
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            raise RuntimeError(f"Failed to load checkpoint: {str(e)}")
    
    def save_model_weights_only(self, filepath: str) -> None:
        """
        Save only model weights without training information.
        
        Args:
            filepath: Path to save weights
        """
        try:
            weights = {
                'model_state_dict': self.model.state_dict(),
                'model_config': self._get_model_config(),
                'timestamp': datetime.now().isoformat(),
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            torch.save(weights, filepath)
            logger.info(f"Model weights saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving weights: {str(e)}")
            raise RuntimeError(f"Failed to save weights: {str(e)}")
    
    def load_pretrained_weights(self, 
                               weights_path: str,
                               strict: bool = False,
                               fallback_to_random: bool = True) -> bool:
        """
        Load pre-trained weights with validation and fallback.
        
        Args:
            weights_path: Path to pre-trained weights
            strict: Whether to strictly enforce key matching
            fallback_to_random: Whether to fallback to random initialization on failure
            
        Returns:
            True if weights loaded successfully, False if fallback used
        """
        try:
            if not os.path.exists(weights_path):
                logger.warning(f"Pre-trained weights not found: {weights_path}")
                if fallback_to_random:
                    logger.info("Falling back to random initialization")
                    self._initialize_random_weights()
                    return False
                else:
                    raise FileNotFoundError(f"Pre-trained weights not found: {weights_path}")
            
            # Load and validate weights
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Validate compatibility
            compatibility_issues = self._check_weight_compatibility(state_dict)
            if compatibility_issues and strict:
                raise ValueError(f"Weight compatibility issues: {compatibility_issues}")
            
            # Load weights
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=strict)
            
            if missing_keys:
                logger.warning(f"Missing keys when loading pre-trained weights: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in pre-trained weights: {unexpected_keys}")
            
            logger.info(f"Pre-trained weights loaded successfully from {weights_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading pre-trained weights: {str(e)}")
            if fallback_to_random:
                logger.info("Falling back to random initialization")
                self._initialize_random_weights()
                return False
            else:
                raise RuntimeError(f"Failed to load pre-trained weights: {str(e)}")
    
    def create_checkpoint_name(self, 
                              epoch: int,
                              loss: float,
                              prefix: str = "checkpoint") -> str:
        """
        Create standardized checkpoint filename.
        
        Args:
            epoch: Training epoch
            loss: Current loss value
            prefix: Filename prefix
            
        Returns:
            Standardized checkpoint filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_epoch_{epoch:04d}_loss_{loss:.6f}_{timestamp}.pth"
        return str(self.checkpoint_dir / filename)
    
    def list_checkpoints(self) -> list:
        """
        List all available checkpoints in checkpoint directory.
        
        Returns:
            List of checkpoint file paths
        """
        checkpoint_files = []
        if self.checkpoint_dir.exists():
            for file_path in self.checkpoint_dir.glob("*.pth"):
                checkpoint_files.append(str(file_path))
        
        return sorted(checkpoint_files)
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get the most recent checkpoint file.
        
        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        checkpoints = self.list_checkpoints()
        if checkpoints:
            # Sort by modification time and return the latest
            latest = max(checkpoints, key=lambda x: os.path.getmtime(x))
            return latest
        return None
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5) -> None:
        """
        Remove old checkpoints, keeping only the most recent ones.
        
        Args:
            keep_last_n: Number of recent checkpoints to keep
        """
        checkpoints = self.list_checkpoints()
        if len(checkpoints) > keep_last_n:
            # Sort by modification time
            checkpoints.sort(key=lambda x: os.path.getmtime(x))
            
            # Remove oldest checkpoints
            for checkpoint in checkpoints[:-keep_last_n]:
                try:
                    os.remove(checkpoint)
                    # Also remove metadata file if it exists
                    metadata_file = checkpoint.replace('.pth', '_metadata.json')
                    if os.path.exists(metadata_file):
                        os.remove(metadata_file)
                    logger.info(f"Removed old checkpoint: {checkpoint}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {checkpoint}: {str(e)}")
    
    def _get_model_config(self) -> Dict[str, Any]:
        """Get model configuration information."""
        config = {}
        
        # Try to get model-specific configuration
        if hasattr(self.model, 'input_channels'):
            config['input_channels'] = self.model.input_channels
        if hasattr(self.model, 'output_channels'):
            config['output_channels'] = self.model.output_channels
        if hasattr(self.model, 'get_model_info'):
            config.update(self.model.get_model_info())
        
        return config
    
    def _validate_checkpoint_format(self, checkpoint: Dict) -> None:
        """Validate checkpoint format."""
        required_keys = ['model_state_dict']
        for key in required_keys:
            if key not in checkpoint:
                raise ValueError(f"Invalid checkpoint format: missing '{key}'")
    
    def _validate_model_compatibility(self, checkpoint: Dict) -> None:
        """Validate model compatibility with checkpoint."""
        if 'model_config' in checkpoint:
            saved_config = checkpoint['model_config']
            current_config = self._get_model_config()
            
            # Check critical configuration parameters
            for key in ['input_channels', 'output_channels']:
                if key in saved_config and key in current_config:
                    if saved_config[key] != current_config[key]:
                        logger.warning(f"Model config mismatch for {key}: "
                                     f"saved={saved_config[key]}, current={current_config[key]}")
    
    def _check_weight_compatibility(self, state_dict: Dict) -> list:
        """Check compatibility between saved weights and current model."""
        issues = []
        current_state = self.model.state_dict()
        
        # Check for shape mismatches
        for key in state_dict:
            if key in current_state:
                saved_shape = state_dict[key].shape
                current_shape = current_state[key].shape
                if saved_shape != current_shape:
                    issues.append(f"Shape mismatch for {key}: saved={saved_shape}, current={current_shape}")
        
        return issues
    
    def _initialize_random_weights(self) -> None:
        """Initialize model with random weights."""
        def init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.model.apply(init_weights)
        logger.info("Model initialized with random weights")
    
    def validate_pretrained_weights(self, weights_path: str) -> Dict[str, Any]:
        """
        Validate pre-trained weights without loading them.
        
        Args:
            weights_path: Path to pre-trained weights
            
        Returns:
            Dictionary containing validation results
        """
        validation_result = {
            'valid': False,
            'file_exists': False,
            'format_valid': False,
            'compatibility_issues': [],
            'model_info': {},
            'error': None
        }
        
        try:
            # Check if file exists
            if not os.path.exists(weights_path):
                validation_result['error'] = f"File not found: {weights_path}"
                return validation_result
            
            validation_result['file_exists'] = True
            
            # Try to load and validate format
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                validation_result['model_info'] = checkpoint.get('model_config', {})
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            validation_result['format_valid'] = True
            
            # Check compatibility
            compatibility_issues = self._check_weight_compatibility(state_dict)
            validation_result['compatibility_issues'] = compatibility_issues
            
            # Overall validation
            validation_result['valid'] = len(compatibility_issues) == 0
            
        except Exception as e:
            validation_result['error'] = str(e)
        
        return validation_result
    
    def load_encoder_weights_only(self, 
                                 weights_path: str,
                                 strict: bool = False) -> bool:
        """
        Load only encoder weights from pre-trained model.
        
        Useful for transfer learning where only encoder is pre-trained.
        
        Args:
            weights_path: Path to pre-trained weights
            strict: Whether to strictly enforce key matching
            
        Returns:
            True if weights loaded successfully
        """
        try:
            if not os.path.exists(weights_path):
                logger.warning(f"Pre-trained weights not found: {weights_path}")
                return False
            
            # Load weights
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                full_state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                full_state_dict = checkpoint['state_dict']
            else:
                full_state_dict = checkpoint
            
            # Filter only encoder weights
            encoder_state_dict = {}
            for key, value in full_state_dict.items():
                if key.startswith('encoder.'):
                    encoder_state_dict[key] = value
            
            if not encoder_state_dict:
                logger.warning("No encoder weights found in checkpoint")
                return False
            
            # Load only encoder weights
            current_state = self.model.state_dict()
            for key, value in encoder_state_dict.items():
                if key in current_state:
                    current_state[key] = value
            
            self.model.load_state_dict(current_state, strict=strict)
            logger.info(f"Encoder weights loaded from {weights_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading encoder weights: {str(e)}")
            return False
    
    def create_pretrained_weight_info(self, weights_path: str) -> Dict[str, Any]:
        """
        Extract information about pre-trained weights.
        
        Args:
            weights_path: Path to pre-trained weights
            
        Returns:
            Dictionary containing weight information
        """
        info = {
            'file_path': weights_path,
            'file_exists': os.path.exists(weights_path),
            'file_size': None,
            'model_config': {},
            'training_info': {},
            'layer_info': {},
            'total_parameters': 0,
            'error': None
        }
        
        try:
            if not info['file_exists']:
                info['error'] = "File not found"
                return info
            
            # Get file size
            info['file_size'] = os.path.getsize(weights_path)
            
            # Load checkpoint
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
            
            # Extract model config
            if 'model_config' in checkpoint:
                info['model_config'] = checkpoint['model_config']
            
            # Extract training info
            training_keys = ['epoch', 'loss', 'timestamp', 'pytorch_version']
            for key in training_keys:
                if key in checkpoint:
                    info['training_info'][key] = checkpoint[key]
            
            # Get state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Analyze layers
            layer_counts = {}
            total_params = 0
            
            for key, tensor in state_dict.items():
                layer_type = key.split('.')[0] if '.' in key else 'root'
                if layer_type not in layer_counts:
                    layer_counts[layer_type] = 0
                layer_counts[layer_type] += 1
                total_params += tensor.numel()
            
            info['layer_info'] = layer_counts
            info['total_parameters'] = total_params
            
        except Exception as e:
            info['error'] = str(e)
        
        return info
    
    def _save_checkpoint_metadata(self, checkpoint_path: str, checkpoint: Dict) -> None:
        """Save human-readable checkpoint metadata."""
        metadata_path = checkpoint_path.replace('.pth', '_metadata.json')
        
        metadata = {
            'checkpoint_path': checkpoint_path,
            'timestamp': checkpoint.get('timestamp'),
            'epoch': checkpoint.get('epoch'),
            'loss': checkpoint.get('loss'),
            'model_config': checkpoint.get('model_config', {}),
            'pytorch_version': checkpoint.get('pytorch_version'),
        }
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save metadata: {str(e)}")


class CheckpointManager:
    """
    High-level checkpoint management utilities.
    
    Provides convenient methods for managing training checkpoints,
    including automatic saving, loading, and cleanup.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 checkpoint_dir: str = "checkpoints",
                 save_frequency: int = 10):
        """
        Initialize checkpoint manager.
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer (optional)
            checkpoint_dir: Directory for checkpoints
            save_frequency: Save checkpoint every N epochs
        """
        self.weight_manager = ModelWeightManager(model, checkpoint_dir)
        self.optimizer = optimizer
        self.save_frequency = save_frequency
        self.best_loss = float('inf')
        self.best_checkpoint_path = None
    
    def save_checkpoint_if_needed(self, 
                                 epoch: int,
                                 loss: float,
                                 force_save: bool = False) -> Optional[str]:
        """
        Save checkpoint if conditions are met.
        
        Args:
            epoch: Current epoch
            loss: Current loss
            force_save: Force save regardless of frequency
            
        Returns:
            Path to saved checkpoint or None
        """
        should_save = force_save or (epoch % self.save_frequency == 0)
        
        if should_save:
            checkpoint_path = self.weight_manager.create_checkpoint_name(epoch, loss)
            
            optimizer_state = None
            if self.optimizer is not None:
                optimizer_state = self.optimizer.state_dict()
            
            self.weight_manager.save_checkpoint(
                checkpoint_path,
                epoch=epoch,
                loss=loss,
                optimizer_state=optimizer_state
            )
            
            return checkpoint_path
        
        return None
    
    def save_best_checkpoint(self, epoch: int, loss: float) -> Optional[str]:
        """
        Save checkpoint if it's the best so far.
        
        Args:
            epoch: Current epoch
            loss: Current loss
            
        Returns:
            Path to saved checkpoint or None
        """
        if loss < self.best_loss:
            self.best_loss = loss
            
            checkpoint_path = self.weight_manager.create_checkpoint_name(
                epoch, loss, prefix="best_checkpoint"
            )
            
            optimizer_state = None
            if self.optimizer is not None:
                optimizer_state = self.optimizer.state_dict()
            
            self.weight_manager.save_checkpoint(
                checkpoint_path,
                epoch=epoch,
                loss=loss,
                optimizer_state=optimizer_state,
                metadata={'is_best': True}
            )
            
            self.best_checkpoint_path = checkpoint_path
            return checkpoint_path
        
        return None
    
    def resume_training(self, checkpoint_path: Optional[str] = None) -> Tuple[int, float]:
        """
        Resume training from checkpoint.
        
        Args:
            checkpoint_path: Specific checkpoint path, or None for latest
            
        Returns:
            Tuple of (start_epoch, best_loss)
        """
        if checkpoint_path is None:
            checkpoint_path = self.weight_manager.get_latest_checkpoint()
        
        if checkpoint_path is None:
            logger.info("No checkpoint found, starting from scratch")
            return 0, float('inf')
        
        checkpoint_info = self.weight_manager.load_checkpoint(
            checkpoint_path, 
            load_optimizer=(self.optimizer is not None)
        )
        
        # Load optimizer state if available
        if self.optimizer is not None and 'optimizer_state_dict' in checkpoint_info:
            self.optimizer.load_state_dict(checkpoint_info['optimizer_state_dict'])
        
        start_epoch = checkpoint_info.get('epoch', 0) + 1
        self.best_loss = checkpoint_info.get('loss', float('inf'))
        
        logger.info(f"Resumed training from epoch {start_epoch}, loss: {self.best_loss}")
        return start_epoch, self.best_loss