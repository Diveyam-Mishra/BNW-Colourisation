"""
Training pipeline for image colorization model.

This module implements the ColorizationTrainer class that handles the complete
training loop including forward/backward passes, validation, and gradient computation.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from src.models.unet import UNetColorizer
from src.training.losses import HybridLoss
from src.data.models import ModelConfig, TrainingConfig
from src.utils.color_converter import ColorSpaceConverter

# Configure logging
logger = logging.getLogger(__name__)


class TrainingMetrics:
    """
    Class to track and manage training metrics.
    """
    
    def __init__(self):
        """Initialize training metrics tracker."""
        self.reset()
        
    def reset(self):
        """Reset all metrics."""
        self.total_loss = 0.0
        self.l1_loss = 0.0
        self.perceptual_loss = 0.0
        self.num_batches = 0
        self.batch_losses = []
        
    def update(self, total_loss: float, l1_loss: float, perceptual_loss: float):
        """
        Update metrics with batch results.
        
        Args:
            total_loss: Total loss for the batch
            l1_loss: L1 loss component for the batch
            perceptual_loss: Perceptual loss component for the batch
        """
        self.total_loss += total_loss
        self.l1_loss += l1_loss
        self.perceptual_loss += perceptual_loss
        self.num_batches += 1
        self.batch_losses.append(total_loss)
        
    def get_averages(self) -> Dict[str, float]:
        """
        Get average metrics.
        
        Returns:
            Dictionary containing average metrics
        """
        if self.num_batches == 0:
            return {
                'avg_total_loss': 0.0,
                'avg_l1_loss': 0.0,
                'avg_perceptual_loss': 0.0
            }
            
        return {
            'avg_total_loss': self.total_loss / self.num_batches,
            'avg_l1_loss': self.l1_loss / self.num_batches,
            'avg_perceptual_loss': self.perceptual_loss / self.num_batches
        }


class ColorizationTrainer:
    """
    Main trainer class for image colorization model.
    
    Handles the complete training pipeline including forward and backward passes,
    validation loops, gradient computation, and optimizer steps.
    """
    
    def __init__(self,
                 model: UNetColorizer,
                 loss_fn: HybridLoss,
                 optimizer: torch.optim.Optimizer,
                 device: Optional[torch.device] = None,
                 model_config: Optional[ModelConfig] = None,
                 training_config: Optional[TrainingConfig] = None):
        """
        Initialize the colorization trainer.
        
        Args:
            model: U-Net colorization model
            loss_fn: Hybrid loss function
            optimizer: Optimizer for training
            device: Device to run training on (CPU/GPU)
            model_config: Model configuration parameters
            training_config: Training configuration parameters
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Move model and loss function to device
        self.model.to(self.device)
        self.loss_fn.to(self.device)
        
        # Store configurations
        self.model_config = model_config
        self.training_config = training_config
        
        # Initialize color converter for processing
        self.color_converter = ColorSpaceConverter()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'l1_loss': [],
            'perceptual_loss': []
        }
        
        logger.info(f"Initialized trainer on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare a batch for training/validation.
        
        Args:
            batch: Batch dictionary from dataloader
            
        Returns:
            Tuple of (l_channel, ab_target, lab_target) tensors on device
        """
        # Extract tensors and move to device
        l_channel = batch['l_channel'].to(self.device)
        ab_channels = batch['ab_channels'].to(self.device)
        lab_image = batch['lab_image'].to(self.device)
        
        return l_channel, ab_channels, lab_image
        
    def _forward_pass(self, l_channel: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass through the model.
        
        Args:
            l_channel: L channel input tensor
            
        Returns:
            Predicted ab channels
        """
        return self.model(l_channel)
        
    def _compute_loss(self, 
                     predicted_ab: torch.Tensor,
                     target_ab: torch.Tensor,
                     l_channel: torch.Tensor,
                     target_lab: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute loss for the batch.
        
        Args:
            predicted_ab: Predicted ab channels
            target_ab: Target ab channels
            l_channel: L channel input
            target_lab: Target LAB image
            
        Returns:
            Tuple of (total_loss, l1_loss, perceptual_loss)
        """
        # Combine L channel with predicted ab channels for perceptual loss
        predicted_lab = torch.cat([l_channel, predicted_ab], dim=1)
        
        # Compute hybrid loss
        total_loss, l1_loss, perceptual_loss = self.loss_fn(
            predicted_ab, target_ab, predicted_lab, target_lab
        )
        
        return total_loss, l1_loss, perceptual_loss
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Batch of training data
            
        Returns:
            Dictionary containing loss values for the step
        """
        # Set model to training mode
        self.model.train()
        
        # Prepare batch
        l_channel, target_ab, target_lab = self._prepare_batch(batch)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        predicted_ab = self._forward_pass(l_channel)
        
        # Compute loss
        total_loss, l1_loss, perceptual_loss = self._compute_loss(
            predicted_ab, target_ab, l_channel, target_lab
        )
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        # Return loss values
        return {
            'total_loss': total_loss.item(),
            'l1_loss': l1_loss.item(),
            'perceptual_loss': perceptual_loss.item()
        }
        
    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single validation step.
        
        Args:
            batch: Batch of validation data
            
        Returns:
            Dictionary containing loss values for the step
        """
        # Set model to evaluation mode
        self.model.eval()
        
        with torch.no_grad():
            # Prepare batch
            l_channel, target_ab, target_lab = self._prepare_batch(batch)
            
            # Forward pass
            predicted_ab = self._forward_pass(l_channel)
            
            # Compute loss
            total_loss, l1_loss, perceptual_loss = self._compute_loss(
                predicted_ab, target_ab, l_channel, target_lab
            )
            
            # Return loss values
            return {
                'total_loss': total_loss.item(),
                'l1_loss': l1_loss.item(),
                'perceptual_loss': perceptual_loss.item()
            }
            
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one complete epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Dictionary containing average loss values for the epoch
        """
        metrics = TrainingMetrics()
        
        # Set model to training mode
        self.model.train()
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Perform training step
                step_losses = self.train_step(batch)
                
                # Update metrics
                metrics.update(
                    step_losses['total_loss'],
                    step_losses['l1_loss'],
                    step_losses['perceptual_loss']
                )
                
                # Log progress
                if self.training_config and batch_idx % self.training_config.log_frequency == 0:
                    logger.info(
                        f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(dataloader)}: "
                        f"Loss = {step_losses['total_loss']:.4f} "
                        f"(L1: {step_losses['l1_loss']:.4f}, "
                        f"Perceptual: {step_losses['perceptual_loss']:.4f})"
                    )
                    
            except Exception as e:
                logger.error(f"Error in training step {batch_idx}: {e}")
                raise
                
        epoch_time = time.time() - start_time
        avg_metrics = metrics.get_averages()
        
        logger.info(
            f"Training epoch {self.current_epoch} completed in {epoch_time:.2f}s. "
            f"Avg Loss: {avg_metrics['avg_total_loss']:.4f}"
        )
        
        return avg_metrics
        
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate the model on validation dataset.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dictionary containing average validation loss values
        """
        metrics = TrainingMetrics()
        
        # Set model to evaluation mode
        self.model.eval()
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                try:
                    # Perform validation step
                    step_losses = self.validation_step(batch)
                    
                    # Update metrics
                    metrics.update(
                        step_losses['total_loss'],
                        step_losses['l1_loss'],
                        step_losses['perceptual_loss']
                    )
                    
                except Exception as e:
                    logger.error(f"Error in validation step {batch_idx}: {e}")
                    raise
                    
        validation_time = time.time() - start_time
        avg_metrics = metrics.get_averages()
        
        logger.info(
            f"Validation completed in {validation_time:.2f}s. "
            f"Avg Loss: {avg_metrics['avg_total_loss']:.4f}"
        )
        
        return avg_metrics
        
    def fit(self, 
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
            num_epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Number of epochs to train (uses config if None)
            
        Returns:
            Training history dictionary
        """
        if num_epochs is None:
            if self.model_config is not None:
                num_epochs = self.model_config.num_epochs
            else:
                num_epochs = 10
                logger.warning("No num_epochs specified, defaulting to 10")
                
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_metrics = self.train_epoch(train_dataloader)
            
            # Validate
            val_metrics = self.validate(val_dataloader)
            
            # Update training history
            self.training_history['train_loss'].append(train_metrics['avg_total_loss'])
            self.training_history['val_loss'].append(val_metrics['avg_total_loss'])
            self.training_history['l1_loss'].append(train_metrics['avg_l1_loss'])
            self.training_history['perceptual_loss'].append(train_metrics['avg_perceptual_loss'])
            
            # Check for best validation loss
            current_val_loss = val_metrics['avg_total_loss']
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                logger.info(f"New best validation loss: {self.best_val_loss:.4f}")
                
            # Log epoch summary
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_metrics['avg_total_loss']:.4f}, "
                f"Val Loss: {val_metrics['avg_total_loss']:.4f}"
            )
            
        logger.info("Training completed!")
        return self.training_history
        
    def get_training_state(self) -> Dict[str, Any]:
        """
        Get current training state for checkpointing.
        
        Returns:
            Dictionary containing training state
        """
        return {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'model_config': self.model_config.__dict__ if self.model_config else None,
            'training_config': self.training_config.__dict__ if self.training_config else None
        }
        
    def load_training_state(self, checkpoint_path: str) -> None:
        """
        Load training state from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint loading fails
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load training state
            self.current_epoch = checkpoint.get('epoch', 0)
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.training_history = checkpoint.get('training_history', {
                'train_loss': [], 'val_loss': [], 'l1_loss': [], 'perceptual_loss': []
            })
            
            logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
            
        except Exception as e:
            raise RuntimeError(f"Error loading checkpoint: {e}")
            
    def get_model_predictions(self, dataloader: DataLoader, num_samples: int = 5) -> List[Dict[str, torch.Tensor]]:
        """
        Get model predictions for visualization/analysis.
        
        Args:
            dataloader: Data loader to get samples from
            num_samples: Number of samples to predict
            
        Returns:
            List of dictionaries containing input, target, and prediction tensors
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if len(predictions) >= num_samples:
                    break
                    
                l_channel, target_ab, target_lab = self._prepare_batch(batch)
                predicted_ab = self._forward_pass(l_channel)
                
                # Move tensors back to CPU for storage
                predictions.append({
                    'l_channel': l_channel.cpu(),
                    'target_ab': target_ab.cpu(),
                    'predicted_ab': predicted_ab.cpu(),
                    'target_lab': target_lab.cpu()
                })
                
        return predictions[:num_samples]
        
    def compute_gradient_norm(self) -> float:
        """
        Compute the norm of gradients for monitoring training stability.
        
        Returns:
            Gradient norm value
        """
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm