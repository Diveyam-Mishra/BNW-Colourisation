"""
Training monitoring and checkpointing utilities for image colorization.

This module provides classes for tracking training progress, implementing
early stopping, and managing model checkpoints during training.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import logging
from dataclasses import dataclass, asdict
import torch
import numpy as np
import matplotlib.pyplot as plt

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """
    Data class to store training metrics for a single epoch.
    """
    epoch: int
    train_loss: float
    val_loss: float
    l1_loss: float
    perceptual_loss: float
    learning_rate: float
    epoch_time: float
    timestamp: float


class TrainingLogger:
    """
    Logger for tracking and persisting training metrics and progress.
    """
    
    def __init__(self, log_dir: str, experiment_name: str = "colorization_training"):
        """
        Initialize training logger.
        
        Args:
            log_dir: Directory to save logs and metrics
            experiment_name: Name of the experiment for organizing logs
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        
        # Create log directory structure
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics_history: List[TrainingMetrics] = []
        self.metrics_file = self.experiment_dir / "training_metrics.json"
        
        # Set up file logging
        self._setup_file_logging()
        
        logger.info(f"Training logger initialized: {self.experiment_dir}")
        
    def _setup_file_logging(self):
        """Set up file logging for training progress."""
        log_file = self.experiment_dir / "training.log"
        
        # Create file handler
        self.file_handler = logging.FileHandler(log_file)
        self.file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.training_logger = logging.getLogger('src.training')
        self.training_logger.addHandler(self.file_handler)
        self.training_logger.setLevel(logging.INFO)
        
    def close(self):
        """Close file handlers to release resources."""
        if hasattr(self, 'file_handler') and self.file_handler:
            self.training_logger.removeHandler(self.file_handler)
            self.file_handler.close()
            self.file_handler = None
        
    def log_epoch(self, 
                  epoch: int,
                  train_loss: float,
                  val_loss: float,
                  l1_loss: float,
                  perceptual_loss: float,
                  learning_rate: float,
                  epoch_time: float):
        """
        Log metrics for a completed epoch.
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Validation loss
            l1_loss: L1 loss component
            perceptual_loss: Perceptual loss component
            learning_rate: Current learning rate
            epoch_time: Time taken for the epoch
        """
        metrics = TrainingMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            l1_loss=l1_loss,
            perceptual_loss=perceptual_loss,
            learning_rate=learning_rate,
            epoch_time=epoch_time,
            timestamp=time.time()
        )
        
        self.metrics_history.append(metrics)
        
        # Log to console
        logger.info(
            f"Epoch {epoch:3d} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"L1: {l1_loss:.4f} | "
            f"Perceptual: {perceptual_loss:.4f} | "
            f"LR: {learning_rate:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        # Save metrics to file
        self._save_metrics()
        
    def _save_metrics(self):
        """Save metrics history to JSON file."""
        try:
            metrics_data = [asdict(metric) for metric in self.metrics_history]
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
            
    def load_metrics(self) -> List[TrainingMetrics]:
        """
        Load metrics history from file.
        
        Returns:
            List of TrainingMetrics objects
        """
        if not self.metrics_file.exists():
            return []
            
        try:
            with open(self.metrics_file, 'r') as f:
                metrics_data = json.load(f)
                
            self.metrics_history = [
                TrainingMetrics(**metric_dict) for metric_dict in metrics_data
            ]
            
            return self.metrics_history
            
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
            return []
            
    def plot_training_curves(self, save_path: Optional[str] = None):
        """
        Plot training curves for loss metrics.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.metrics_history:
            logger.warning("No metrics to plot")
            return
            
        epochs = [m.epoch for m in self.metrics_history]
        train_losses = [m.train_loss for m in self.metrics_history]
        val_losses = [m.val_loss for m in self.metrics_history]
        l1_losses = [m.l1_loss for m in self.metrics_history]
        perceptual_losses = [m.perceptual_loss for m in self.metrics_history]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Training and validation loss
        ax1.plot(epochs, train_losses, label='Training Loss', color='blue')
        ax1.plot(epochs, val_losses, label='Validation Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # L1 loss component
        ax2.plot(epochs, l1_losses, label='L1 Loss', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('L1 Loss')
        ax2.set_title('L1 Loss Component')
        ax2.legend()
        ax2.grid(True)
        
        # Perceptual loss component
        ax3.plot(epochs, perceptual_losses, label='Perceptual Loss', color='orange')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Perceptual Loss')
        ax3.set_title('Perceptual Loss Component')
        ax3.legend()
        ax3.grid(True)
        
        # Learning rate
        learning_rates = [m.learning_rate for m in self.metrics_history]
        ax4.plot(epochs, learning_rates, label='Learning Rate', color='purple')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.legend()
        ax4.grid(True)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.experiment_dir / "training_curves.png"
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training curves saved to {save_path}")
        
    def get_best_epoch(self, metric: str = 'val_loss') -> Optional[TrainingMetrics]:
        """
        Get the epoch with the best (lowest) metric value.
        
        Args:
            metric: Metric name to optimize ('val_loss', 'train_loss', etc.)
            
        Returns:
            TrainingMetrics object for the best epoch, or None if no metrics
        """
        if not self.metrics_history:
            return None
            
        return min(self.metrics_history, key=lambda m: getattr(m, metric))


class EarlyStopping:
    """
    Early stopping utility to prevent overfitting during training.
    """
    
    def __init__(self, 
                 patience: int = 10,
                 min_delta: float = 0.0,
                 metric: str = 'val_loss',
                 mode: str = 'min',
                 restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            metric: Metric to monitor for early stopping
            mode: 'min' for metrics that should decrease, 'max' for metrics that should increase
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        
        # Set comparison function based on mode
        if mode == 'min':
            self.is_better = lambda current, best: current < best - min_delta
        elif mode == 'max':
            self.is_better = lambda current, best: current > best + min_delta
        else:
            raise ValueError(f"Mode {mode} is unknown, use 'min' or 'max'")
            
        logger.info(f"Early stopping initialized: patience={patience}, metric={metric}, mode={mode}")
        
    def __call__(self, current_score: float, model: torch.nn.Module) -> bool:
        """
        Check if training should be stopped early.
        
        Args:
            current_score: Current value of the monitored metric
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = current_score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif self.is_better(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
            
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                logger.info("Restored best model weights")
                
        return self.early_stop
        
    def reset(self):
        """Reset early stopping state."""
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False


class CheckpointManager:
    """
    Manager for saving and loading model checkpoints during training.
    """
    
    def __init__(self, 
                 checkpoint_dir: str,
                 save_frequency: int = 10,
                 max_checkpoints: int = 5,
                 save_best_only: bool = False):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_frequency: Frequency of checkpoint saving (epochs)
            max_checkpoints: Maximum number of checkpoints to keep
            save_best_only: Whether to save only the best checkpoint
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_frequency = save_frequency
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        
        self.best_score = float('inf')
        self.checkpoint_files: List[Path] = []
        
        logger.info(f"Checkpoint manager initialized: {self.checkpoint_dir}")
        
    def save_checkpoint(self,
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       loss: float,
                       metrics: Optional[Dict[str, Any]] = None,
                       is_best: bool = False) -> str:
        """
        Save a model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            loss: Current loss value
            metrics: Additional metrics to save
            is_best: Whether this is the best checkpoint so far
            
        Returns:
            Path to the saved checkpoint file
        """
        # Create checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'timestamp': time.time()
        }
        
        if metrics:
            checkpoint['metrics'] = metrics
            
        # Determine checkpoint filename
        if is_best or self.save_best_only:
            checkpoint_path = self.checkpoint_dir / "best_checkpoint.pth"
            if is_best:
                self.best_score = loss
        else:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pth"
            
        # Save checkpoint
        try:
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Track checkpoint files for cleanup
            if not self.save_best_only and checkpoint_path not in self.checkpoint_files:
                self.checkpoint_files.append(checkpoint_path)
                self._cleanup_old_checkpoints()
                
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            raise
            
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to maintain max_checkpoints limit."""
        if len(self.checkpoint_files) > self.max_checkpoints:
            # Sort by modification time and remove oldest
            self.checkpoint_files.sort(key=lambda p: p.stat().st_mtime)
            
            while len(self.checkpoint_files) > self.max_checkpoints:
                old_checkpoint = self.checkpoint_files.pop(0)
                try:
                    if old_checkpoint.exists():
                        old_checkpoint.unlink()
                        logger.info(f"Removed old checkpoint: {old_checkpoint}")
                except Exception as e:
                    logger.warning(f"Error removing old checkpoint {old_checkpoint}: {e}")
                    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load a checkpoint from file.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary containing checkpoint data
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint loading fails
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return checkpoint
            
        except Exception as e:
            raise RuntimeError(f"Error loading checkpoint: {e}")
            
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get the path to the most recent checkpoint.
        
        Returns:
            Path to latest checkpoint, or None if no checkpoints exist
        """
        checkpoint_pattern = "checkpoint_epoch_*.pth"
        checkpoint_files = list(self.checkpoint_dir.glob(checkpoint_pattern))
        
        if not checkpoint_files:
            return None
            
        # Sort by epoch number and return the latest
        latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        return str(latest_checkpoint)
        
    def get_best_checkpoint(self) -> Optional[str]:
        """
        Get the path to the best checkpoint.
        
        Returns:
            Path to best checkpoint, or None if it doesn't exist
        """
        best_checkpoint = self.checkpoint_dir / "best_checkpoint.pth"
        return str(best_checkpoint) if best_checkpoint.exists() else None
        
    def should_save_checkpoint(self, epoch: int, is_best: bool = False) -> bool:
        """
        Determine if a checkpoint should be saved at the current epoch.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best checkpoint so far
            
        Returns:
            True if checkpoint should be saved
        """
        if self.save_best_only:
            return is_best
            
        return epoch % self.save_frequency == 0 or is_best


class TrainingMonitor:
    """
    Comprehensive training monitor that combines logging, early stopping, and checkpointing.
    """
    
    def __init__(self,
                 log_dir: str,
                 checkpoint_dir: str,
                 experiment_name: str = "colorization_training",
                 early_stopping_patience: int = 15,
                 checkpoint_frequency: int = 10,
                 save_best_only: bool = False):
        """
        Initialize comprehensive training monitor.
        
        Args:
            log_dir: Directory for logs and metrics
            checkpoint_dir: Directory for model checkpoints
            experiment_name: Name of the experiment
            early_stopping_patience: Patience for early stopping
            checkpoint_frequency: Frequency of checkpoint saving
            save_best_only: Whether to save only the best checkpoint
        """
        self.logger = TrainingLogger(log_dir, experiment_name)
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            save_frequency=checkpoint_frequency,
            save_best_only=save_best_only
        )
        
        self.best_val_loss = float('inf')
        
    def log_epoch(self,
                  epoch: int,
                  train_loss: float,
                  val_loss: float,
                  l1_loss: float,
                  perceptual_loss: float,
                  learning_rate: float,
                  epoch_time: float,
                  model: torch.nn.Module,
                  optimizer: torch.optim.Optimizer) -> bool:
        """
        Log epoch results and handle checkpointing and early stopping.
        
        Args:
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss
            l1_loss: L1 loss component
            perceptual_loss: Perceptual loss component
            learning_rate: Current learning rate
            epoch_time: Time taken for the epoch
            model: Model to checkpoint
            optimizer: Optimizer to checkpoint
            
        Returns:
            True if training should stop (early stopping triggered)
        """
        # Log metrics
        self.logger.log_epoch(
            epoch, train_loss, val_loss, l1_loss, 
            perceptual_loss, learning_rate, epoch_time
        )
        
        # Check if this is the best validation loss
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
            
        # Save checkpoint if needed
        if self.checkpoint_manager.should_save_checkpoint(epoch, is_best):
            self.checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_loss,
                metrics={
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'l1_loss': l1_loss,
                    'perceptual_loss': perceptual_loss,
                    'learning_rate': learning_rate
                },
                is_best=is_best
            )
            
        # Check early stopping
        should_stop = self.early_stopping(val_loss, model)
        
        return should_stop
        
    def finalize_training(self):
        """Finalize training by saving plots and summary."""
        # Save training curves
        self.logger.plot_training_curves()
        
        # Get best epoch info
        best_epoch = self.logger.get_best_epoch('val_loss')
        if best_epoch:
            logger.info(
                f"Best epoch: {best_epoch.epoch} with validation loss: {best_epoch.val_loss:.4f}"
            )
            
        logger.info("Training monitoring finalized")