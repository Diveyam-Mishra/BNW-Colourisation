#!/usr/bin/env python3
"""
Final Image Colorization Training Script
- 50 epochs
- Save weights after every epoch
- Calculate error after every epoch
- Memory optimized for 4GB GPU
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from pathlib import Path
from datetime import datetime
import time

# TPU support
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.unet import UNetColorizer
from src.training.dataset import ColorizationDataset
from src.training.losses import HybridLoss
from src.data.preprocessor import ImagePreprocessor

class ColorTrainer:
    def __init__(self, config_path="configs/training_config.yaml"):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device (TPU > CUDA > CPU)
        if TPU_AVAILABLE and xm.xla_device() is not None:
            self.device = xm.xla_device()
            self.is_tpu = True
            self.is_cuda = False
            print(f"Using TPU device: {self.device}")
            print(f"TPU cores: {xm.xrt_world_size()}")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.is_tpu = False
            self.is_cuda = True
            print(f"Using CUDA device: {self.device}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            self.device = torch.device("cpu")
            self.is_tpu = False
            self.is_cuda = False
            print(f"Using CPU device: {self.device}")
        
        # Create directories
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Setup model
        self.model = UNetColorizer(
            input_channels=1,
            output_channels=2
        ).to(self.device)
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        
        # Setup loss
        self.criterion = HybridLoss(
            l1_weight=self.config['l1_weight'],
            perceptual_weight=self.config['perceptual_weight']
        ).to(self.device)
        
        # Setup data
        self.setup_data()
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        
    def setup_data(self):
        """Setup training and validation data loaders."""
        dataset = ColorizationDataset(
            dataset_path=self.config['dataset_path'],
            target_size=tuple(self.config['input_size']),
            augmentation_enabled=self.config['augmentation_enabled']
        )
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        print(f"Dataset: {len(dataset):,} images")
        print(f"Training: {len(train_dataset):,} images")
        print(f"Validation: {len(val_dataset):,} images")
        
        # Create data loaders with device-specific optimization
        if self.is_tpu:
            batch_size = 8  # TPUs can handle larger batches
            num_workers = 0  # TPU doesn't need multiprocessing
            pin_memory = False
        elif self.is_cuda:
            batch_size = 2  # Reduced for 4GB GPU
            num_workers = 2
            pin_memory = True
        else:
            batch_size = 4  # CPU can handle moderate batches
            num_workers = 4
            pin_memory = False
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        
        # Wrap with TPU parallel loader if using TPU
        if self.is_tpu:
            self.train_loader = pl.ParallelLoader(train_loader, [self.device])
            self.val_loader = pl.ParallelLoader(val_loader, [self.device])
        else:
            self.train_loader = train_loader
            self.val_loader = val_loader
        
        print(f"Batch size: {batch_size}")
        print(f"Train batches: {len(self.train_loader):,}")
        print(f"Val batches: {len(self.val_loader):,}")
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        print(f"\nEpoch {epoch+1}/50 - Training...")
        start_time = time.time()
        
        # Get the actual loader for TPU or regular loader
        loader = self.train_loader.per_device_loader(self.device) if self.is_tpu else self.train_loader
        
        for batch_idx, batch in enumerate(loader):
            if self.is_tpu:
                grayscale = batch['l_channel'].to(self.device)
                color_ab = batch['ab_channels'].to(self.device)
            else:
                grayscale = batch['l_channel'].to(self.device, non_blocking=True)
                color_ab = batch['ab_channels'].to(self.device, non_blocking=True)
            
            # Forward pass
            self.optimizer.zero_grad()
            predicted_ab = self.model(grayscale)
            
            # Construct full LAB images for loss calculation
            predicted_lab = torch.cat([grayscale, predicted_ab], dim=1)
            target_lab = torch.cat([grayscale, color_ab], dim=1)
            
            total_loss, l1_loss, perceptual_loss = self.criterion(predicted_ab, color_ab, predicted_lab, target_lab)
            loss = total_loss
            
            # Backward pass
            loss.backward()
            
            if self.is_tpu:
                xm.optimizer_step(self.optimizer)
            else:
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Print progress every 1000 batches
            if (batch_idx + 1) % 1000 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"  Batch {batch_idx+1}/{num_batches}: Loss = {avg_loss:.4f}")
            
            # Clear cache periodically (only for CUDA)
            if self.is_cuda and (batch_idx + 1) % 500 == 0:
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches
        epoch_time = time.time() - start_time
        
        print(f"  Training completed: {avg_loss:.4f} loss in {epoch_time:.1f}s")
        return avg_loss
    
    def validate_epoch(self, epoch):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        print(f"Epoch {epoch+1}/50 - Validation...")
        start_time = time.time()
        
        # Get the actual loader for TPU or regular loader
        loader = self.val_loader.per_device_loader(self.device) if self.is_tpu else self.val_loader
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if self.is_tpu:
                    grayscale = batch['l_channel'].to(self.device)
                    color_ab = batch['ab_channels'].to(self.device)
                else:
                    grayscale = batch['l_channel'].to(self.device, non_blocking=True)
                    color_ab = batch['ab_channels'].to(self.device, non_blocking=True)
                
                predicted_ab = self.model(grayscale)
                
                # Construct full LAB images for loss calculation
                predicted_lab = torch.cat([grayscale, predicted_ab], dim=1)
                target_lab = torch.cat([grayscale, color_ab], dim=1)
                
                total_loss, l1_loss, perceptual_loss = self.criterion(predicted_ab, color_ab, predicted_lab, target_lab)
                loss = total_loss
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        epoch_time = time.time() - start_time
        
        print(f"  Validation completed: {avg_loss:.4f} loss in {epoch_time:.1f}s")
        return avg_loss
    
    def save_checkpoint(self, epoch, train_loss, val_loss):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': self.config
        }
        
        # Save epoch checkpoint
        checkpoint_path = f"checkpoints/epoch_{epoch+1:02d}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if not hasattr(self, 'best_val_loss') or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(checkpoint, "checkpoints/best_model.pth")
            print(f"  ✓ New best model saved (val_loss: {val_loss:.4f})")
        
        print(f"  ✓ Checkpoint saved: {checkpoint_path}")
    
    def log_metrics(self, epoch, train_loss, val_loss):
        """Log training metrics."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        # Write to log file
        log_entry = f"{datetime.now().isoformat()},Epoch,{epoch+1},Train,{train_loss:.6f},Val,{val_loss:.6f}\n"
        with open("logs/training_log.csv", "a") as f:
            f.write(log_entry)
        
        # Print summary
        print(f"  📊 Epoch {epoch+1:2d}: Train={train_loss:.4f}, Val={val_loss:.4f}")
        
        if hasattr(self, 'best_val_loss'):
            print(f"      Best Val Loss: {self.best_val_loss:.4f}")
    
    def train(self):
        """Main training loop."""
        print("=" * 60)
        print("🚀 Starting Image Colorization Training")
        print("=" * 60)
        
        # Initialize log file
        with open("logs/training_log.csv", "w") as f:
            f.write("timestamp,type,epoch,metric,train_loss,val_metric,val_loss\n")
        
        start_time = time.time()
        
        for epoch in range(50):
            print(f"\n{'='*20} EPOCH {epoch+1}/50 {'='*20}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate_epoch(epoch)
            
            # Save checkpoint
            self.save_checkpoint(epoch, train_loss, val_loss)
            
            # Log metrics
            self.log_metrics(epoch, train_loss, val_loss)
            
            # Memory cleanup (only for CUDA)
            if self.is_cuda:
                torch.cuda.empty_cache()
            elif self.is_tpu:
                xm.mark_step()  # TPU step marking
        
        total_time = time.time() - start_time
        print(f"\n🎉 Training completed in {total_time/3600:.1f} hours!")
        print(f"📁 Checkpoints saved in: checkpoints/")
        print(f"📊 Training log saved in: logs/training_log.csv")

def main():
    trainer = ColorTrainer()
    trainer.train()

if __name__ == "__main__":
    main()