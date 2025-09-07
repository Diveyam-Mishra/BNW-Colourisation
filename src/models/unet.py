"""
U-Net model implementation for image colorization.

This module contains the U-Net architecture with encoder-decoder structure
and skip connections for preserving spatial details during colorization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class EncoderBlock(nn.Module):
    """
    Encoder block for U-Net architecture.
    
    Each block consists of:
    - Two convolutional layers with BatchNorm and ReLU
    - MaxPool2D for downsampling
    - Progressive channel dimension increases
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize encoder block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super(EncoderBlock, self).__init__()
        
        # First convolution block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Second convolution block
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Downsampling layer
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder block.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (skip_connection_features, downsampled_features)
        """
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        skip_features = self.relu2(x)
        
        # Downsample for next layer
        downsampled = self.maxpool(skip_features)
        
        return skip_features, downsampled


class UNetEncoder(nn.Module):
    """
    Complete encoder path for U-Net architecture.
    
    Implements 5 downsampling blocks with increasing channel dimensions:
    64, 128, 256, 512, 1024
    """
    
    def __init__(self, input_channels: int = 1):
        """
        Initialize U-Net encoder.
        
        Args:
            input_channels: Number of input channels (1 for grayscale L channel)
        """
        super(UNetEncoder, self).__init__()
        
        # Define channel dimensions for each encoder block
        self.channel_dims = [64, 128, 256, 512, 1024]
        
        # Create encoder blocks
        self.encoder_blocks = nn.ModuleList()
        
        # First block takes input channels
        in_ch = input_channels
        for out_ch in self.channel_dims[:-1]:  # All but the last one
            self.encoder_blocks.append(EncoderBlock(in_ch, out_ch))
            in_ch = out_ch
        
        # Bottleneck block (no maxpool at the end)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.channel_dims[-2], self.channel_dims[-1], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel_dims[-1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel_dims[-1], self.channel_dims[-1], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channel_dims[-1]),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor of shape (batch_size, input_channels, height, width)
            
        Returns:
            Tuple of (skip_connections, bottleneck_features)
        """
        skip_connections = []
        
        # Pass through encoder blocks
        for encoder_block in self.encoder_blocks:
            skip_features, x = encoder_block(x)
            skip_connections.append(skip_features)
        
        # Pass through bottleneck
        bottleneck_features = self.bottleneck(x)
        
        return skip_connections, bottleneck_features


class DecoderBlock(nn.Module):
    """
    Decoder block for U-Net architecture.
    
    Each block consists of:
    - Upsampling with transposed convolution
    - Skip connection concatenation
    - Two convolutional layers with BatchNorm and ReLU
    - Progressive channel dimension decreases
    """
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        """
        Initialize decoder block.
        
        Args:
            in_channels: Number of input channels from previous decoder layer
            skip_channels: Number of channels from skip connection
            out_channels: Number of output channels
        """
        super(DecoderBlock, self).__init__()
        
        # Upsampling layer (transposed convolution)
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, 
            kernel_size=2, stride=2
        )
        
        # Convolutional layers after concatenation
        # Input channels = out_channels (from upconv) + skip_channels (from skip connection)
        concat_channels = out_channels + skip_channels
        
        # First convolution block
        self.conv1 = nn.Conv2d(concat_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Second convolution block
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder block.
        
        Args:
            x: Input tensor from previous decoder layer
            skip_connection: Skip connection tensor from encoder
            
        Returns:
            Output tensor after upsampling, concatenation, and convolution
        """
        # Upsample input
        x = self.upconv(x)
        
        # Concatenate with skip connection
        # Handle potential size mismatches due to padding
        if x.size() != skip_connection.size():
            # Crop or pad to match skip connection size
            diff_h = skip_connection.size(2) - x.size(2)
            diff_w = skip_connection.size(3) - x.size(3)
            
            if diff_h > 0 or diff_w > 0:
                # Pad x if it's smaller
                x = F.pad(x, (diff_w // 2, diff_w - diff_w // 2,
                             diff_h // 2, diff_h - diff_h // 2))
            elif diff_h < 0 or diff_w < 0:
                # Crop x if it's larger
                x = x[:, :, :skip_connection.size(2), :skip_connection.size(3)]
        
        # Concatenate along channel dimension
        x = torch.cat([x, skip_connection], dim=1)
        
        # Apply convolution blocks
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        return x


class UNetDecoder(nn.Module):
    """
    Complete decoder path for U-Net architecture.
    
    Implements 4 upsampling blocks with decreasing channel dimensions:
    512, 256, 128, 64
    """
    
    def __init__(self, bottleneck_channels: int = 1024):
        """
        Initialize U-Net decoder.
        
        Args:
            bottleneck_channels: Number of channels from bottleneck layer
        """
        super(UNetDecoder, self).__init__()
        
        # Define channel dimensions for decoder blocks
        # (in_channels, skip_channels, out_channels)
        self.decoder_configs = [
            (bottleneck_channels, 512, 512),  # First decoder block
            (512, 256, 256),                  # Second decoder block
            (256, 128, 128),                  # Third decoder block
            (128, 64, 64),                    # Fourth decoder block
        ]
        
        # Create decoder blocks
        self.decoder_blocks = nn.ModuleList()
        for in_ch, skip_ch, out_ch in self.decoder_configs:
            self.decoder_blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))
    
    def forward(self, bottleneck_features: torch.Tensor, 
                skip_connections: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            bottleneck_features: Features from encoder bottleneck
            skip_connections: List of skip connection features from encoder
            
        Returns:
            Decoded features tensor
        """
        x = bottleneck_features
        
        # Reverse skip connections to match decoder order
        # (encoder produces skip connections in forward order, decoder needs reverse)
        reversed_skips = skip_connections[::-1]
        
        # Pass through decoder blocks
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip_connection = reversed_skips[i]
            x = decoder_block(x, skip_connection)
        
        return x


class UNetColorizer(nn.Module):
    """
    Complete U-Net model for image colorization.
    
    Combines encoder and decoder into a complete architecture that takes
    grayscale L channel as input and predicts a, b channels in CIELAB space.
    """
    
    def __init__(self, input_channels: int = 1, output_channels: int = 2):
        """
        Initialize complete U-Net colorizer.
        
        Args:
            input_channels: Number of input channels (1 for grayscale L channel)
            output_channels: Number of output channels (2 for a, b channels)
        """
        super(UNetColorizer, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Initialize encoder and decoder
        self.encoder = UNetEncoder(input_channels=input_channels)
        self.decoder = UNetDecoder(bottleneck_channels=1024)
        
        # Output layer to produce final a, b channels
        self.output_layer = nn.Sequential(
            nn.Conv2d(64, output_channels, kernel_size=1),
            nn.Tanh()  # Constrain output to [-1, 1] range for a, b channels
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through complete U-Net.
        
        Args:
            x: Input tensor of shape (batch_size, input_channels, height, width)
               Expected to be L channel in CIELAB space, normalized to [0, 1]
            
        Returns:
            Output tensor of shape (batch_size, output_channels, height, width)
            Predicted a, b channels in CIELAB space, range [-1, 1]
        """
        # Encode input through encoder path
        skip_connections, bottleneck_features = self.encoder(x)
        
        # Decode through decoder path with skip connections
        decoded_features = self.decoder(bottleneck_features, skip_connections)
        
        # Generate final output (a, b channels)
        output = self.output_layer(decoded_features)
        
        return output
    
    def load_pretrained_weights(self, 
                               weights_path: str, 
                               strict: bool = False,
                               fallback_to_random: bool = True) -> bool:
        """
        Load pre-trained weights from file with enhanced validation and fallback.
        
        Args:
            weights_path: Path to the weights file
            strict: Whether to strictly enforce that the keys match
            fallback_to_random: Whether to fallback to random initialization on failure
            
        Returns:
            True if weights loaded successfully, False if fallback used
        """
        from .weight_manager import ModelWeightManager
        
        # Use the weight manager for enhanced functionality
        weight_manager = ModelWeightManager(self)
        return weight_manager.load_pretrained_weights(
            weights_path, 
            strict=strict, 
            fallback_to_random=fallback_to_random
        )
    
    def save_model(self, save_path: str, epoch: int = None, loss: float = None, 
                   optimizer_state: dict = None) -> None:
        """
        Save model weights and training state.
        
        Args:
            save_path: Path to save the model
            epoch: Current training epoch (optional)
            loss: Current loss value (optional)
            optimizer_state: Optimizer state dict (optional)
        """
        try:
            checkpoint = {
                'model_state_dict': self.state_dict(),
                'model_config': {
                    'input_channels': self.input_channels,
                    'output_channels': self.output_channels,
                }
            }
            
            # Add optional training information
            if epoch is not None:
                checkpoint['epoch'] = epoch
            if loss is not None:
                checkpoint['loss'] = loss
            if optimizer_state is not None:
                checkpoint['optimizer_state_dict'] = optimizer_state
            
            torch.save(checkpoint, save_path)
            print(f"Model saved to {save_path}")
            
        except Exception as e:
            raise RuntimeError(f"Error saving model: {str(e)}")
    
    def get_model_info(self) -> dict:
        """
        Get information about the model architecture.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'UNetColorizer',
            'input_channels': self.input_channels,
            'output_channels': self.output_channels,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'encoder_blocks': len(self.encoder.encoder_blocks),
            'decoder_blocks': len(self.decoder.decoder_blocks),
        }