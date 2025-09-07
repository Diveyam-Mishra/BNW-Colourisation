"""
Loss functions for image colorization training.

This module implements various loss functions used in the colorization model training,
including L1 loss, perceptual loss, and hybrid loss combinations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import torchvision.models as models
from torchvision.models import VGG16_Weights


class L1Loss(nn.Module):
    """
    L1 loss component for colorization training in CIELAB space.
    
    Implements per-pixel L1 loss calculation with optional normalization
    and weighting functions for the 'a' and 'b' channels in CIELAB color space.
    """
    
    def __init__(self, reduction: str = 'mean', weight: float = 1.0):
        """
        Initialize L1 loss component.
        
        Args:
            reduction: Specifies the reduction to apply to the output ('mean', 'sum', 'none')
            weight: Weight factor for the L1 loss component
        """
        super(L1Loss, self).__init__()
        self.reduction = reduction
        self.weight = weight
        
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate L1 loss between predicted and target 'a' and 'b' channels.
        
        Args:
            predicted: Predicted 'a' and 'b' channels, shape (batch_size, 2, height, width)
            target: Ground truth 'a' and 'b' channels, shape (batch_size, 2, height, width)
            
        Returns:
            L1 loss value as a scalar tensor
            
        Raises:
            ValueError: If input tensors have incompatible shapes
        """
        # Check tensor dimensions first
        if predicted.dim() != 4 or predicted.shape[1] != 2:
            raise ValueError(f"Expected 4D tensor with 2 channels, got shape {predicted.shape}")
            
        if target.dim() != 4 or target.shape[1] != 2:
            raise ValueError(f"Expected 4D tensor with 2 channels, got shape {target.shape}")
            
        # Check shape compatibility
        if predicted.shape != target.shape:
            raise ValueError(f"Shape mismatch: predicted {predicted.shape} vs target {target.shape}")
        
        # Calculate per-pixel L1 loss
        l1_loss = F.l1_loss(predicted, target, reduction=self.reduction)
        
        # Apply weighting
        weighted_loss = self.weight * l1_loss
        
        return weighted_loss
    
    def normalize_loss(self, loss: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Normalize loss by batch size if needed.
        
        Args:
            loss: Loss tensor to normalize
            batch_size: Current batch size
            
        Returns:
            Normalized loss tensor
        """
        if self.reduction == 'sum':
            return loss / batch_size
        return loss


class VGGFeatureExtractor(nn.Module):
    """
    VGG feature extractor for perceptual loss calculation.
    
    Uses pre-trained VGG16 network to extract features at multiple layers
    for computing perceptual loss in colorization training.
    """
    
    def __init__(self, feature_layers: Tuple[int, ...] = (3, 8, 15, 22)):
        """
        Initialize VGG feature extractor.
        
        Args:
            feature_layers: Tuple of layer indices to extract features from
                          Default corresponds to relu1_2, relu2_2, relu3_3, relu4_3
        """
        super(VGGFeatureExtractor, self).__init__()
        
        # Load pre-trained VGG16
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features
        
        # Freeze VGG parameters
        for param in self.features.parameters():
            param.requires_grad = False
            
        self.feature_layers = feature_layers
        
        # Set to evaluation mode
        self.eval()
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Extract features from specified VGG layers.
        
        Args:
            x: Input tensor, shape (batch_size, 3, height, width)
            
        Returns:
            Tuple of feature tensors from specified layers
        """
        features = []
        
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
                
        return tuple(features)


class PerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss for colorization training.
    
    Computes perceptual loss using features extracted from a pre-trained VGG network.
    The loss measures the difference between predicted and target images in feature space.
    """
    
    def __init__(self, 
                 feature_layers: Tuple[int, ...] = (3, 8, 15, 22),
                 layer_weights: Optional[Tuple[float, ...]] = None,
                 weight: float = 1.0):
        """
        Initialize perceptual loss.
        
        Args:
            feature_layers: VGG layer indices to use for feature extraction
            layer_weights: Weights for each feature layer (if None, uses equal weights)
            weight: Overall weight for the perceptual loss component
        """
        super(PerceptualLoss, self).__init__()
        
        self.feature_extractor = VGGFeatureExtractor(feature_layers)
        self.weight = weight
        
        if layer_weights is None:
            self.layer_weights = [1.0] * len(feature_layers)
        else:
            if len(layer_weights) != len(feature_layers):
                raise ValueError("Number of layer weights must match number of feature layers")
            self.layer_weights = list(layer_weights)
            
    def _prepare_input(self, lab_tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert LAB tensor to RGB for VGG processing.
        
        Args:
            lab_tensor: LAB color space tensor, shape (batch_size, 3, height, width)
                       where channels are [L, a, b]
            
        Returns:
            RGB tensor normalized for VGG input
        """
        # Convert LAB to RGB (simplified conversion for feature extraction)
        # Note: This is a simplified conversion. In practice, you might want
        # to use the proper LAB to RGB conversion from color_converter.py
        
        # Normalize L channel from [0, 100] to [0, 1]
        l_channel = lab_tensor[:, 0:1, :, :] / 100.0
        
        # Normalize a, b channels from [-128, 127] to [-1, 1] then to [0, 1]
        a_channel = (lab_tensor[:, 1:2, :, :] + 128.0) / 255.0
        b_channel = (lab_tensor[:, 2:3, :, :] + 128.0) / 255.0
        
        # Create RGB approximation (this is simplified)
        rgb_approx = torch.cat([l_channel, a_channel, b_channel], dim=1)
        
        # Normalize for VGG (ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(rgb_approx.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(rgb_approx.device)
        
        normalized = (rgb_approx - mean) / std
        
        return normalized
        
    def forward(self, predicted_lab: torch.Tensor, target_lab: torch.Tensor) -> torch.Tensor:
        """
        Calculate perceptual loss between predicted and target LAB images.
        
        Args:
            predicted_lab: Predicted LAB image, shape (batch_size, 3, height, width)
            target_lab: Target LAB image, shape (batch_size, 3, height, width)
            
        Returns:
            Perceptual loss value as a scalar tensor
        """
        # Check tensor dimensions first
        if predicted_lab.dim() != 4 or predicted_lab.shape[1] != 3:
            raise ValueError(f"Expected 4D tensor with 3 channels, got shape {predicted_lab.shape}")
            
        if target_lab.dim() != 4 or target_lab.shape[1] != 3:
            raise ValueError(f"Expected 4D tensor with 3 channels, got shape {target_lab.shape}")
            
        # Check shape compatibility
        if predicted_lab.shape != target_lab.shape:
            raise ValueError(f"Shape mismatch: predicted {predicted_lab.shape} vs target {target_lab.shape}")
        
        # Convert LAB to RGB for VGG processing
        predicted_rgb = self._prepare_input(predicted_lab)
        target_rgb = self._prepare_input(target_lab)
        
        # Extract features
        predicted_features = self.feature_extractor(predicted_rgb)
        target_features = self.feature_extractor(target_rgb)
        
        # Calculate weighted feature loss
        total_loss = 0.0
        
        for pred_feat, target_feat, layer_weight in zip(predicted_features, target_features, self.layer_weights):
            # Calculate MSE loss between features
            feature_loss = F.mse_loss(pred_feat, target_feat)
            total_loss += layer_weight * feature_loss
            
        # Apply overall weight
        weighted_loss = self.weight * total_loss
        
        return weighted_loss


class HybridLoss(nn.Module):
    """
    Hybrid loss function combining L1 and perceptual losses for colorization training.
    
    This loss function combines per-pixel L1 loss in CIELAB space with VGG-based
    perceptual loss to achieve both pixel-level accuracy and perceptual quality.
    """
    
    def __init__(self, 
                 l1_weight: float = 1.0,
                 perceptual_weight: float = 0.1,
                 vgg_layers: Tuple[int, ...] = (3, 8, 15, 22),
                 vgg_layer_weights: Optional[Tuple[float, ...]] = None):
        """
        Initialize hybrid loss function.
        
        Args:
            l1_weight: Weight for L1 loss component
            perceptual_weight: Weight for perceptual loss component
            vgg_layers: VGG layer indices for perceptual loss
            vgg_layer_weights: Weights for each VGG layer (if None, uses equal weights)
        """
        super(HybridLoss, self).__init__()
        
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        
        # Initialize L1 loss for ab channels
        self.l1_loss = L1Loss(reduction='mean', weight=l1_weight)
        
        # Initialize perceptual loss for full LAB images
        self.perceptual_loss = PerceptualLoss(
            feature_layers=vgg_layers,
            layer_weights=vgg_layer_weights,
            weight=perceptual_weight
        )
        
    def forward(self, 
                predicted_ab: torch.Tensor, 
                target_ab: torch.Tensor,
                predicted_lab: torch.Tensor,
                target_lab: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate hybrid loss combining L1 and perceptual losses.
        
        Args:
            predicted_ab: Predicted 'a' and 'b' channels, shape (batch_size, 2, height, width)
            target_ab: Target 'a' and 'b' channels, shape (batch_size, 2, height, width)
            predicted_lab: Full predicted LAB image, shape (batch_size, 3, height, width)
            target_lab: Full target LAB image, shape (batch_size, 3, height, width)
            
        Returns:
            Tuple of (total_loss, l1_loss, perceptual_loss)
        """
        # Calculate L1 loss on ab channels
        l1_loss_value = self.l1_loss(predicted_ab, target_ab)
        
        # Calculate perceptual loss on full LAB images
        perceptual_loss_value = self.perceptual_loss(predicted_lab, target_lab)
        
        # Combine losses
        total_loss = l1_loss_value + perceptual_loss_value
        
        return total_loss, l1_loss_value, perceptual_loss_value
    
    def get_loss_weights(self) -> Tuple[float, float]:
        """
        Get current loss component weights.
        
        Returns:
            Tuple of (l1_weight, perceptual_weight)
        """
        return self.l1_weight, self.perceptual_weight
    
    def set_loss_weights(self, l1_weight: float, perceptual_weight: float) -> None:
        """
        Update loss component weights.
        
        Args:
            l1_weight: New weight for L1 loss component
            perceptual_weight: New weight for perceptual loss component
        """
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        
        # Update component weights
        self.l1_loss.weight = l1_weight
        self.perceptual_loss.weight = perceptual_weight
        
    def calculate_loss_ratio(self, 
                           predicted_ab: torch.Tensor, 
                           target_ab: torch.Tensor,
                           predicted_lab: torch.Tensor,
                           target_lab: torch.Tensor) -> Tuple[float, float]:
        """
        Calculate the ratio of L1 to perceptual loss for analysis.
        
        Args:
            predicted_ab: Predicted 'a' and 'b' channels
            target_ab: Target 'a' and 'b' channels  
            predicted_lab: Full predicted LAB image
            target_lab: Full target LAB image
            
        Returns:
            Tuple of (l1_ratio, perceptual_ratio) where ratios sum to 1.0
        """
        with torch.no_grad():
            # Calculate unweighted losses
            l1_unweighted = F.l1_loss(predicted_ab, target_ab)
            
            # Temporarily set perceptual weight to 1.0 for fair comparison
            original_weight = self.perceptual_loss.weight
            self.perceptual_loss.weight = 1.0
            perceptual_unweighted = self.perceptual_loss(predicted_lab, target_lab)
            self.perceptual_loss.weight = original_weight
            
            # Calculate ratios
            total = l1_unweighted + perceptual_unweighted
            if total > 0:
                l1_ratio = (l1_unweighted / total).item()
                perceptual_ratio = (perceptual_unweighted / total).item()
            else:
                l1_ratio = 0.5
                perceptual_ratio = 0.5
                
        return l1_ratio, perceptual_ratio