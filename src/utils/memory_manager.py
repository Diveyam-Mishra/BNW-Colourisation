"""
Memory management and optimization utilities for the image colorization system.

This module provides automatic batch size reduction, GPU/CPU fallback mechanisms,
memory usage monitoring, and cleanup utilities.
"""

import gc
import logging
import psutil
import torch
from typing import Optional, Dict, Any, Callable, Tuple
from functools import wraps
import threading
import time

from .exceptions import InsufficientMemoryError, handle_memory_error
from .logging_utils import get_logger


class MemoryManager:
    """Manages memory usage and provides optimization strategies."""
    
    def __init__(self, initial_batch_size: int = 32, min_batch_size: int = 1):
        self.initial_batch_size = initial_batch_size
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = get_logger()
        
        # Memory monitoring
        self.memory_threshold_mb = 1024  # 1GB threshold
        self.monitoring_enabled = False
        self.monitoring_thread = None
        
    def get_available_memory(self, device: str = None) -> float:
        """Get available memory in MB for the specified device."""
        if device is None:
            device = self.device
            
        if device == "cuda" and torch.cuda.is_available():
            # GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_allocated = torch.cuda.memory_allocated(0)
            available_gpu = (gpu_memory - gpu_allocated) / (1024 ** 2)  # Convert to MB
            return available_gpu
        else:
            # CPU memory
            memory_info = psutil.virtual_memory()
            available_cpu = memory_info.available / (1024 ** 2)  # Convert to MB
            return available_cpu
    
    def get_memory_usage(self, device: str = None) -> Dict[str, float]:
        """Get detailed memory usage information."""
        if device is None:
            device = self.device
            
        usage = {}
        
        if device == "cuda" and torch.cuda.is_available():
            # GPU memory usage
            gpu_allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
            gpu_cached = torch.cuda.memory_reserved(0) / (1024 ** 2)
            gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            
            usage.update({
                'gpu_allocated_mb': gpu_allocated,
                'gpu_cached_mb': gpu_cached,
                'gpu_total_mb': gpu_total,
                'gpu_available_mb': gpu_total - gpu_allocated
            })
        
        # CPU memory usage
        memory_info = psutil.virtual_memory()
        usage.update({
            'cpu_used_mb': memory_info.used / (1024 ** 2),
            'cpu_available_mb': memory_info.available / (1024 ** 2),
            'cpu_total_mb': memory_info.total / (1024 ** 2),
            'cpu_percent': memory_info.percent
        })
        
        return usage
    
    def reduce_batch_size(self) -> int:
        """Reduce batch size by half, respecting minimum."""
        old_batch_size = self.current_batch_size
        self.current_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
        
        self.logger.log_memory_usage(
            f"batch_size_reduction", 
            self.get_available_memory(),
            self.device
        )
        
        self.logger.logger.info(
            f"Reduced batch size from {old_batch_size} to {self.current_batch_size}"
        )
        
        return self.current_batch_size
    
    def fallback_to_cpu(self):
        """Switch processing to CPU."""
        if self.device == "cuda":
            old_device = self.device
            self.device = "cpu"
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.logger.info(f"Switched from {old_device} to {self.device}")
            self.logger.log_memory_usage(
                "device_fallback", 
                self.get_available_memory(),
                self.device
            )
    
    def cleanup_memory(self):
        """Perform memory cleanup operations."""
        # Python garbage collection
        collected = gc.collect()
        
        # PyTorch GPU cache cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.logger.debug(f"Memory cleanup: collected {collected} objects")
        
        # Log memory usage after cleanup
        usage = self.get_memory_usage()
        self.logger.log_memory_usage("after_cleanup", usage.get('cpu_available_mb', 0))
    
    def check_memory_threshold(self, required_memory_mb: float) -> bool:
        """Check if there's enough memory for the operation."""
        available = self.get_available_memory()
        
        if available < required_memory_mb:
            self.logger.logger.warning(
                f"Insufficient memory: required {required_memory_mb}MB, "
                f"available {available}MB on {self.device}"
            )
            return False
        
        return True
    
    def estimate_memory_requirement(self, batch_size: int, image_size: Tuple[int, int], 
                                  channels: int = 3, dtype_size: int = 4) -> float:
        """Estimate memory requirement for a batch of images in MB."""
        # Basic estimation: batch_size * height * width * channels * dtype_size * overhead_factor
        height, width = image_size
        overhead_factor = 3.0  # Account for intermediate tensors and gradients
        
        memory_mb = (batch_size * height * width * channels * dtype_size * overhead_factor) / (1024 ** 2)
        return memory_mb
    
    def start_memory_monitoring(self, interval_seconds: float = 5.0):
        """Start background memory monitoring."""
        if self.monitoring_enabled:
            return
        
        self.monitoring_enabled = True
        
        def monitor():
            while self.monitoring_enabled:
                usage = self.get_memory_usage()
                
                # Log CPU memory usage
                self.logger.log_memory_usage("monitoring", usage.get('cpu_available_mb', 0), "cpu")
                
                # Log GPU memory usage if available
                if self.device == "cuda" and torch.cuda.is_available():
                    self.logger.log_memory_usage("monitoring", usage.get('gpu_available_mb', 0), "gpu")
                
                # Check for memory pressure
                if self.device == "cuda" and torch.cuda.is_available():
                    gpu_usage_percent = (usage.get('gpu_allocated_mb', 0) / 
                                       usage.get('gpu_total_mb', 1)) * 100
                    if gpu_usage_percent > 90:
                        self.logger.logger.warning(f"High GPU memory usage: {gpu_usage_percent:.1f}%")
                
                cpu_usage_percent = usage.get('cpu_percent', 0)
                if cpu_usage_percent > 90:
                    self.logger.logger.warning(f"High CPU memory usage: {cpu_usage_percent:.1f}%")
                
                time.sleep(interval_seconds)
        
        self.monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.logger.info("Started memory monitoring")
    
    def stop_memory_monitoring(self):
        """Stop background memory monitoring."""
        self.monitoring_enabled = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        
        self.logger.logger.info("Stopped memory monitoring")
    
    def reset_batch_size(self):
        """Reset batch size to initial value."""
        self.current_batch_size = self.initial_batch_size
        self.logger.logger.info(f"Reset batch size to {self.current_batch_size}")


# Global memory manager instance
_global_memory_manager = None


def get_memory_manager(initial_batch_size: int = 32, min_batch_size: int = 1) -> MemoryManager:
    """Get or create a global memory manager instance."""
    global _global_memory_manager
    
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager(initial_batch_size, min_batch_size)
    
    return _global_memory_manager


def memory_optimized(estimate_memory: Optional[Callable] = None, 
                    auto_reduce_batch: bool = True,
                    auto_fallback_cpu: bool = True):
    """
    Decorator for automatic memory optimization.
    
    Args:
        estimate_memory: Function to estimate memory requirements
        auto_reduce_batch: Whether to automatically reduce batch size on memory errors
        auto_fallback_cpu: Whether to automatically fallback to CPU on memory errors
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            memory_manager = get_memory_manager()
            
            # Estimate memory requirements if function provided
            if estimate_memory:
                try:
                    required_memory = estimate_memory(*args, **kwargs)
                    if not memory_manager.check_memory_threshold(required_memory):
                        raise InsufficientMemoryError(
                            required_memory, 
                            memory_manager.get_available_memory(),
                            memory_manager.device
                        )
                except InsufficientMemoryError:
                    # Re-raise memory errors
                    raise
                except Exception as e:
                    # If estimation fails for other reasons, continue without pre-check
                    memory_manager.logger.logger.debug(f"Memory estimation failed: {e}")
            
            try:
                return func(*args, **kwargs)
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    # Convert to our custom exception
                    raise InsufficientMemoryError(
                        None, 
                        memory_manager.get_available_memory(),
                        memory_manager.device
                    )
                else:
                    raise
        
        # Apply memory error handling if requested
        if auto_reduce_batch or auto_fallback_cpu:
            wrapper = handle_memory_error(wrapper)
        
        return wrapper
    
    return decorator


class MemoryContext:
    """Context manager for memory-aware operations."""
    
    def __init__(self, operation_name: str, cleanup_on_exit: bool = True):
        self.operation_name = operation_name
        self.cleanup_on_exit = cleanup_on_exit
        self.memory_manager = get_memory_manager()
        self.initial_memory = None
    
    def __enter__(self):
        self.initial_memory = self.memory_manager.get_memory_usage()
        self.memory_manager.logger.logger.info(
            f"Starting memory-aware operation: {self.operation_name}"
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        final_memory = self.memory_manager.get_memory_usage()
        
        # Calculate memory delta
        if self.initial_memory:
            cpu_delta = (final_memory.get('cpu_used_mb', 0) - 
                        self.initial_memory.get('cpu_used_mb', 0))
            
            self.memory_manager.logger.logger.info(
                f"Completed operation: {self.operation_name}, "
                f"CPU memory delta: {cpu_delta:.1f}MB"
            )
            
            if 'gpu_allocated_mb' in final_memory and 'gpu_allocated_mb' in self.initial_memory:
                gpu_delta = (final_memory['gpu_allocated_mb'] - 
                           self.initial_memory['gpu_allocated_mb'])
                self.memory_manager.logger.logger.info(
                    f"GPU memory delta: {gpu_delta:.1f}MB"
                )
        
        if self.cleanup_on_exit:
            self.memory_manager.cleanup_memory()
        
        return False  # Don't suppress exceptions


def monitor_memory_usage(func):
    """Decorator to monitor memory usage of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with MemoryContext(f"{func.__module__}.{func.__name__}"):
            return func(*args, **kwargs)
    
    return wrapper