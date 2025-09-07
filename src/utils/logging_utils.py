"""
Logging utilities for the image colorization system.

This module provides centralized logging configuration and utilities
for error tracking, performance monitoring, and debugging.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json


class ColorizationLogger:
    """Centralized logger for the colorization system."""
    
    def __init__(self, name: str = "colorization", log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Set up file and console handlers with appropriate formatters."""
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler for general logs
        log_file = self.log_dir / f"{self.name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Error-specific handler
        error_file = self.log_dir / f"{self.name}_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file, maxBytes=5*1024*1024, backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Log an error with additional context information."""
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {}
        }
        
        if hasattr(error, 'error_code'):
            error_info['error_code'] = error.error_code
        
        self.logger.error(f"Error occurred: {json.dumps(error_info, indent=2)}")
    
    def log_performance(self, operation: str, duration: float, 
                       additional_metrics: Optional[Dict[str, Any]] = None):
        """Log performance metrics for operations."""
        metrics = {
            'operation': operation,
            'duration_seconds': duration,
            'timestamp': datetime.now().isoformat()
        }
        
        if additional_metrics:
            metrics.update(additional_metrics)
        
        self.logger.info(f"Performance: {json.dumps(metrics)}")
    
    def log_memory_usage(self, stage: str, memory_mb: float, device: str = "cpu"):
        """Log memory usage at different stages."""
        memory_info = {
            'stage': stage,
            'memory_mb': memory_mb,
            'device': device,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Memory usage: {json.dumps(memory_info)}")
    
    def log_training_progress(self, epoch: int, loss: float, 
                            validation_loss: Optional[float] = None,
                            additional_metrics: Optional[Dict[str, Any]] = None):
        """Log training progress information."""
        progress_info = {
            'epoch': epoch,
            'training_loss': loss,
            'timestamp': datetime.now().isoformat()
        }
        
        if validation_loss is not None:
            progress_info['validation_loss'] = validation_loss
        
        if additional_metrics:
            progress_info.update(additional_metrics)
        
        self.logger.info(f"Training progress: {json.dumps(progress_info)}")


# Global logger instance
_global_logger = None


def get_logger(name: str = "colorization", log_dir: str = "logs") -> ColorizationLogger:
    """Get or create a global logger instance."""
    global _global_logger
    
    if _global_logger is None:
        _global_logger = ColorizationLogger(name, log_dir)
    
    return _global_logger


def setup_logging(log_level: str = "INFO", log_dir: str = "logs"):
    """Set up logging configuration for the entire application."""
    
    # Create log directory
    Path(log_dir).mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(log_dir) / "application.log")
        ]
    )
    
    # Suppress verbose logging from external libraries
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)


class LoggingContext:
    """Context manager for logging operations with automatic error handling."""
    
    def __init__(self, operation_name: str, logger: Optional[ColorizationLogger] = None):
        self.operation_name = operation_name
        self.logger = logger or get_logger()
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.logger.info(f"Starting operation: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.logger.info(f"Completed operation: {self.operation_name} in {duration:.2f}s")
            self.logger.log_performance(self.operation_name, duration)
        else:
            self.logger.logger.error(f"Failed operation: {self.operation_name} after {duration:.2f}s")
            self.logger.log_error(exc_val, {'operation': self.operation_name, 'duration': duration})
        
        return False  # Don't suppress exceptions


def log_function_call(func):
    """Decorator to automatically log function calls and their results."""
    def wrapper(*args, **kwargs):
        logger = get_logger()
        func_name = f"{func.__module__}.{func.__name__}"
        
        with LoggingContext(func_name, logger):
            return func(*args, **kwargs)
    
    return wrapper