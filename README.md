# Image Colorization System

A deep learning-based system for automatically colorizing grayscale images using a U-Net architecture with skip connections.

## Project Structure

```
├── src/
│   ├── models/          # Neural network models
│   ├── data/            # Data processing utilities
│   ├── training/        # Training pipeline
│   └── utils/           # Helper utilities
├── configs/             # Configuration files
├── tests/               # Test suite
├── checkpoints/         # Model checkpoints (created during training)
├── data/                # Dataset directory (create manually)
└── outputs/             # Generated colorized images (created during inference)
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install the package in development mode:
```bash
pip install -e .
```

## Configuration

The system uses YAML configuration files located in the `configs/` directory:

- `model_config.yaml`: Model architecture parameters
- `training_config.yaml`: Training hyperparameters and settings
- `inference_config.yaml`: Inference and output settings

## Usage

Training and inference scripts will be available after implementing the respective components.