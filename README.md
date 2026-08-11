# MDSSegNet

Official PyTorch implementation of **MDSSegNet**, a lightweight multi-scale deep-supervision framework for concrete-crack segmentation and geometric quantification.

> The repository is currently private while the associated manuscript is under revision.

## Repository structure

```text
MDSSegNet/
├── dataset/                         # Dataset loading and augmentation code
├── model/                           # Network architecture and losses
├── util/                            # Metrics and utility functions
├── train.py                         # Training entry point
├── test.py                          # Evaluation entry point
├── Prediction.py                    # Single-image prediction
├── Batch_Prediction.py              # Batch prediction
├── Crack_length_and_width_extraction.py
├── Labelme_conversion_mask.py
└── error_map.py
```

Datasets, experiment outputs, checkpoints, and pretrained weights are not included in this repository.

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

For GPU acceleration, install the PyTorch build matching your CUDA environment by following the official PyTorch installation instructions.

## Usage

The shell scripts contain the command-line arguments used for training, evaluation, prediction, and crack geometry extraction:

```bash
bash train.sh
bash test.sh
bash Prediction.sh
bash Batch_Prediction.sh
bash Crack_length_and_width_extraction.sh
```

Before running them, update local dataset, checkpoint, input, and output paths as needed.

## Data and model weights

Dataset files and trained model weights are intentionally excluded. Place local data and checkpoints outside version control and provide their paths through the corresponding command-line arguments.

## Citation

Citation information will be added after publication.

