# MDSSegNet

Official PyTorch implementation of **MDSSegNet**, a lightweight multi-scale deep-supervision framework for concrete-crack segmentation and geometric quantification.

> The repository is currently private while the associated manuscript is under revision.

## Repository structure

```text
MDSSegNet/
├── checkpoints/                     # Pretrained MDSSegNet weights
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

Experiment outputs and raw dataset files are not stored directly in this repository.

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

## CTC dataset

The CTC dataset is the custom concrete-crack segmentation dataset prepared for the MDSSegNet study. It contains **2,600 image-mask pairs** and is supplied with fixed 80/10/10 training, validation, and test splits.

- **Download:** [CTC_dataset.zip (Google Drive)](https://drive.google.com/file/d/16t8wOdmIq4y_H5ol-ApG__UVxLqNLof8/view?usp=sharing)
- **Archive size:** 71.2 MB
- **SHA-256:** `9933B7598833EC82DC6374F5CD3468DC4EF16D336E6D33FC5B4DC66A64B069BA`
- **Source:** Custom dataset prepared for the MDSSegNet study; see the [MDSSegNet manuscript record](https://ssrn.com/abstract=6549934).
- **License and permitted use:** No separate open-source license is included with the dataset. It is shared for academic research use. For other uses or redistribution, please contact the authors.

After downloading, extract the archive so that the dataset has the following structure:

```text
CTC/
├── train/
│   ├── images/                      # 2,080 JPG images (+ one JSON metadata file)
│   └── masks/                       # 2,080 PNG segmentation masks
├── val/
│   ├── images/                      # 260 JPG images
│   └── masks/                       # 260 PNG segmentation masks
└── test/
    ├── images/                      # 260 JPG images
    └── masks/                       # 260 PNG segmentation masks
```

Each image and its segmentation mask use the same filename stem. Point the corresponding training or evaluation argument to the required split directory, for example `path/to/CTC/test`.

## Pretrained weights

Pretrained MDSSegNet weights are available in the repository's `checkpoints/` directory. Select the checkpoint matching the dataset being evaluated and pass its path through the checkpoint argument used by the relevant script.

## Citation

Citation information will be added after publication.
