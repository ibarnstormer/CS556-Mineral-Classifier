This repository contains all the necessary components and scripts to train the Mineral CNN model to be used in the Mineral Classification Mobile App found here: 
https://github.com/ibarnstormer/CS556-Mineral-Classifier-Mobile-App

## Setup Instructions:
1. Clone the current repository as well as the Mobile App repository (https://github.com/ibarnstormer/CS556-Mineral-Classifier-Mobile-App) to a local folder
2. Setup a python virtual environment for the local cloned project
3. If you are planning to train the model locally, download and install the appropriate version of CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit if you have an NVIDIA GPU as well as the corresponding PyTorch versions here: https://pytorch.org/get-started/previous-versions/. If you are planning to train this model using a HPC (high performance compute) instance that supports SLURM jobs, assert that the HPC instance has CUDA installed.
4. Run pip install -r requirements.txt to get the rest of the required libraries for the project
5. Download and extract the MineralImage5K dataset to a separate folder and set the dataset path parameter in model_train.py to point to the dataset folder https://www.kaggle.com/datasets/sergeynesteruk/minerals
6. Run the model_train.py script or edit the SLURM job (if using HPC cluster) script to run the model_train.py script to begin training the model. The model will save the weights to the output folder specified in the output directory argument (-o).
7. Copy the saved weights file and place into the app -> src -> main-> res -> raw directory and change the file extension from .pt to .pte (if needed).

### Additional script parameters for model_train.py:

# Argparse arguments:
**-m:** Model specifier / name of file to which model weights would be saved to
**-p:** Dataset path for MineralImage5K
**-e:** Number of epochs
**-b:** Batch size
**-o:** Output directory relative to the project folder to which to save the model weights to
**-md:** media directory to which to save the model visualizations to
**-pm:** Flag for using pretrained model weights

### Additional internal arguments (found after Argparse arguments):

**do_train:** Flag for performing model training
**do_prune:** Flag for performing magnitude-based fine-grained pruning
**do_test:** Flag for testing model on the test dataset portion of MineralImage5K
**do_viz:** Flag for performing visualization of the model weights
**prune_loaded:** Flag for skipping training/testing and go to pruning / fine-tuning model

## Performance metrics for 10 mineral classes:

| Mineral Class | Precision | Recall | F1-Score |
| ----------- | ----------- |----------- | ----------- |
| Agate | 0.79 | 0.71 | 0.75 |
| Amethyst | 0.82 | 0.90 | 0.86 |
| Beryl | 0.67 | 0.70 | 0.68 |
| Copper | 0.71 | 0.81 | 0.76 |
| Diopside | 0.72 | 0.65 | 0.68 |
| Gold | 0.66 | 0.80 | 0.72 |
| Quartz | 0.75 | 0.83 | 0.79 |
| Silver | 0.78 | 0.70 | 0.74 |
| Spinel | 0.86 | 0.65 | 0.74 |
| Topaz | 0.82 | 0.80 | 0.81 |

## Mineral CNN architecture
![Mineral CNN architecture](https://github.com/ibarnstormer/CS556-Mineral-Classifier/blob/main/media/mineralcnn_dsc_4_21_2025_viz.png)

