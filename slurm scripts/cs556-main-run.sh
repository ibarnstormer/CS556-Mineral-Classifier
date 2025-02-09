#!/bin/bash
#SBATCH -N 1
#SBATCH -n 25
#SBATCH --mem=12g
#SBATCH -J "CS 556 Mineral Classifier model"
#SBATCH -p long
#SBATCH -t 2-23:00:00
#SBATCH --gres=gpu:1
#SBATCH -C H100|L40S|A100|V100

module load python/3.10.12/f5uihwq
module load cuda11.2/blas/11.2.2
module load cuda11.2/fft/11.2.2
module load cuda11.2/toolkit/11.2.2
module load cudnn8.1-cuda11.2/8.1.1.33

export set XLA_FLAGS=--xla_gpu_cuda_data_dir=/cm/shared/apps/cuda11.2/toolkit/11.2.2

source # TODO: create directory on the Turing Cluster and add files there using SFTP

python3 model_train.py
python3 model_train.py