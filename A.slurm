#!/bin/bash

#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --constraint=v100  # Ensure the node has V100
#SBATCH --output=/lustre/fs1/home/cap5415.student28/VIT_model-%J.out
#SBATCH --job-name=oursv1_partial_01

module load anaconda/anaconda-2023.09

nvidia-smi topo -m

source activate pytorch2.2.0+py3.11+cuda12.1 

# Install timm library
pip install timm --no-cache-dir

python ViT_ASL_Model.py
