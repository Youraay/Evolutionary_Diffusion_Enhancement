#!/bin/bash

#SBATCH --job-name=latents_genetic
#SBATCH --output=logs/simulation/a_cat_to_a_dog_1549660.out
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --account=dldevel
#SBATCH --mem=50G
#SBATCH --time=10:00:00

# module load nvidia/cuda/12.3.0

source /scratch/dldevel/sinziri/Evolutionary_Diffusion_Enhancement/.env
export HF_HOME=/scratch/dldevel/sinziri/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export DIFFUSERS_CACHE=$HF_HOME
echo "load conda environment"
eval "$(/scratch/dldevel/sinziri/miniconda3/bin/conda shell.bash hook)"
conda activate genetic
which python3
echo "loaded conda environment"
echo "start transfer from cat to dogs"
date

python3 simulations/seed_transfare.py --experiment_id 1549660
 
echo job finished
date