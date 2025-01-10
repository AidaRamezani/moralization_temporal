#!/bin/bash
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:0:0    
#SBATCH --array=97-114
cd $project/moralization_temporal
module purge
module load python/3.10 scipy-stack
source ~/venv2/bin/activate

echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
python SWOW_prediction/congressional_speech_processing.py --congress_id $SLURM_ARRAY_TASK_ID

