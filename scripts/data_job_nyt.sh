#!/bin/bash
#SBATCH --mem=140G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=30:0:0    
#SBATCH --gres=gpu:v100l:1
#SBATCH --array=1987-2007

cd $project/moralization_temporal
module purge
module load python/3.10 scipy-stack
source ~/venv2/bin/activate

year=$($SLURM_ARRAY_TASK_ID)

model='bert-base-uncased' 

python SWOW_prediction/data_preprocessing.py --data coha --year $year --model $model --function encoding --length 200 ;
python SWOW_prediction/data_preprocessing.py --data coha --year $year --model $model --function embedding --length 200 ;
python SWOW_prediction/data_preprocessing.py --data coha --year $year --model $model --function graph --length 200 ;
