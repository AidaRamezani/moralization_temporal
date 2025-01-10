#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:0:0    
#SBATCH --mail-type=ALL
#SBATCH --gpus-per-node=1

cd $project/moralization_temporal
module purge
module load python/3.10 scipy-stack
source ~/venv2/bin/activate


for p in "previous_link" "polarity"
do 
    for section in 0 1 2 3 4
    do
        python SWOW_prediction/train_features.py --data_name coha --section $section --property $p  --config_path SWOW_prediction/config_features.yml --baseline 
        python SWOW_prediction/train_features.py --data_name nyt --section $section --property $p  --config_path SWOW_prediction/config_features.yml --baseline 
        
        python SWOW_prediction/train_features.py --data_name coha --section $section --property $p --config_path SWOW_prediction/config_features.yml --reduce forward 
        python SWOW_prediction/train_features.py --data_name nyt --section $section --property $p --config_path SWOW_prediction/config_features.yml --reduce forward 

        python SWOW_prediction/train_features.py --data_name coha --section $section --property $p --config_path SWOW_prediction/config_features.yml --reduce both
        python SWOW_prediction/train_features.py --data_name nyt --section $section --property $p --config_path SWOW_prediction/config_features.yml --reduce both
    done

done
