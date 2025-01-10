#!/bin/bash
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=6:0:0    
#SBATCH --gpus-per-node=1

cd $project/moralization_temporal
module purge
module load python/3.10 scipy-stack StdEnv/2023 gentoo/2023
source ~/venv2/bin/activate

for p in "previous_link" "polarity"
do 
    for i in {0..20}
    do
        for j in {0..4}
        do
            python SWOW_prediction/train_features.py --config_path SWOW_prediction/config_features.yml --eval --section $j --eval_section $i --reduce both --data_name coha --property $p
        
            python SWOW_prediction/train_features.py --config_path SWOW_prediction/config_features.yml --eval --section $j --eval_section $i --reduce both --data_name nyt --property $p 

            
        done
    done
done 
