#!/bin/bash
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=3:0:0    
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:v100l:1

# Define project directory
#project=/path/to/your/project

cd $project/moralization_temporal
module purge
module load python/3.10 scipy-stack
source ~/venv2/bin/activate

# Loops
for p in "previous_link" "polarity"
do 
    for reduce in "forward" "both"
    do
        for section in 0 1 2 3 4
        do
            python SWOW_prediction/train_features.py --config_path SWOW_prediction/config_features_eval.yml --section $section --eval_section test --reduce $reduce --data_name coha --property $p
            python SWOW_prediction/train_features.py --config_path SWOW_prediction/config_features_eval.yml --section $section --eval_section dev --reduce $reduce --data_name coha --property $p
            python SWOW_prediction/train_features.py --config_path SWOW_prediction/config_features_eval.yml --section $section --eval_section test --reduce $reduce --data_name nyt --property $p
            python SWOW_prediction/train_features.py --config_path SWOW_prediction/config_features_eval.yml --section $section --eval_section dev --reduce $reduce --data_name nyt --property $p
        done
    done

    for section in 0 1 2 3 4
    do
        python SWOW_prediction/train_features.py --config_path SWOW_prediction/config_features_eval.yml --section $section --eval_section test --baseline --data_name coha --property $p
        python SWOW_prediction/train_features.py --config_path SWOW_prediction/config_features_eval.yml --section $section --eval_section dev --baseline --data_name coha --property $p
        python SWOW_prediction/train_features.py --config_path SWOW_prediction/config_features_eval.yml --section $section --eval_section test --baseline --data_name nyt --property $p
        python SWOW_prediction/train_features.py --config_path SWOW_prediction/config_features_eval.yml --section $section --eval_section dev --baseline --data_name nyt --property $p
    done
done
