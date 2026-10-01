#!/bin/bash
#SBATCH --job-name syn_msg_train
#SBATCH --partition=skinniderlabhp,main,skinniderlab
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=32    # cpu-cores per task 
#SBATCH --mem=256G              # memory per node (4G per cpu-core is default)
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=vg8892@princeton.edu
#SBATCH --output=out_msg_train/job_output_%A_%a.out
#SBATCH --error=out_msg_train/job_error_%A_%a.err
#SBATCH --array=7

source /Genomics/argo/users/vg8892/miniforge3/bin/activate
conda activate massspecgym

line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" train_msg.txt)
source=$(echo $line | cut -d ' ' -f     1)
exclude=$(echo $line | cut -d ' ' -f 2)
aug_only=$(echo $line | cut -d ' ' -f 3)

val_fold='10'
test_fold='11'
augmentation_multiplier=13
cand_type="formula"

python train.py --input /Genomics/argo/users/vg8892/MassSpecGym_main/data/main/msgym/MassSpecGym_with_test.mgf \
    --augmentation_file /Genomics/argo/users/vg8892/git_check/ms-pred/results/iceberg_massspecgym/split_11/retrieval_massspecgym_split_11_${cand_type}_combined/preds_normalized.mgf \
    --augmentation_multiplier ${augmentation_multiplier} \
    --augmentation_candidates /Genomics/argo/users/vg8892/MassSpecGym_main/data/main/msgym/MassSpecGym_retrieval_candidates_${cand_type}_with_test.json \
    --output_dir runs/msg_iceberg_augx${augmentation_multiplier}/split${test_fold}_${cand_type}_${source}_${exclude}_${aug_only} --max_epochs 10 --val_fold ${val_fold} --test_fold ${test_fold} \
    --num_workers 31 --augmentation_source ${source} --augmentation_exclude ${exclude} --augmentation_only ${aug_only}