#!/bin/bash
#SBATCH --job-name syn_predict
#SBATCH --partition=skinniderlabhp,main,skinniderlab
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1    # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G              # memory per node (4G per cpu-core is default)
#SBATCH --time=1:00:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=vg8892@princeton.edu
#SBATCH --output=out_msg_predict/job_output_%A_%a.out
#SBATCH --error=out_msg_predict/job_error_%A_%a.err
#SBATCH --array=1-8

source /Genomics/argo/users/vg8892/miniforge3/bin/activate
conda activate massspecgym

line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" train_msg.txt)
source=$(echo $line | cut -d ' ' -f 1)
exclude=$(echo $line | cut -d ' ' -f 2)
aug_only=$(echo $line | cut -d ' ' -f 3)

augmentation_multiplier='13'
aug_model='iceberg'
test_fold='11'
cand_type="formula"

python predict.py \
    --checkpoint runs/msg_iceberg_augx${augmentation_multiplier}/split${test_fold}_augce_${cand_type}_${source}_${exclude}_${aug_only}/checkpoints/last.ckpt \
    --input /Genomics/argo/users/vg8892/MassSpecGym_main/data/main/msgym/MassSpecGym_with_test.mgf \
    --output runs/msg_iceberg_augx${augmentation_multiplier}/split${test_fold}_augce_${cand_type}_${source}_${exclude}_${aug_only}/preds.csv \
    --fold $test_fold