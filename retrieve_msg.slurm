#!/bin/bash
#SBATCH --job-name syn_retrieval
#SBATCH --partition=skinniderlabhp,main,skinniderlab
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=16    # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G              # memory per node (4G per cpu-core is default)
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=vg8892@princeton.edu
#SBATCH --output=out_msg_ret/job_output_%A_%a.out
#SBATCH --error=out_msg_ret/job_error_%A_%a.err
#SBATCH --array=1-8

source /Genomics/argo/users/vg8892/miniforge3/bin/activate
conda activate massspecgym

line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" train_msg.txt)
source=$(echo $line | cut -d ' ' -f 1)
exclude=$(echo $line | cut -d ' ' -f 2)
aug_only=$(echo $line | cut -d ' ' -f 3)

aug_count='13'
aug_model='iceberg'
cand_type="formula"
ret_cand='mass'
test_fold='11'

python retrieve.py \
        --predictions /Genomics/argo/users/vg8892/git_check/ion2fp/runs/msg_${aug_model}_augx${aug_count}/split${test_fold}_${cand_type}_${source}_${exclude}_${aug_only}/preds.csv \
        --candidates /Genomics/skinniderlab/msms-triangulation/data/toy_dataset_102/massspecgym/${ret_cand}/massspecgym_${ret_cand}_${test_fold}_morgan4096.pkl \
        --scores_out /Genomics/argo/users/vg8892/git_check/ion2fp/runs/msg_${aug_model}_augx${aug_count}/split${test_fold}_${cand_type}_${source}_${exclude}_${aug_only}/${ret_cand}_scores_${test_fold}.txt \
        --ranks_out /Genomics/argo/users/vg8892/git_check/ion2fp/runs/msg_${aug_model}_augx${aug_count}/split${test_fold}_${cand_type}_${source}_${exclude}_${aug_only}/${ret_cand}_ranks_${test_fold}.txt \
        --workers 15