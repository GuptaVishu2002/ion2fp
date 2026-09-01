#!/bin/bash
#SBATCH --job-name syn_retrieval
#SBATCH --partition=skinniderlabhp,main,skinniderlab
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=16    # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=128G              # memory per node (4G per cpu-core is default)
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=vg8892@princeton.edu
#SBATCH --output=out_retrieval/job_output_%A_%a.out
#SBATCH --error=out_retrieval/job_error_%A_%a.err
#SBATCH --array=2

source /Genomics/argo/users/vg8892/miniforge3/bin/activate
conda activate massspecgym

aug_count='1'
cand='s4_formula_50'

python retrieve.py \
        --predictions /Genomics/argo/users/vg8892/git_check/ion2fp/runs/sv102_augx${aug_count}/split${SLURM_ARRAY_TASK_ID}/preds.csv \
        --candidates /Genomics/skinniderlab/msms-triangulation/data/toy_dataset_102/spectraverse101_main/${cand}/spectraverse101_main_${cand}_mces-3-single_${SLURM_ARRAY_TASK_ID}_morgan4096.pkl \
        --scores_out /Genomics/argo/users/vg8892/git_check/ion2fp/runs/sv102_augx${aug_count}/split${SLURM_ARRAY_TASK_ID}/${cand}_scores_${SLURM_ARRAY_TASK_ID}.txt \
        --ranks_out /Genomics/argo/users/vg8892/git_check/ion2fp/runs/sv102_augx${aug_count}/split${SLURM_ARRAY_TASK_ID}/${cand}_ranks_${SLURM_ARRAY_TASK_ID}.txt \
        --workers 15