#!/bin/bash
#SBATCH --job-name syn_predict
#SBATCH --partition=skinniderlabhp,main,skinniderlab
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1    # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G              # memory per node (4G per cpu-core is default)
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=vg8892@princeton.edu
#SBATCH --output=out_predict/job_output_%A_%a.out
#SBATCH --error=out_predict/job_error_%A_%a.err
#SBATCH --array=1-10

source /Genomics/argo/users/vg8892/miniforge3/bin/activate
conda activate massspecgym

aug_count='1'

python predict.py \
    --checkpoint /Genomics/argo/users/vg8892/git_check/ion2fp/runs/sv102_augx${aug_count}/split${SLURM_ARRAY_TASK_ID}/checkpoints/last.ckpt \
    --input /Genomics/skinniderlab/msms-triangulation/spectra/benchmark/releases/spectraverse-1.0.2.mgf \
    --output /Genomics/argo/users/vg8892/git_check/ion2fp/runs/sv102_augx${aug_count}/split${SLURM_ARRAY_TASK_ID}/preds.csv \
    --fold $SLURM_ARRAY_TASK_ID