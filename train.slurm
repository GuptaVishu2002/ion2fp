#!/bin/bash
#SBATCH --job-name syn_train
#SBATCH --partition=main,skinniderlab
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=32    # cpu-cores per task 
#SBATCH --mem=128G              # memory per node (4G per cpu-core is default)
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=vg8892@princeton.edu
#SBATCH --output=out_train/job_output_%A_%a.out
#SBATCH --error=out_train/job_error_%A_%a.err
#SBATCH --array=1-10


source /Genomics/argo/users/vg8892/miniforge3/bin/activate
conda activate massspecgym

line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" split.txt)
val_fold=$(echo $line | cut -d ' ' -f 1)
test_fold=$(echo $line | cut -d ' ' -f 2)
augmentation_multiplier=$(echo $line | cut -d ' ' -f 3)

python train.py --input /Genomics/skinniderlab/msms-triangulation/spectra/benchmark/releases/spectraverse-1.0.2.mgf \
    --augmentation_multiplier ${augmentation_multiplier} \
    --output_dir runs/sv102_augx${augmentation_multiplier}/split${test_fold} --val_fold ${val_fold} --test_fold ${test_fold} \
    --num_workers 31 --augmentation_resample_every_epoch n