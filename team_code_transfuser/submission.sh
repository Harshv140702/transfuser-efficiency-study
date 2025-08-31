#!/bin/bash --login
#SBATCH -p gpuA             # v100 GPUs, gpuA for A100 GPUs
#SBATCH -G 1                   # 1 GPU
#SBATCH -t 4-0                 # Wallclock limit (1-0 is 1 day, 4-0 is the max permitted)
#SBATCH -n 1                   # One Slurm task
#SBATCH -c 8                   # 8 CPU cores available to the host code.
                                         # Can use up to 12 CPUs with an A100 GPU.

module purge

#Add below command for activating conda virtual environment if required
#e.g. source activate my_condaenv
conda activate project

python train.py --batch_size 10 --epochs 10 --start_epoch 7 --load_file /mnt/iusers01/fse-ugpgt01/compsci01/f15583hs/scratch/log_eff_100/transfuser/model_7.pth  --parallel_training 0