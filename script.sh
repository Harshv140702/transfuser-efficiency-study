#!/bin/bash --login
#SBATCH -p gpuV            # v100 GPUs, gpuA for A100 GPUs
#SBATCH -G 1                   # 1 GPU
#SBATCH -t 4-0                 # Wallclock limit (1-0 is 1 day, 4-0 is the max permitted)
#SBATCH -n 1                   # One Slurm task
#SBATCH -c 8                   # 8 CPU cores available to the host code.
                                         # Can use up to 12 CPUs with an A100 GPU.

module purge

#Add below command for activating conda virtual environment if required
#e.g. source activate my_condaenv
conda activate project

./carla/CarlaUE4.sh --world-port=2000 -opengl &
sleep 120
./leaderboard/scripts/local_evaluation.sh

