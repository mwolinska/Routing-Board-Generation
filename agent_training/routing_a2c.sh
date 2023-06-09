#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=<your-username> # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/${USER}/myvenv/bin/:$PATH
source activate
source /vol/cuda/11.4.120-cudnn8.2.4/setup.sh

export NEPTUNE_API_TOKEN=<your-token>

python3 training_script.py env.ic_board.generation_type=seq_parallel_rw env.ic_board.board_name=none

TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime
