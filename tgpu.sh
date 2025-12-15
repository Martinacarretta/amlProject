#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -D /tmp # working directory
#SBATCH -t 0-20:00 # Runtime in D-HH:MM
#SBATCH -p tfg # Partition to submit to
#SBATCH --mem 2048 # 2GB solicitados.
#SBATCH -o /export/fhome/amlai01/PROJECT/project/train.out
#SBATCH -e /export/fhome/amlai01/PROJECT/project/train.err
#SBATCH --gres gpu:2 # Para pedir 3090 MAX 8

/ghome/share/example/deviceQuery
nvidia-smi

/ghome/share/example/deviceQuery
nvidia-smi

source /export/fhome/amlai01/.venv/bin/activate

cd /export/fhome/amlai01/PROJECT/project

python3 train.py