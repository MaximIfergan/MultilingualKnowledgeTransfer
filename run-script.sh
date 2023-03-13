#!/bin/bash
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 2      # cores requested
#SBATCH --mem=100000  # memory in Mb
#SBATCH -o JobsOutput/Job_STDOUT_%j  # send stdout to outfile
#SBATCH -e JobsOutput/Job_STDERR_%j  # send stderr to errfile
#SBATCH -t 0-15:00:00  # time requested in hour:minute:second
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=maxim.ifergan@mail.huji.ac.il
#SBATCH --gres=gpu:1,vmem:6g

source /cs/labs/oabend/maximifergan/MultilingualKnowledgeTransfer/venv/bin/activate.csh
/cs/labs/oabend/maximifergan/MultilingualKnowledgeTransfer/venv/bin/python /cs/labs/oabend/maximifergan/MultilingualKnowledgeTransfer/main.py
