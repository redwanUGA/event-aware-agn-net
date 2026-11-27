#!/bin/bash
#SBATCH --job-name=hfl_x
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/%x_%j.out
module load p7zip/17.05-GCCcore-13.3.0
cd $RUN_ROOT/HF_Lung_V1
7za x -y train.7z.001 > /dev/null && echo train ok
7za x -y test.7z.001 > /dev/null && echo test ok
find . -name "*.wav" | wc -l
ls | head
