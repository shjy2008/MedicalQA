#!/bin/bash
# Job name:
#SBATCH --job-name=test
#
# Partition:
#SBATCH --partition=aoraki
#
# Request one node:
#SBATCH --nodes=1
#
# Specify one task:
#SBATCH --ntasks-per-node=1
#
# Number of processors for single task needed for use case (example):
#SBATCH --cpus-per-task=4
#
# Wall clock limit:
#SBATCH --time=00:00:30
#
echo "hello world"