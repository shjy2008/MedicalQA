#!/bin/bash
# #SBATCH --job-name=medical_qa # define the job name
# #SBATCH --mem=4GB # request an allocation with 4GB of ram
# #SBATCH --time=00:01:00 # job time limit of 1 minute (hh:mm:ss)
# #SBATCH --partition=aoraki # 'aoraki' or 'aoraki_gpu' (for gpu access)

#SBATCH --job-name=llm
#SBATCH --account=sjy

# #SBATCH --partition=aoraki
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=20
# #SBATCH --time=00:00:30

# #SBATCH --out=log.txt

# #SBATCH --partition=aoraki_gpu
#SBATCH --partition=aoraki_gpu_H100
# #SBATCH --partition=aoraki_gpu_A100_80GB
#SBATCH --gpus-per-node=1
#SBATCH --mem=60GB
#SBATCH --time=00:10:00

# echo "hello world"

# usual bash commands go below here:
echo "my script will now start"
# nvidia-smi
# sleep 10 # pretend to do something

conda init bash
source ~/.bashrc
conda --version

echo "conda init"

conda activate LLM

echo "conda acticate LLM"

python ./train_phi_3_mini.py

echo "my script has finished."