#!/bin/bash
# #SBATCH --job-name=medical_qa # define the job name
# #SBATCH --mem=4GB # request an allocation with 4GB of ram
# #SBATCH --time=00:01:00 # job time limit of 1 minute (hh:mm:ss)
# #SBATCH --partition=aoraki # 'aoraki' or 'aoraki_gpu' (for gpu access)

#SBATCH --job-name=medical_qa_test
#SBATCH --account=sheju347

# #SBATCH --partition=aoraki
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=20
# #SBATCH --time=00:00:30

#SBATCH --out=log.log

#SBATCH --partition=aoraki_gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=4GB
#SBATCH --time=00:10:00

# echo "hello world"

# usual bash commands go below here:
echo "my script will now start"
# nvidia-smi
# sleep 10 # pretend to do something

conda init
conda activate LLM

python ./training_test.py

echo "my script has finished."