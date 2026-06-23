#!/bin/bash
#SBATCH -p kisski-h100
#SBATCH --job-name=schain_filter
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH -t 08:00:00
#SBATCH --output=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/SChain/runs/schain_filter/slurm.log
#SBATCH --exclude=ggpu102,ggpu103,ggpu104,ggpu105,ggpu106,ggpu107,ggpu108,ggpu109,ggpu110,ggpu111,ggpu112,ggpu113,ggpu114,ggpu115,ggpu116,ggpu117,ggpu118,ggpu119,ggpu120,ggpu121,ggpu122,ggpu123,ggpu124,ggpu125,ggpu126,ggpu127,ggpu128,ggpu129,ggpu130,ggpu131,ggpu132,ggpu133,ggpu134,ggpu135,ggpu136

mkdir -p /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/SChain/runs/schain_filter

SCRIPT_DIR=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/SChain/Hoang-Development/MedVLThinker

# GPUS is set to the 2 GPUs SLURM allocated (indices 0,1 within the job)
export GPUS=0,1

# To swap model: set MODEL= here
# export MODEL=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/SChain/model_weights/Qwen2.5-VL-7B-Instruct

bash "${SCRIPT_DIR}/run_schain_filter.sh"
