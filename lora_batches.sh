#!/bin/bash
#SBATCH --job-name=debias
#SBATCH --partition=
#SBATCH --ntasks=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH -o result/outputs/test_output_%j.txt
#SBATCH -e result/errors/test_error_%j.txt
#SBATCH --time=0-23:59:00

echo "Starting job on $(hostname) at $(date)"

# Check GPU status
nvidia-smi

# Load environment
source path/MarcVenv/bin/activate

MODEL_PATH="path/LLaDA-8B-Instruct"
CHECKPOINT_PATH="path/Marc_files/checkpoints_2000"
PROMPTS_CSV="path/data/mcq.csv"

srun python lora_batches.py \
  --model_path       "$MODEL_PATH" \
  --prompts_csv      "$PROMPTS_CSV" \
  --mask_id          126336 \
  --steps            128 \
  --gen_length       128 \
  --block_length     32 \
  --temperature      0.0 \
  --cfg_scale        0.0 \
  --kl_coeff         0.1 \
  --num_epochs       3 \
  --batch_size       64 \
  --lr               2e-5 \
  --dataset_size     2000 \
  --seed             42 \
  --checkpoint_dir   "$CHECKPOINT_PATH" \
  --resume

echo "Job completed at: $(date)"


