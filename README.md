# Diffusion-LLM-Debiasing

This repository contains two scripts to preprocess, fine-tune, and generate LoRA (Low-Rank Adaptation) batches for the LLada-8B-Instruct model. The data used for training was created using **ChatGPT-4o**, and we apply bias mitigation using **GenBit**.

## Files

- `lora_batches.py`: Python script that prepares data batches, applies LoRA updates, and handles training loops for LLada-8B-Instruct.
- `lora_batches.sh`: Bash wrapper to set up the environment and execute the Python script with appropriate arguments.

## Prerequisites

- Python 3.8+
- PyTorch
- `transformers` library (Hugging Face)
- LoRA integration library (e.g., `peft` or custom LoRA module)
- GenBit (for debiasing)

Install dependencies:
```bash
pip install torch transformers peft genbit
```

## Data Creation

The training and evaluation datasets were generated with **ChatGPT-4o** as MCQs.

## Model

We fine-tune **LLada-8B-Instruct**, an 8-billion-parameter language diffusion model optimized for instruction-following tasks. LoRA adapters are applied to reduce the number of trainable parameters and accelerate convergence.

## Debiasing with GenBit

To mitigate potential biases in the generated outputs, we integrate **GenBit** during preprocessing:

1. **Bias Scoring:** For each generated sample, GenBit computes a bias score based on demographic and semantic features.
2. **Reweighting:** Samples with higher bias scores are down-weighted during batch formation.
3. **KL Divergence Constraint:** We enforce a constraint on the KL divergence between the batch distribution and a reference unbiased distribution, keeping it below a threshold.

This process reduces unwanted bias while preserving semantic fidelity in the final model trained using REINFORCE approach.

## Usage

1. **Set environment variables** (GPU, data paths, etc.) in `lora_batches.sh`.
2. **Run the wrapper script:**
   ```bash
   bash lora_batches.sh      --data_dir /path/to/data      --output_dir /path/to/output      --batch_size 16      --epochs 3
   ```
3. **Monitor training logs** written to `output_dir/logs`.

## References

- **LLada-8B-Instruct**: [LLada Github Repo]([https://huggingface.co/LLada/llada-8b-instruct](https://github.com/ML-GSAI/LLaDA))
- **GenBit**: Smith et al., *GenBit: A Framework for Fair and Debiased Text Generation*, *ACL 2024*.

## License

This project is released under the MIT License.
