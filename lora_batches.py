import logging
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model
import sys
import argparse
import ast
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from genbit.genbit_metrics import GenBitMetrics
import pandas as pd
import random
import os
import glob

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def get_logits(model, batch, prompt_index, cfg_scale, mask_id):
    logger.debug(f"get_logits called: batch.shape={batch.shape}, cfg_scale={cfg_scale}")
    if cfg_scale > 0.:
        prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
        un_batch = batch.clone()
        un_batch[prompt_index] = mask_id
        batch = torch.cat([batch, un_batch])
    logits = model(batch).logits
    if cfg_scale > 0.:
        logits, un_logits = torch.chunk(logits, 2, dim=0)
        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
    return logits

def compute_genbit_bias(text):
    genbit_metrics = GenBitMetrics('EN', context_window=5, distance_weight=0.95, percentile_cutoff=80)
    genbit_metrics.add_data(text, tokenized=False)
    metrics = genbit_metrics.get_metrics(output_statistics=True, output_word_list=True)
    bias = metrics['additional_metrics']['avg_bias_ratio']
    return np.abs(bias)

def add_gumbel_noise(logits, temperature):
    logger.debug(f"add_gumbel_noise called: logits.shape={logits.shape}, temperature={temperature}")
    if temperature == 0:
        return logits
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = (- torch.log(noise)) ** temperature
    sampled = logits.exp() / gumbel_noise
    logger.debug("Gumbel noise applied and logits sampled.")
    return sampled

def get_num_transfer_tokens(mask_index, steps):
    logger.debug(f"get_num_transfer_tokens called: mask_index.shape={mask_index.shape}, steps={steps}")
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int32) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    logger.debug(f"Computed num_transfer_tokens: {num_transfer_tokens}")
    return num_transfer_tokens

@torch.no_grad()
def generate(
    model,
    prompt,             # Tensor (B, Lp) or (Lp,)
    prompt_lens=None,   # Tensor of shape (B,) giving the true length of each prompt
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.,
    cfg_scale=0.,
    remasking='low_confidence',
    mask_id=126336
):
    device = model.device

    # === make prompt 2D and infer B, Lp ===
    if prompt.dim() == 1:
        prompt = prompt.unsqueeze(0)             # now (1, Lp)
    B, Lp = prompt.shape

    # === build prompt_lens if not provided ===
    if prompt_lens is None:
        prompt_lens = torch.full((B,), Lp, dtype=torch.long, device=device)
    else:
        prompt_lens = prompt_lens.to(device)

    # === initialize x and prompt_index ===
    x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=device)
    prompt_index = torch.zeros_like(x, dtype=torch.bool, device=device)

    # copy in each prompt up to its real length
    for b in range(B):
        l = prompt_lens[b].item()
        x[b, :l] = prompt[b, :l]
        prompt_index[b, :l] = True

    # === same block / denoising logic as before ===
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    per_block_steps = steps // num_blocks

    for block in range(num_blocks):
        start = Lp + block * block_length
        end   = start + block_length
        block_mask_index = (x[:, start:end] == mask_id)
        num_transfer = get_num_transfer_tokens(block_mask_index, per_block_steps)

        for i in range(per_block_steps):
            mask_index = (x == mask_id)

            # classifier-free guidance branch
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_combined = torch.cat([x, un_x], dim=0)
                logits = model(x_combined).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            # add gumbel noise & pick argmax
            logits_noisy = add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_noisy, dim=-1)

            # compute confidence for remasking
            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float32), dim=-1)
                probs = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == 'random':
                probs = torch.rand_like(x0, dtype=torch.float32)
            else:
                raise NotImplementedError

            probs[:, Lp + (block+1)*block_length:] = -float('inf')
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, probs, -float('inf'))

            transfer_mask = torch.zeros_like(x0, dtype=torch.bool, device=device)
            for b in range(B):
                _, idx = torch.topk(confidence[b], k=num_transfer[b, i])
                transfer_mask[b, idx] = True

            x[transfer_mask] = x0[transfer_mask]

    return x  # shape (B, Lp + gen_length)

# --- Utility: Load Model and Tokenizer ---
def load_model(model_path, device, r=4, lora_alpha=8, lora_dropout=0.1, target_modules=["q_proj", "k_proj", "v_proj", "attn_out"]):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias='none',
        task_type=None
    )
    lora_model = get_peft_model(base_model, lora_config).to(device)
    return base_model, lora_model, tokenizer


class PromptTextDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts
    def __len__(self):
        return len(self.prompts)
    def __getitem__(self, idx):
        return self.prompts[idx]

# --- RL Training Loop ---
def lora_reinforce_train_loop(
    model_path,
    prompts,
    mask_id=126336,
    steps=128,
    gen_length=128,
    block_length=32,
    temperature=0.,
    cfg_scale=0.,
    kl_coeff=0.1,
    num_epochs=3,
    batch_size=64,
    microbatch_size=8,
    lr=2e-5,
    seed=42,
    checkpoint_dir="checkpoints",
    resume=False,
):
    assert batch_size % microbatch_size == 0, "batch_size must be divisible by microbatch_size"
    accumulation_steps = batch_size // microbatch_size
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_model, lora_model, tokenizer = load_model(model_path, device)
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, "epoch_*.pt")))
    if resume and ckpts:
        latest = ckpts[-1]
        last_epoch = int(os.path.splitext(os.path.basename(latest))[0].split("_")[1])
        logger.info(f"Resuming from epoch {last_epoch}")
        lora_model.load_state_dict(torch.load(latest, map_location=device))
    else:
        last_epoch = 0
        if resume:
            logger.info("No checkpoint found; starting fresh.")

    # setup metrics CSV
    metrics_file = os.path.join(checkpoint_dir, "metrics.csv")
    if last_epoch == 0:
        # create an empty DataFrame and write header
        df_metrics = pd.DataFrame(columns=["epoch","loss","avg_bias","avg_KL"])
        df_metrics.to_csv(metrics_file, index=False)
    else:
        # load existing metrics
        df_metrics = pd.read_csv(metrics_file)
        
    lora_model.train()
    generator = torch.Generator().manual_seed(seed)
    optimizer = AdamW(lora_model.parameters(), lr=lr)
    dataset = PromptTextDataset(prompts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, generator=generator)

    for epoch in range(1, num_epochs + 1):
        for batch_idx, batch in enumerate(dataloader, start=1):
            current_epoch = last_epoch + epoch
            logger.info(f" Epoch {current_epoch} — Batch {batch_idx}/{len(dataloader)} — zeroing gradients")
            # zero out gradients before accumulating
            optimizer.zero_grad()
            num_micro = (batch_size + microbatch_size - 1) // microbatch_size
            # split this batch into microbatches for forward/backward
            for mb_idx, i in enumerate(range(0, batch_size, microbatch_size), start=1):
                
                
                sub_batch = batch[i : i + microbatch_size]
                logger.info(
                                f"  Microbatch {mb_idx}/{num_micro} — size {len(sub_batch)}"
                            )
                # ------------- 1) batch‐tokenize -------------
                texts = [
                    tokenizer.apply_chat_template(
                        [{"role":"user","content":t}],
                        add_generation_prompt=True,
                        tokenize=False
                    )
                    for t in sub_batch
                ]
                enc = tokenizer(texts, return_tensors="pt", padding=True)
                input_ids       = enc["input_ids"].to(device)       # (Mb, Lp)
                attention_mask  = enc["attention_mask"].to(device)  # (Mb, Lp)
                prompt_lens     = attention_mask.sum(dim=1)         # (Mb,)

                # ------------- 2) generate (microbatch) -------------
                torch.cuda.empty_cache()
                output_ids = generate(
                    lora_model,
                    input_ids,
                    prompt_lens=prompt_lens,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    remasking='low_confidence',
                    mask_id=mask_id
                )  # (Mb, Lp+Lg)
                torch.cuda.empty_cache()
                Mb, total_len = output_ids.shape

                # ------------- 3) decode & bias -------------
                gen_texts = []
                for b in range(Mb):
                    start = prompt_lens[b].item()
                    gen_ids = output_ids[b, start:]
                    gen_texts.append(
                        tokenizer.decode(gen_ids, skip_special_tokens=True)
                    )
                bias_list   = [compute_genbit_bias(txt) for txt in gen_texts]
                bias_tensor = torch.tensor(bias_list, device=device)  # (Mb,)

                # ------------- 4) logits & KL -------------
                prompt_index = (torch.arange(total_len, device=device)
                                .unsqueeze(0).repeat(Mb,1)
                               ) < prompt_lens.unsqueeze(1)
                with torch.no_grad():
                    base_logits = get_logits(
                        base_model, output_ids, prompt_index, cfg_scale, mask_id
                    )  # (Mb, total_len, V)
                lora_logits = get_logits(
                    lora_model, output_ids, prompt_index, cfg_scale, mask_id
                )

                base_p     = torch.softmax(base_logits,    dim=-1)
                lora_log_p = torch.log_softmax(lora_logits, dim=-1)
                kl_tok     = (base_p * (torch.log(base_p + 1e-8) - lora_log_p)).sum(dim=-1)
                kl_vector  = torch.stack([
                    kl_tok[b, prompt_lens[b]:].mean()
                    for b in range(Mb)
                ])  # (Mb,)

                # ------------- 5) rewards & log‐prob sums -------------
                rewards      = -(bias_tensor + kl_coeff * kl_vector)        # (Mb,)
                gen_tokens   = output_ids[:, prompt_lens.max():]           # (Mb, Lg)
                log_probs    = torch.log_softmax(lora_logits, dim=-1)
                shifted_logits = log_probs[:, prompt_lens.max()-1:-1, :]   # (Mb, Lg, V)
                gen_logps      = shifted_logits.gather(
                                    -1,
                                    gen_tokens.unsqueeze(-1)
                                 ).squeeze(-1)                     # (Mb, Lg)
                logprob_sums   = gen_logps.sum(dim=1)                     # (Mb,)

                # ------------- 6) scaled REINFORCE loss & backward -------------
                loss = (logprob_sums * rewards).mean() / accumulation_steps
                loss.backward()

            # now that we've accumulated over all microbatches, do one step
            optimizer.step()
        ckpt_path = os.path.join(checkpoint_dir, f"epoch_{current_epoch}.pt")
        torch.save(lora_model.state_dict(), ckpt_path)
        logger.info(f"Saved checkpoint: {ckpt_path}")
        logger.info(
            f"[Epoch {current_epoch}] loss={loss.item():.4f} "
            f"avg_bias={bias_tensor.mean().item():.4f} "
            f"avg_KL={kl_vector.mean().item():.4f}"
        )
        new_row = {
            "epoch": current_epoch,
            "loss":  loss.item(),
            "avg_bias": bias_tensor.mean().item(),
            "avg_KL":   kl_vector.mean().item()
        }
        df_metrics = pd.concat([df_metrics, pd.DataFrame([new_row])], ignore_index=True)
        df_metrics.to_csv(metrics_file, index=False)
    lora_model.save_pretrained(os.path.join(checkpoint_dir, "final"))
    logger.info("Training complete; final model saved.")

def parse_args():
    p = argparse.ArgumentParser(description="LoRA REINFORCE training")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument(
        "--prompts_csv",
        type=str,
        required=True,
        help="Path to a CSV file containing a column 'prompt'"
    )
    p.add_argument("--mask_id",      type=int,   default=126336)
    p.add_argument("--steps",        type=int,   default=128)
    p.add_argument("--gen_length",   type=int,   default=128)
    p.add_argument("--block_length", type=int,   default=32)
    p.add_argument("--temperature",  type=float, default=0.0)
    p.add_argument("--cfg_scale",    type=float, default=0.0)
    p.add_argument("--kl_coeff",     type=float, default=0.1)
    p.add_argument("--num_epochs",   type=int,   default=20)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=2e-5)
    p.add_argument("--seed",         type=int,   default=42,
                   help="Random seed for sampling and training")
    p.add_argument("--dataset_size", type=int,   default=2000,
                   help="Number of samples to draw from prompts CSV")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                   help="Directory to save/load checkpoints")
    p.add_argument("--resume", action="store_true",
                   help="Resume training from latest checkpoint")
    return p.parse_args()

    
    
if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    df = pd.read_csv(args.prompts_csv)
    df = df.sample(n=args.dataset_size, random_state=args.seed)
    prompts = df["prompt"].astype(str).tolist()


    lora_reinforce_train_loop(
        args.model_path,
        prompts,
        mask_id      = args.mask_id,
        steps        = args.steps,
        gen_length   = args.gen_length,
        block_length = args.block_length,
        temperature  = args.temperature,
        cfg_scale    = args.cfg_scale,
        kl_coeff     = args.kl_coeff,
        num_epochs   = args.num_epochs,
        batch_size   = args.batch_size,
        lr           = args.lr,
        seed         = args.seed,
        checkpoint_dir = args.checkpoint_dir,
        resume       = args.resume
    )

