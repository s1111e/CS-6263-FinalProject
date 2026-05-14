#!/usr/bin/env python3
"""Text-only sanity test for direct <action_*> generation.

This removes video and long CoT prompts. It asks one question:
can the current tokenizer/model checkpoint overfit a tiny mapping from a short
text prompt to the gold action tokens?
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "navsim"))

from dataset_utils.sft_dataset import SFTDataset
from models.action_tokenizer import ActionTokenizer


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def atomic_write_json(path: Path, payload: Dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_vlm_checkpoint(model, checkpoint_path: str) -> None:
    if not checkpoint_path:
        return
    print(f"Loading VLM weights from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    vlm_state = {}
    for key, value in state.items():
        if key.startswith("autovla.vlm."):
            vlm_state[key[len("autovla.vlm."):]] = value
    missing, unexpected = model.load_state_dict(vlm_state, strict=False)
    print(f"Loaded VLM state: missing={len(missing)} unexpected={len(unexpected)}")


def extract_action_texts(dataset: SFTDataset, num_samples: int) -> List[Dict]:
    rows = []
    pattern = re.compile(r"(?:<action_\d+>)+")
    for idx in range(min(num_samples, len(dataset))):
        item = dataset[idx]
        matches = pattern.findall(item["text"])
        if not matches:
            continue
        rows.append(
            {
                "dataset_index": idx,
                "data_path": str(item.get("data_path", "")),
                "target": matches[-1],
            }
        )
    return rows


def build_prompt(tokenizer, row: Dict) -> str:
    messages = [
        {
            "role": "system",
            "content": "You output only AutoVLA action tokens. No words. No explanation.",
        },
        {
            "role": "user",
            "content": (
                "Predict the next driving trajectory as exactly 10 action tokens. "
                "Allowed format: <action_i><action_i>..."
            ),
        },
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def encode_example(tokenizer, row: Dict, device: str, max_length: int) -> Dict[str, torch.Tensor]:
    prompt = build_prompt(tokenizer, row)
    answer = row["target"] + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
    input_ids = torch.tensor((prompt_ids + answer_ids)[-max_length:], dtype=torch.long, device=device).unsqueeze(0)
    labels = input_ids.clone()
    prompt_len = min(len(prompt_ids), input_ids.shape[1])
    labels[:, :prompt_len] = -100
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def compute_loss(model, batch: Dict[str, torch.Tensor], action_start: int, n_bins: int, gate_weight: float) -> Dict[str, torch.Tensor]:
    output = model(**batch)
    logits = output.logits[..., :-1, :].contiguous().float()
    labels = batch["labels"][..., 1:].contiguous()
    valid = labels.ne(-100)
    action_end = action_start + n_bins
    action_pos = valid & labels.ge(action_start) & labels.lt(action_end)
    flat_logits = logits.view(-1, logits.shape[-1])
    flat_labels = labels.view(-1)
    flat_action = action_pos.view(-1)
    if not flat_action.any():
        zero = torch.zeros((), device=logits.device)
        return {"loss": zero, "restricted_loss": zero, "gate_loss": zero, "action_count": zero}
    action_logits = torch.nan_to_num(
        flat_logits[flat_action].float(),
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    ).clamp(min=-60.0, max=60.0)
    action_labels = flat_labels[flat_action]
    restricted_logits = action_logits[:, action_start:action_end]
    restricted_labels = action_labels - action_start
    restricted_loss = torch.nan_to_num(
        F.cross_entropy(restricted_logits, restricted_labels),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    action_group_logprob = torch.logsumexp(restricted_logits, dim=-1) - torch.logsumexp(action_logits, dim=-1)
    gate_loss = torch.nan_to_num(-action_group_logprob.mean(), nan=0.0, posinf=0.0, neginf=0.0)
    loss = restricted_loss + gate_weight * gate_loss
    return {
        "loss": loss,
        "restricted_loss": restricted_loss.detach(),
        "gate_loss": gate_loss.detach(),
        "action_count": flat_action.sum().detach(),
    }


@torch.no_grad()
def evaluate(model, tokenizer, rows: List[Dict], device: str, action_start: int, max_new_tokens: int) -> Dict:
    model.eval()
    examples = []
    samples_with_text = 0
    samples_with_ids = 0
    total_text = 0
    total_ids = 0
    for row in rows:
        prompt = build_prompt(tokenizer, row)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        outputs = model.generate(**inputs, do_sample=False, min_new_tokens=1, max_new_tokens=max_new_tokens)
        trimmed = outputs[0, inputs.input_ids.shape[1]:]
        text = tokenizer.decode(trimmed, skip_special_tokens=False)
        text_matches = re.findall(r"<action_\d+>", text)
        id_matches = trimmed[trimmed >= action_start]
        samples_with_text += int(len(text_matches) > 0)
        samples_with_ids += int(id_matches.numel() > 0)
        total_text += len(text_matches)
        total_ids += int(id_matches.numel())
        examples.append(
            {
                "dataset_index": row["dataset_index"],
                "target": row["target"],
                "text_action_count": len(text_matches),
                "token_id_action_count": int(id_matches.numel()),
                "preview": text[:300],
            }
        )
    return {
        "eval_samples": len(rows),
        "samples_with_text_action": samples_with_text,
        "samples_with_token_id_action": samples_with_ids,
        "text_action_sample_rate": samples_with_text / len(rows) if rows else 0.0,
        "token_id_action_sample_rate": samples_with_ids / len(rows) if rows else 0.0,
        "total_text_action_tokens": total_text,
        "total_token_id_action_tokens": total_ids,
        "examples": examples,
    }


def mask_action_row_grads(model, action_start: int, n_bins: int) -> None:
    action_end = action_start + n_bins
    for weight in [model.get_input_embeddings().weight, model.lm_head.weight]:
        if weight.grad is None:
            continue
        weight.grad[:action_start].zero_()
        weight.grad[action_end:].zero_()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/training/qwen2.5-vl-3B-mix-sft.yaml")
    parser.add_argument("--base-checkpoint", default="runs/sft/2026-04-22_08-18-13/epoch=4-loss=1.7191.ckpt")
    parser.add_argument("--output-dir", default="autovla-nuscenes-reproduction/evaluation_results/action_token_only_mini_sft")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gate-weight", type=float, default=5.0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--eval-every-steps", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if torch.cuda.is_available() and "V100" in torch.cuda.get_device_name(0):
        os.environ.setdefault("AUTOVLA_TORCH_DTYPE", "float16")
        os.environ.setdefault("AUTOVLA_ATTN_IMPLEMENTATION", "eager")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"

    config = load_config(args.config)
    processor = AutoProcessor.from_pretrained(config["model"]["pretrained_model_path"], use_fast=True)
    action_tokenizer = ActionTokenizer(processor.tokenizer, model_config=config["model"])
    action_start = int(config["model"]["tokens"]["action_start_id"])

    dataset = SFTDataset(config["data"]["train"], config["model"], processor, using_cot=config["model"].get("use_cot", True))
    rows = extract_action_texts(dataset, args.num_samples)
    if not rows:
        raise RuntimeError("No action-token targets found in selected samples.")

    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config["model"]["pretrained_model_path"], torch_dtype=dtype
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    load_vlm_checkpoint(model, args.base_checkpoint)
    model.to(args.device)
    model.train()

    for param in model.parameters():
        param.requires_grad = False
    model.get_input_embeddings().weight.requires_grad = True
    model.lm_head.weight.requires_grad = True
    trainable_params = []
    seen_params = set()
    for param in [model.get_input_embeddings().weight, model.lm_head.weight]:
        if id(param) not in seen_params:
            trainable_params.append(param)
            seen_params.add(id(param))
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    history = []
    global_step = 0
    eval_rows = rows[: min(10, len(rows))]
    initial_eval = evaluate(model, processor.tokenizer, eval_rows, args.device, action_start, args.max_new_tokens)
    history.append({"type": "eval", "epoch": 0, "global_step": 0, **initial_eval})
    print(f"[eval] step=0 text_action_rate={initial_eval['text_action_sample_rate']:.3f} token_id_action_rate={initial_eval['token_id_action_sample_rate']:.3f}")

    for epoch in range(args.epochs):
        progress = tqdm(rows, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for row in progress:
            batch = encode_example(processor.tokenizer, row, args.device, args.max_length)
            optimizer.zero_grad(set_to_none=True)
            losses = compute_loss(model, batch, action_start, action_tokenizer.n_bins, args.gate_weight)
            if not torch.isfinite(losses["loss"]):
                history.append({
                    "type": "skip_nonfinite",
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": repr(float(losses["loss"].detach().cpu())),
                    "restricted_loss": repr(float(losses["restricted_loss"].cpu())),
                    "gate_loss": repr(float(losses["gate_loss"].cpu())),
                })
                atomic_write_json(metrics_path, {"rows": rows, "history": history})
                continue
            losses["loss"].backward()
            mask_action_row_grads(model, action_start, action_tokenizer.n_bins)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            global_step += 1
            record = {
                "type": "train",
                "epoch": epoch,
                "global_step": global_step,
                "loss": float(losses["loss"].detach().cpu()),
                "restricted_loss": float(losses["restricted_loss"].cpu()),
                "gate_loss": float(losses["gate_loss"].cpu()),
                "action_tokens_in_batch": int(losses["action_count"].cpu()),
            }
            history.append(record)
            progress.set_postfix(loss=f"{record['loss']:.4f}", restricted=f"{record['restricted_loss']:.4f}", gate=f"{record['gate_loss']:.4f}")
            if args.eval_every_steps and global_step % args.eval_every_steps == 0:
                result = evaluate(model, processor.tokenizer, eval_rows, args.device, action_start, args.max_new_tokens)
                history.append({"type": "eval", "epoch": epoch, "global_step": global_step, **result})
                print(f"[eval] step={global_step} text_action_rate={result['text_action_sample_rate']:.3f} token_id_action_rate={result['token_id_action_sample_rate']:.3f} total_text_actions={result['total_text_action_tokens']}")
                atomic_write_json(metrics_path, {"rows": rows, "history": history})
        atomic_write_json(metrics_path, {"rows": rows, "history": history})

    final_eval = evaluate(model, processor.tokenizer, eval_rows, args.device, action_start, args.max_new_tokens)
    history.append({"type": "eval", "epoch": args.epochs, "global_step": global_step, **final_eval})
    atomic_write_json(metrics_path, {"rows": rows, "history": history, "status": "complete"})
    print(f"[eval] step={global_step} text_action_rate={final_eval['text_action_sample_rate']:.3f} token_id_action_rate={final_eval['token_id_action_sample_rate']:.3f} total_text_actions={final_eval['total_text_action_tokens']}")
    print(f"Done. Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
