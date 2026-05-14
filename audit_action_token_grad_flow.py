#!/usr/bin/env python3
"""Audit whether action-token loss produces gradients and parameter updates."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "navsim"))

from dataset_utils.sft_dataset import SFTDataset
from models.action_tokenizer import ActionTokenizer


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_vlm_checkpoint(model, checkpoint_path: str) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    vlm_state = {k[len("autovla.vlm."):]: v for k, v in state.items() if k.startswith("autovla.vlm.")}
    missing, unexpected = model.load_state_dict(vlm_state, strict=False)
    print(f"Loaded VLM state: missing={len(missing)} unexpected={len(unexpected)}")


def get_first_target(dataset: SFTDataset) -> str:
    pattern = re.compile(r"(?:<action_\d+>)+")
    for idx in range(len(dataset)):
        item = dataset[idx]
        matches = pattern.findall(item["text"])
        if matches:
            print(f"Using dataset index {idx}, target={matches[-1][:120]}")
            return matches[-1]
    raise RuntimeError("No action-token target found")


def build_batch(tokenizer, target: str, device: str):
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You output only AutoVLA action tokens. No words."},
            {"role": "user", "content": "Predict exactly 10 action tokens."},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    answer_ids = tokenizer(target + tokenizer.eos_token, add_special_tokens=False).input_ids
    input_ids = torch.tensor(prompt_ids + answer_ids, dtype=torch.long, device=device).unsqueeze(0)
    labels = input_ids.clone()
    labels[:, : len(prompt_ids)] = -100
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": labels}


def restricted_loss(model, batch, action_start: int, n_bins: int):
    out = model(**batch)
    logits = out.logits[..., :-1, :].contiguous().float()
    labels = batch["labels"][..., 1:].contiguous()
    valid = labels.ne(-100)
    action_end = action_start + n_bins
    action_pos = valid & labels.ge(action_start) & labels.lt(action_end)
    flat_logits = logits.view(-1, logits.shape[-1])
    flat_labels = labels.view(-1)
    flat_action = action_pos.view(-1)
    action_logits = flat_logits[flat_action]
    action_labels = flat_labels[flat_action]
    restricted_logits = action_logits[:, action_start:action_end]
    restricted_labels = action_labels - action_start
    loss = F.cross_entropy(restricted_logits, restricted_labels)
    return loss, int(flat_action.sum().item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/training/qwen2.5-vl-3B-mix-sft.yaml")
    parser.add_argument("--checkpoint", default="runs/sft/2026-04-22_08-18-13/epoch=4-loss=1.7191.ckpt")
    parser.add_argument("--output-json", default="autovla-nuscenes-reproduction/evaluation_results/action_token_grad_flow_audit.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float32")
    args = parser.parse_args()

    config = load_config(args.config)
    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    processor = AutoProcessor.from_pretrained(config["model"]["pretrained_model_path"], use_fast=True)
    action_tokenizer = ActionTokenizer(processor.tokenizer, model_config=config["model"])
    action_start = int(config["model"]["tokens"]["action_start_id"])
    action_end = action_start + action_tokenizer.n_bins

    dataset = SFTDataset(config["data"]["train"], config["model"], processor, using_cot=config["model"].get("use_cot", True))
    target = get_first_target(dataset)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(config["model"]["pretrained_model_path"], torch_dtype=dtype)
    model.resize_token_embeddings(len(processor.tokenizer))
    load_vlm_checkpoint(model, args.checkpoint)
    model.to(args.device)
    model.train()

    for p in model.parameters():
        p.requires_grad = False
    model.get_input_embeddings().weight.requires_grad = True
    model.lm_head.weight.requires_grad = True

    batch = build_batch(processor.tokenizer, target, args.device)
    emb = model.get_input_embeddings().weight
    head = model.lm_head.weight
    emb_before = emb[action_start:action_end].detach().float().clone()
    head_before = head[action_start:action_end].detach().float().clone()

    optimizer = torch.optim.AdamW([emb, head], lr=args.lr)
    loss_before, action_count = restricted_loss(model, batch, action_start, action_tokenizer.n_bins)
    optimizer.zero_grad(set_to_none=True)
    loss_before.backward()

    emb_grad = emb.grad[action_start:action_end].detach().float() if emb.grad is not None else torch.zeros_like(emb_before)
    head_grad = head.grad[action_start:action_end].detach().float() if head.grad is not None else torch.zeros_like(head_before)
    optimizer.step()

    emb_after = emb[action_start:action_end].detach().float().clone()
    head_after = head[action_start:action_end].detach().float().clone()
    loss_after, _ = restricted_loss(model, batch, action_start, action_tokenizer.n_bins)

    summary = {
        "dtype": args.dtype,
        "lr": args.lr,
        "action_count": action_count,
        "loss_before": float(loss_before.detach().cpu()),
        "loss_after_one_step": float(loss_after.detach().cpu()),
        "embedding_grad_norm": float(emb_grad.norm().cpu()),
        "lm_head_grad_norm": float(head_grad.norm().cpu()),
        "embedding_delta_norm": float((emb_after - emb_before).norm().cpu()),
        "lm_head_delta_norm": float((head_after - head_before).norm().cpu()),
        "embedding_is_lm_head_same_object": emb.data_ptr() == head.data_ptr(),
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for k, v in summary.items():
        print(f"{k}={v}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
