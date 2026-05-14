#!/usr/bin/env python3
"""Audit action-token gradient flow on the real multimodal AutoVLA batch."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "navsim"))

from dataset_utils.sft_dataset import DataCollator, SFTDataset
from models.autovla import SFTAutoVLA


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def compute_restricted_loss(model: SFTAutoVLA, batch: Dict, gate_weight: float):
    # AutoVLA.forward mutates its input by popping gt fields, so use a shallow copy.
    model_batch = dict(batch)
    output = model.autovla(model_batch)
    logits = output.logits[..., :-1, :].contiguous().float()
    labels = batch["labels"][..., 1:].contiguous()
    valid = labels.ne(model.cfg["model"]["tokens"]["ignore_index"])
    valid = valid & labels.ne(model.autovla.action_tokenizer.tokenizer.pad_token_id)
    action_start = model.autovla.action_start_id
    action_end = action_start + model.autovla.action_tokenizer.n_bins
    action_pos = valid & labels.ge(action_start) & labels.lt(action_end)
    flat_logits = logits.view(-1, logits.shape[-1])
    flat_labels = labels.view(-1)
    flat_action = action_pos.view(-1)
    if not flat_action.any():
        zero = torch.zeros((), device=logits.device)
        return zero, zero, zero, 0
    action_logits = torch.nan_to_num(
        flat_logits[flat_action].float(), nan=0.0, posinf=1e4, neginf=-1e4
    ).clamp(min=-60.0, max=60.0)
    action_labels = flat_labels[flat_action]
    restricted_logits = action_logits[:, action_start:action_end]
    restricted_labels = action_labels - action_start
    restricted_loss = F.cross_entropy(restricted_logits, restricted_labels)
    group_logprob = torch.logsumexp(restricted_logits, dim=-1) - torch.logsumexp(action_logits, dim=-1)
    gate_loss = -group_logprob.mean()
    loss = restricted_loss + gate_weight * gate_loss
    return loss, restricted_loss.detach(), gate_loss.detach(), int(flat_action.sum().item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/training/qwen2.5-vl-3B-mix-sft.yaml")
    parser.add_argument("--checkpoint", default="runs/sft/2026-04-22_08-18-13/epoch=4-loss=1.7191.ckpt")
    parser.add_argument("--output-json", default="autovla-nuscenes-reproduction/evaluation_results/multimodal_action_grad_flow_audit.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gate-weight", type=float, default=1.0)
    parser.add_argument("--min-pixels", type=int, default=1024)
    parser.add_argument("--max-pixels", type=int, default=2048)
    args = parser.parse_args()

    os.environ["AUTOVLA_TORCH_DTYPE"] = args.dtype
    if torch.cuda.is_available() and "V100" in torch.cuda.get_device_name(0):
        os.environ.setdefault("AUTOVLA_ATTN_IMPLEMENTATION", "eager")

    config = load_config(args.config)
    config["training"]["batch_size"] = 1
    config["training"]["num_workers"] = 0
    config["model"].setdefault("video", {})["min_pixels"] = args.min_pixels
    config["model"].setdefault("video", {})["max_pixels"] = args.max_pixels

    processor = AutoProcessor.from_pretrained(config["model"]["pretrained_model_path"], use_fast=True)
    dataset = SFTDataset(config["data"]["train"], config["model"], processor, using_cot=config["model"].get("use_cot", True))
    collator = DataCollator(
        processor=processor,
        ignore_index=config["model"]["tokens"]["ignore_index"],
        assistant_id=config["model"]["tokens"]["assistant_id"],
    )
    loader = DataLoader(Subset(dataset, [0]), batch_size=1, collate_fn=collator, num_workers=0)
    batch = next(iter(loader))

    model = SFTAutoVLA(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint missing={len(missing)} unexpected={len(unexpected)}")
    model.to(args.device)
    model.train()

    for param in model.parameters():
        param.requires_grad = False
    emb = model.autovla.vlm.get_input_embeddings().weight
    head = model.autovla.vlm.lm_head.weight
    emb.requires_grad = True
    head.requires_grad = True
    params = []
    seen = set()
    for param in [emb, head]:
        if id(param) not in seen:
            params.append(param)
            seen.add(id(param))

    batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    action_start = model.autovla.action_start_id
    action_end = action_start + model.autovla.action_tokenizer.n_bins
    emb_before = emb[action_start:action_end].detach().float().clone()
    head_before = head[action_start:action_end].detach().float().clone()

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    loss_before, restricted_before, gate_before, action_count = compute_restricted_loss(model, batch, args.gate_weight)
    opt.zero_grad(set_to_none=True)
    loss_before.backward()
    emb_grad = emb.grad[action_start:action_end].detach().float() if emb.grad is not None else torch.zeros_like(emb_before)
    head_grad = head.grad[action_start:action_end].detach().float() if head.grad is not None else torch.zeros_like(head_before)
    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
    opt.step()

    emb_after = emb[action_start:action_end].detach().float().clone()
    head_after = head[action_start:action_end].detach().float().clone()
    loss_after, restricted_after, gate_after, _ = compute_restricted_loss(model, batch, args.gate_weight)

    summary = {
        "dtype": args.dtype,
        "lr": args.lr,
        "action_count": action_count,
        "loss_before": float(loss_before.detach().cpu()),
        "restricted_before": float(restricted_before.cpu()),
        "gate_before": float(gate_before.cpu()),
        "loss_after_one_step": float(loss_after.detach().cpu()),
        "restricted_after_one_step": float(restricted_after.cpu()),
        "gate_after_one_step": float(gate_after.cpu()),
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
