#!/usr/bin/env python3
"""Two-phase action-token repair for AutoVLA multimodal checkpoint.

Phase 1 — Text warm-up:
  Identical approach to action_token_only_mini_sft.py (proven to work).
  Short text prompt, no video. Runs until text_action_rate hits --phase1-target-rate
  or for --phase1-epochs, whichever comes first. Can be skipped with --phase1-epochs 0.

Phase 2 — Multimodal repair:
  Uses the EXACT prompt that nusc_eval uses (get_prompt → "<answer>\\nThe final output
  action is: "), appends the gold action tokens as the target, and restricts the CE loss
  to those positions only.  After every --eval-every-steps steps it generates on the
  same samples to observe whether action tokens appear.  No checkpoint is written
  unless --save-checkpoint is passed.

Root cause addressed: In the multimodal setting the natural-language prior from the
long CoT prompt overwhelms the action-segment loss when computed over the full sequence.
This script focuses the gradient purely on the action output positions plus a gate loss
that pushes probability mass into the action token group.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "navsim"))

from dataset_utils.sft_dataset import SFTDataset
from models.action_tokenizer import ActionTokenizer


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def atomic_write_json(path: Path, payload: Dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_vlm_checkpoint(model: Qwen2_5_VLForConditionalGeneration, checkpoint_path: str) -> None:
    if not checkpoint_path:
        return
    print(f"Loading VLM weights from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    vlm_state = {}
    for key, value in state.items():
        for prefix in ("autovla.vlm.", "drivevla.vlm.", "vlm."):
            if key.startswith(prefix):
                vlm_state[key[len(prefix):]] = value
                break
    missing, unexpected = model.load_state_dict(vlm_state, strict=False)
    print(f"  missing={len(missing)}  unexpected={len(unexpected)}")


def mask_non_action_grads(model: Qwen2_5_VLForConditionalGeneration, action_start: int, n_bins: int) -> None:
    """Zero gradients on all embedding/lm_head rows except the action token rows."""
    action_end = action_start + n_bins
    for weight in [model.get_input_embeddings().weight, model.lm_head.weight]:
        if weight.grad is None:
            continue
        weight.grad[:action_start].zero_()
        weight.grad[action_end:].zero_()


def compute_action_loss(
    model: Qwen2_5_VLForConditionalGeneration,
    batch: Dict[str, torch.Tensor],
    action_start: int,
    n_bins: int,
    gate_weight: float,
) -> Dict[str, torch.Tensor]:
    """Restricted CE loss + gate loss, computed only on action-token positions."""
    labels = batch.pop("labels")
    try:
        output = model(**batch)
    finally:
        batch["labels"] = labels

    logits = output.logits[..., :-1, :].contiguous().float()
    tgt = labels[..., 1:].contiguous()

    action_end = action_start + n_bins
    action_pos = tgt.ne(-100) & tgt.ge(action_start) & tgt.lt(action_end)

    flat_logits = logits.view(-1, logits.shape[-1])
    flat_labels = tgt.view(-1)
    flat_action = action_pos.view(-1)

    if not flat_action.any():
        z = torch.zeros((), device=logits.device)
        return {"loss": z, "restricted_loss": z.detach(), "gate_loss": z.detach(), "action_count": z.detach()}

    act_logits = flat_logits[flat_action].clamp(-60.0, 60.0)
    act_labels = flat_labels[flat_action]

    restricted = act_logits[:, action_start:action_end]
    restricted_labels = act_labels - action_start

    restricted_loss = F.cross_entropy(restricted, restricted_labels)
    gate_lp = torch.logsumexp(restricted, dim=-1) - torch.logsumexp(act_logits, dim=-1)
    gate_loss = -gate_lp.mean()

    loss = restricted_loss + gate_weight * gate_loss
    return {
        "loss": loss,
        "restricted_loss": restricted_loss.detach(),
        "gate_loss": gate_loss.detach(),
        "action_count": flat_action.sum().detach(),
    }


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def extract_action_rows(dataset: SFTDataset, num_samples: int) -> List[Dict]:
    """Extract gold action token strings from SFTDataset items."""
    rows: List[Dict] = []
    pattern = re.compile(r"(?:<action_\d+>)+")
    for idx in range(min(num_samples, len(dataset))):
        item = dataset[idx]
        matches = pattern.findall(item.get("text", ""))
        if not matches:
            continue
        scene_path, sensor_path = dataset.scenes[idx]
        rows.append({
            "dataset_index": idx,
            "scene_path": str(scene_path),
            "sensor_data_path": str(sensor_path),
            "target": matches[-1],
        })
    return rows


# ---------------------------------------------------------------------------
# Phase 1: Text-only warm-up
# ---------------------------------------------------------------------------

def _text_prompt(tokenizer) -> str:
    messages = [
        {"role": "system", "content": "You output only AutoVLA action tokens. No words. No explanation."},
        {"role": "user", "content": (
            "Predict the next driving trajectory as exactly 10 action tokens. "
            "Allowed format: <action_i><action_i>..."
        )},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def encode_text_example(tokenizer, row: Dict, device: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
    prompt = _text_prompt(tokenizer)
    answer = row["target"] + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
    input_ids = torch.tensor(
        (prompt_ids + answer_ids)[-max_length:], dtype=torch.long, device=device
    ).unsqueeze(0)
    labels = input_ids.clone()
    labels[:, : min(len(prompt_ids), input_ids.shape[1])] = -100
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": labels}


@torch.no_grad()
def eval_text(model, tokenizer, rows, device, action_start, max_new_tokens) -> Dict:
    model.eval()
    n_text = n_ids = 0
    examples = []
    for row in rows:
        prompt = _text_prompt(tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        trimmed = out[0, inputs.input_ids.shape[1]:]
        text = tokenizer.decode(trimmed, skip_special_tokens=False)
        tm = re.findall(r"<action_\d+>", text)
        im = trimmed[trimmed >= action_start]
        n_text += int(bool(tm))
        n_ids += int(im.numel() > 0)
        examples.append({"dataset_index": row["dataset_index"], "target": row["target"],
                          "preview": text[:200], "text_action_count": len(tm)})
    n = len(rows)
    return {"eval_samples": n,
            "text_action_rate": n_text / n if n else 0.0,
            "token_id_action_rate": n_ids / n if n else 0.0,
            "examples": examples}


# ---------------------------------------------------------------------------
# Phase 2: Multimodal repair
# ---------------------------------------------------------------------------

def _build_multimodal_messages(config: Dict, row: Dict):
    """Return (messages, video_inputs) using the exact eval prompt format (no-CoT)."""
    with open(row["scene_path"]) as f:
        scene_data = json.load(f)

    video_conf = config["model"]["video"]
    min_px = video_conf.get("min_pixels", 28 * 28 * 128)
    max_px = video_conf.get("max_pixels", 28 * 28 * 128)

    # Actual NuScenes scene JSON keys (absolute paths, no sensor_data_path join needed)
    camera_images = {
        "front_camera":       scene_data["front_camera_paths"],
        "front_left_camera":  scene_data["front_left_camera_paths"],
        "front_right_camera": scene_data["front_right_camera_paths"],
    }

    vel = scene_data.get("velocity", 0.0)
    if isinstance(vel, (list, np.ndarray)):
        vel = float(np.sqrt(vel[0] ** 2 + vel[1] ** 2))
    else:
        vel = float(vel)

    acc = scene_data.get("acceleration", 0.0)
    if isinstance(acc, (list, np.ndarray)):
        acc = float(np.sqrt(acc[0] ** 2 + acc[1] ** 2))
    else:
        acc = float(acc)

    instruction = scene_data.get("instruction", "go straight").lower()

    user_content = [
        {"type": "text", "text": "The autonomous vehicle is equipped with three cameras mounted at the front, left, and right, enabling a comprehensive perception of the surrounding environment."},
        {"type": "text", "text": "The first video presents the front view of the vehicle, comprising four sequential frames sampled at 2 Hz."},
        {"type": "video", "min_pixels": min_px, "max_pixels": max_px,
         "video": [f"file://{p}" for p in camera_images["front_camera"]]},
        {"type": "text", "text": "The second video presents the front-left view of the vehicle, comprising four sequential frames sampled at 2 Hz."},
        {"type": "video", "min_pixels": min_px, "max_pixels": max_px,
         "video": [f"file://{p}" for p in camera_images["front_left_camera"]]},
        {"type": "text", "text": "The third video presents the front-right view of the vehicle, comprising four sequential frames sampled at 2 Hz."},
        {"type": "video", "min_pixels": min_px, "max_pixels": max_px,
         "video": [f"file://{p}" for p in camera_images["front_right_camera"]]},
        {"type": "text", "text": (
            f"The current velocity of the vehicle is {vel:.3f} m/s, and the current acceleration is {acc:.3f} m/s². "
            f"The driving instruction is: {instruction}. Based on this information, plan the action trajectory for the autonomous vehicle over the next five seconds."
        )},
    ]

    # Mirror the no-CoT system prompt from autovla.py:get_prompt exactly
    messages = [
        {"role": "system", "content": [{"type": "text", "text": (
            "You are an Advanced Driver Assistance and Full Self-Driving System. "
            "You will be provided with video observations from the ego vehicle's surrounding cameras, along with the vehicle's current dynamic states. "
            "Your task is to predict the most appropriate driving action for the next five seconds.\n"
            "Output action trajectory tokens using this exact prefix format: "
            "The final output action is: <action_...>"
        )}]},
        {"role": "user", "content": user_content},
    ]
    _, video_inputs = process_vision_info(messages)
    return messages, video_inputs


def _build_prompt_text(processor, messages) -> str:
    """Apply chat template and append the answer prefix (identical to eval)."""
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
    )
    text += "<answer>\nThe final output action is: "
    return text


def encode_multimodal_example(
    processor, config: Dict, row: Dict, device: str, max_length: int
) -> Optional[Dict[str, torch.Tensor]]:
    """Encode one multimodal example with labels masked to action positions only."""
    try:
        messages, video_inputs = _build_multimodal_messages(config, row)
        prompt_text = _build_prompt_text(processor, messages)
        answer_text = row["target"] + processor.tokenizer.eos_token

        # Encode prompt alone to get its token length
        prompt_enc = processor(
            text=[prompt_text], videos=video_inputs, padding=False, return_tensors="pt"
        )
        prompt_len = prompt_enc.input_ids.shape[1]

        # Encode full sequence
        full_enc = processor(
            text=[prompt_text + answer_text], videos=video_inputs, padding=False, return_tensors="pt"
        )

        # Adjust prompt_len after possible left-truncation of sequence
        trunc_offset = max(0, full_enc.input_ids.shape[1] - max_length)
        effective_prompt_len = max(0, prompt_len - trunc_offset)

        # Build batch: only truncate sequence-length tensors (input_ids, attention_mask)
        batch = {}
        for k, v in full_enc.items():
            if not isinstance(v, torch.Tensor):
                continue
            if k in ("input_ids", "attention_mask"):
                batch[k] = v[:, -max_length:].to(device)
            else:
                batch[k] = v.to(device)

        labels = batch["input_ids"].clone()
        labels[:, :effective_prompt_len] = -100  # mask everything before the answer
        batch["labels"] = labels
        return batch

    except Exception as e:
        print(f"[WARN] encode_multimodal_example failed (idx={row['dataset_index']}): {e}")
        return None


@torch.no_grad()
def eval_multimodal(model, processor, config, rows, device, action_start, max_new_tokens) -> Dict:
    model.eval()
    n_text = n_ids = 0
    examples = []
    for row in rows:
        try:
            messages, video_inputs = _build_multimodal_messages(config, row)
            prompt_text = _build_prompt_text(processor, messages)
            inputs = processor(text=[prompt_text], videos=video_inputs, padding=False, return_tensors="pt")
            model_inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            out = model.generate(**model_inputs, do_sample=False, max_new_tokens=max_new_tokens)
            trimmed = out[0, model_inputs["input_ids"].shape[1]:]
            text = processor.tokenizer.decode(trimmed, skip_special_tokens=False)
            tm = re.findall(r"<action_\d+>", text)
            im = trimmed[trimmed >= action_start]
            n_text += int(bool(tm))
            n_ids += int(im.numel() > 0)
            examples.append({
                "dataset_index": row["dataset_index"],
                "target": row["target"],
                "text_action_count": len(tm),
                "token_id_action_count": int(im.numel()),
                "preview": text[:300],
            })
        except Exception as e:
            print(f"[WARN] eval_multimodal failed (idx={row['dataset_index']}): {e}")
    n = len(rows)
    return {
        "eval_samples": n,
        "text_action_rate": n_text / n if n else 0.0,
        "token_id_action_rate": n_ids / n if n else 0.0,
        "examples": examples,
    }


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/training/qwen2.5-vl-3B-mix-sft.yaml")
    p.add_argument("--base-checkpoint", default="runs/sft/2026-04-22_08-18-13/epoch=4-loss=1.7191.ckpt")
    p.add_argument("--output-dir", default="autovla-nuscenes-reproduction/evaluation_results/multimodal_repair")
    p.add_argument("--num-samples", type=int, default=10,
                   help="Training samples (both phases)")
    p.add_argument("--eval-samples", type=int, default=5,
                   help="Samples used for generation evaluation (taken from num-samples)")
    # Phase 1
    p.add_argument("--phase1-epochs", type=int, default=30,
                   help="Text-only warm-up epochs; set 0 to skip")
    p.add_argument("--phase1-target-rate", type=float, default=0.9,
                   help="Early-stop Phase 1 when text_action_rate >= this")
    p.add_argument("--phase1-lr", type=float, default=1e-3)
    p.add_argument("--phase1-gate-weight", type=float, default=5.0)
    # Phase 2
    p.add_argument("--phase2-steps", type=int, default=200)
    p.add_argument("--phase2-lr", type=float, default=3e-4)
    p.add_argument("--gate-weight", type=float, default=15.0,
                   help="Gate loss coefficient for Phase 2 (higher = stronger push into action group)")
    p.add_argument("--eval-every-steps", type=int, default=20)
    # Common
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["float16", "float32"], default="float32")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-checkpoint", action="store_true",
                   help="Save repaired embedding+lm_head weights at end")
    p.add_argument("--save-every-eval", action="store_true",
                   help="Also save a checkpoint at every eval interval (step_N.pt); lets you pick best before mode collapse")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"

    config = load_config(args.config)
    processor = AutoProcessor.from_pretrained(config["model"]["pretrained_model_path"], use_fast=True)
    action_tokenizer = ActionTokenizer(processor.tokenizer, model_config=config["model"])
    action_start = int(config["model"]["tokens"]["action_start_id"])
    n_bins = action_tokenizer.n_bins

    dataset = SFTDataset(
        config["data"]["train"],
        config["model"],
        processor,
        using_cot=config["model"].get("use_cot", True),
    )
    rows = extract_action_rows(dataset, args.num_samples)
    if not rows:
        raise RuntimeError("No action-token targets found in dataset. Check data paths.")
    eval_rows = rows[: min(args.eval_samples, len(rows))]
    print(f"Training rows: {len(rows)}   Eval rows: {len(eval_rows)}")

    # Load model
    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config["model"]["pretrained_model_path"], torch_dtype=dtype
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    load_vlm_checkpoint(model, args.base_checkpoint)
    model.to(args.device)

    # Freeze everything; only train action token rows of embedding + lm_head
    for param in model.parameters():
        param.requires_grad = False
    model.get_input_embeddings().weight.requires_grad = True
    model.lm_head.weight.requires_grad = True
    seen: set = set()
    trainable: List[torch.Tensor] = []
    for w in [model.get_input_embeddings().weight, model.lm_head.weight]:
        if id(w) not in seen:
            trainable.append(w)
            seen.add(id(w))

    history: List[Dict] = []

    # -----------------------------------------------------------------------
    # Phase 1: Text-only warm-up
    # -----------------------------------------------------------------------
    if args.phase1_epochs > 0:
        print(f"\n=== Phase 1: Text-only warm-up ({args.phase1_epochs} epochs max) ===")
        opt1 = torch.optim.AdamW(trainable, lr=args.phase1_lr)
        model.train()

        ev = eval_text(model, processor.tokenizer, eval_rows, args.device, action_start, args.max_new_tokens)
        history.append({"phase": 1, "type": "eval", "epoch": 0,
                         "text_action_rate": ev["text_action_rate"],
                         "token_id_action_rate": ev["token_id_action_rate"]})
        print(f"[P1 eval] epoch=0  text_action_rate={ev['text_action_rate']:.3f}")

        p1_step = 0
        for epoch in range(args.phase1_epochs):
            if ev["text_action_rate"] >= args.phase1_target_rate and epoch > 0:
                print(f"[P1] Early stop: rate {ev['text_action_rate']:.3f} >= target {args.phase1_target_rate}")
                break
            for row in tqdm(rows, desc=f"P1 epoch {epoch+1}/{args.phase1_epochs}"):
                batch = encode_text_example(processor.tokenizer, row, args.device)
                opt1.zero_grad(set_to_none=True)
                losses = compute_action_loss(model, batch, action_start, n_bins, args.phase1_gate_weight)
                if not torch.isfinite(losses["loss"]):
                    continue
                losses["loss"].backward()
                mask_non_action_grads(model, action_start, n_bins)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt1.step()
                p1_step += 1
                history.append({"phase": 1, "type": "train", "epoch": epoch, "step": p1_step,
                                 "loss": float(losses["loss"].item()),
                                 "restricted_loss": float(losses["restricted_loss"].item()),
                                 "gate_loss": float(losses["gate_loss"].item())})
            ev = eval_text(model, processor.tokenizer, eval_rows, args.device, action_start, args.max_new_tokens)
            history.append({"phase": 1, "type": "eval", "epoch": epoch + 1,
                             "text_action_rate": ev["text_action_rate"],
                             "token_id_action_rate": ev["token_id_action_rate"]})
            print(f"[P1 eval] epoch={epoch+1}  text_action_rate={ev['text_action_rate']:.3f}  "
                  f"token_id_rate={ev['token_id_action_rate']:.3f}")
            atomic_write_json(metrics_path, {"rows": rows, "history": history})
        print(f"Phase 1 done. Final text_action_rate={ev['text_action_rate']:.3f}")

    # -----------------------------------------------------------------------
    # Phase 2: Multimodal repair
    # -----------------------------------------------------------------------
    print(f"\n=== Phase 2: Multimodal repair ({args.phase2_steps} steps, gate_weight={args.gate_weight}) ===")

    # Pre-encode training batches (heavy — do once before training loop)
    print("[P2] Pre-encoding multimodal training batches (this may take a minute)...")
    mm_batches: List[tuple] = []
    for row in rows:
        batch = encode_multimodal_example(processor, config, row, args.device, args.max_length)
        if batch is not None:
            # Count action token positions to surface data problems early
            labels = batch["labels"]
            act_count = (labels.ne(-100) & labels.ge(action_start) & labels.lt(action_start + n_bins)).sum().item()
            print(f"  idx={row['dataset_index']}  action_token_positions={int(act_count)}  seq_len={labels.shape[1]}")
            mm_batches.append((row, batch))
        else:
            print(f"  idx={row['dataset_index']}  SKIPPED (encode failed)")
    if not mm_batches:
        raise RuntimeError("All multimodal encodes failed. Check data/sensor paths.")
    print(f"[P2] {len(mm_batches)} batches ready.")

    opt2 = torch.optim.AdamW(trainable, lr=args.phase2_lr)
    model.train()

    # Initial multimodal eval before any Phase 2 training
    print("[P2] Initial multimodal generation eval...")
    mm_ev = eval_multimodal(model, processor, config, eval_rows, args.device, action_start, args.max_new_tokens)
    history.append({"phase": 2, "type": "eval", "step": 0,
                     "text_action_rate": mm_ev["text_action_rate"],
                     "token_id_action_rate": mm_ev["token_id_action_rate"],
                     "examples": mm_ev["examples"]})
    print(f"[P2 eval] step=0  text_action_rate={mm_ev['text_action_rate']:.3f}  "
          f"token_id_rate={mm_ev['token_id_action_rate']:.3f}")
    if mm_ev["examples"]:
        print(f"  sample[0] target={mm_ev['examples'][0]['target'][:50]}")
        print(f"  sample[0] output: {mm_ev['examples'][0]['preview'][:120]}")
    atomic_write_json(metrics_path, {"rows": rows, "history": history})

    p2_step = 0
    cycle = 0
    while p2_step < args.phase2_steps:
        model.train()
        for row, batch in mm_batches:
            if p2_step >= args.phase2_steps:
                break
            opt2.zero_grad(set_to_none=True)
            # Detach tensors so we can reuse the pre-encoded batch across steps
            step_batch = {k: v.detach().clone() for k, v in batch.items()}
            losses = compute_action_loss(model, step_batch, action_start, n_bins, args.gate_weight)
            if not torch.isfinite(losses["loss"]):
                print(f"[P2] step={p2_step}  non-finite loss — skipping")
                p2_step += 1
                continue
            losses["loss"].backward()
            mask_non_action_grads(model, action_start, n_bins)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt2.step()
            p2_step += 1

            history.append({
                "phase": 2, "type": "train", "step": p2_step, "cycle": cycle,
                "loss": float(losses["loss"].item()),
                "restricted_loss": float(losses["restricted_loss"].item()),
                "gate_loss": float(losses["gate_loss"].item()),
                "action_count": int(losses["action_count"].item()),
            })

            if p2_step % args.eval_every_steps == 0:
                mm_ev = eval_multimodal(model, processor, config, eval_rows, args.device, action_start, args.max_new_tokens)
                history.append({"phase": 2, "type": "eval", "step": p2_step,
                                 "text_action_rate": mm_ev["text_action_rate"],
                                 "token_id_action_rate": mm_ev["token_id_action_rate"],
                                 "examples": mm_ev["examples"]})
                print(f"[P2 eval] step={p2_step}  text_action_rate={mm_ev['text_action_rate']:.3f}  "
                      f"token_id_rate={mm_ev['token_id_action_rate']:.3f}")
                if mm_ev["examples"]:
                    ex0 = mm_ev["examples"][0]
                    print(f"  target : {ex0['target'][:60]}")
                    print(f"  output : {ex0['preview'][:120]}")
                if args.save_every_eval:
                    snap_path = output_dir / f"checkpoint_step{p2_step:04d}.pt"
                    torch.save({
                        "embedding": model.get_input_embeddings().weight.data.cpu(),
                        "lm_head": model.lm_head.weight.data.cpu(),
                        "action_start": action_start,
                        "n_bins": n_bins,
                        "step": p2_step,
                        "text_action_rate": mm_ev["text_action_rate"],
                    }, snap_path)
                    print(f"  saved → {snap_path}")
                atomic_write_json(metrics_path, {"rows": rows, "history": history})
                model.train()
        cycle += 1

    # Final eval
    mm_ev_final = eval_multimodal(model, processor, config, eval_rows, args.device, action_start, args.max_new_tokens)
    history.append({"phase": 2, "type": "eval", "step": p2_step, "final": True,
                     "text_action_rate": mm_ev_final["text_action_rate"],
                     "token_id_action_rate": mm_ev_final["token_id_action_rate"],
                     "examples": mm_ev_final["examples"]})
    print(f"\n[P2 FINAL] text_action_rate={mm_ev_final['text_action_rate']:.3f}  "
          f"token_id_rate={mm_ev_final['token_id_action_rate']:.3f}")
    for ex in mm_ev_final["examples"]:
        print(f"  [{ex['dataset_index']}] target={ex['target'][:50]} | out: {ex['preview'][:100]}")

    if args.save_checkpoint:
        ckpt_path = output_dir / "repaired_embedding_lmhead.pt"
        torch.save({
            "embedding": model.get_input_embeddings().weight.data.cpu(),
            "lm_head": model.lm_head.weight.data.cpu(),
            "action_start": action_start,
            "n_bins": n_bins,
        }, ckpt_path)
        print(f"Repaired weights saved → {ckpt_path}")

    atomic_write_json(metrics_path, {"rows": rows, "history": history, "status": "complete"})
    print(f"Metrics → {metrics_path}")


if __name__ == "__main__":
    main()
