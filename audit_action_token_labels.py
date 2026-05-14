#!/usr/bin/env python3
"""Audit whether <action_*> tokens reach the supervised labels.

This diagnostic checks the dataset/collator path before training:
- raw rendered prompt contains action tokens
- tokenizer maps them to ids >= action_start_id
- labels keep those ids after assistant masking
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from transformers import AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "navsim"))

from dataset_utils.sft_dataset import DataCollator, SFTDataset


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def count_action_ids(tensor: torch.Tensor, action_start_id: int) -> int:
    return int((tensor >= action_start_id).sum().item())


def decode_label_tokens(processor: AutoProcessor, labels: torch.Tensor, ignore_index: int) -> str:
    visible = labels[labels != ignore_index]
    if visible.numel() == 0:
        return ""
    return processor.tokenizer.decode(visible, skip_special_tokens=False)


def audit_sample(
    dataset: SFTDataset,
    collator: DataCollator,
    processor: AutoProcessor,
    idx: int,
    action_start_id: int,
    ignore_index: int,
) -> Dict:
    item = dataset[idx]
    batch = collator([item])

    text = item["text"]
    raw_matches = re.findall(r"<action_(\d+)>", text)
    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]
    label_action_mask = labels >= action_start_id
    input_action_mask = input_ids >= action_start_id

    label_action_ids = labels[label_action_mask].tolist()
    input_action_ids = input_ids[input_action_mask].tolist()
    decoded_labels = decode_label_tokens(processor, labels, ignore_index)
    decoded_actions = [
        processor.tokenizer.decode(torch.tensor([token_id]), skip_special_tokens=False)
        for token_id in label_action_ids[:20]
    ]

    return {
        "dataset_index": idx,
        "data_path": str(item.get("data_path", "")),
        "has_cot": bool(item.get("has_cot", False)),
        "raw_text_action_count": len(raw_matches),
        "raw_text_first_actions": raw_matches[:20],
        "input_ids_action_count": count_action_ids(input_ids, action_start_id),
        "labels_action_count": count_action_ids(labels, action_start_id),
        "input_ids_first_action_ids": input_action_ids[:20],
        "labels_first_action_ids": label_action_ids[:20],
        "labels_first_action_tokens": decoded_actions,
        "label_visible_token_count": int((labels != ignore_index).sum().item()),
        "input_token_count": int(input_ids.numel()),
        "decoded_label_preview": decoded_labels[-800:],
        "raw_text_preview": text[-800:],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/training/qwen2.5-vl-3B-mix-sft.yaml")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--output-json",
        default="autovla-nuscenes-reproduction/evaluation_results/action_token_label_audit.json",
    )
    parser.add_argument(
        "--output-md",
        default="autovla-nuscenes-reproduction/evaluation_results/action_token_label_audit.md",
    )
    return parser.parse_args()


def write_markdown(path: Path, summary: Dict) -> None:
    lines = [
        "# Action-Token Label Audit",
        "",
        f"Config: `{summary['config']}`",
        f"Split: `{summary['split']}`",
        f"Action start id: `{summary['action_start_id']}`",
        "",
        "| Sample | Raw `<action_*>` | Input action ids | Label action ids | Visible label tokens |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary["samples"]:
        lines.append(
            f"| {row['dataset_index']} | {row['raw_text_action_count']} | "
            f"{row['input_ids_action_count']} | {row['labels_action_count']} | "
            f"{row['label_visible_token_count']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If `Label action ids` is zero, the supervised loss cannot teach direct action-token generation.",
            "If it is non-zero, action tokens reach the labels and the problem is likely model/training/generation strength rather than collator masking.",
            "",
            "## First Sample Preview",
            "",
            "```text",
            summary["samples"][0]["decoded_label_preview"] if summary["samples"] else "",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    processor = AutoProcessor.from_pretrained(config["model"]["pretrained_model_path"], use_fast=True)
    dataset = SFTDataset(
        config["data"][args.split],
        config["model"],
        processor,
        using_cot=config["model"]["use_cot"],
    )
    collator = DataCollator(
        processor=processor,
        ignore_index=config["model"]["tokens"]["ignore_index"],
        assistant_id=config["model"]["tokens"]["assistant_id"],
    )

    action_start_id = int(config["model"]["tokens"]["action_start_id"])
    ignore_index = int(config["model"]["tokens"]["ignore_index"])
    samples = []
    for idx in range(args.start_index, min(args.start_index + args.num_samples, len(dataset))):
        samples.append(audit_sample(dataset, collator, processor, idx, action_start_id, ignore_index))

    summary = {
        "config": args.config,
        "split": args.split,
        "action_start_id": action_start_id,
        "num_samples": len(samples),
        "samples": samples,
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(output_md, summary)

    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")
    print()
    for row in samples:
        print(
            f"sample={row['dataset_index']} raw={row['raw_text_action_count']} "
            f"input_ids={row['input_ids_action_count']} labels={row['labels_action_count']} "
            f"visible_labels={row['label_visible_token_count']}"
        )


if __name__ == "__main__":
    main()
