#!/usr/bin/env python3
"""Audit AutoVLA action-token tokenizer mapping.

Checks whether <action_i> tokens are single tokens, whether their token ids match
configured action_start_id + i, and whether model embedding/lm_head sizes match
the tokenizer after adding action tokens.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.action_tokenizer import ActionTokenizer


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def inspect_token(tokenizer, token: str, expected_id: int) -> Dict:
    ids = tokenizer(token, add_special_tokens=False).input_ids
    converted_id = tokenizer.convert_tokens_to_ids(token)
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    return {
        "token": token,
        "expected_id": expected_id,
        "convert_tokens_to_ids": converted_id,
        "encoded_ids": ids,
        "encoded_len": len(ids),
        "decoded": decoded,
        "single_token": len(ids) == 1,
        "id_matches_expected": len(ids) == 1 and ids[0] == expected_id and converted_id == expected_id,
        "roundtrip_matches": decoded == token,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/training/qwen2.5-vl-3B-mix-sft.yaml")
    parser.add_argument("--checkpoint", default="runs/sft/2026-04-22_08-18-13/epoch=4-loss=1.7191.ckpt")
    parser.add_argument("--output-json", default="autovla-nuscenes-reproduction/evaluation_results/action_token_mapping_audit.json")
    parser.add_argument("--sample-ids", default="0,1,2,10,100,511,1024,2047")
    args = parser.parse_args()

    config = load_config(args.config)
    processor = AutoProcessor.from_pretrained(config["model"]["pretrained_model_path"], use_fast=True)
    before_vocab = len(processor.tokenizer)
    action_tokenizer = ActionTokenizer(processor.tokenizer, model_config=config["model"])
    after_vocab = len(processor.tokenizer)
    action_start_id = int(config["model"]["tokens"]["action_start_id"])
    n_bins = int(action_tokenizer.n_bins)

    sample_ids = [int(x.strip()) for x in args.sample_ids.split(",") if x.strip()]
    rows: List[Dict] = []
    for i in sample_ids:
        rows.append(inspect_token(processor.tokenizer, f"<action_{i}>", action_start_id + i))

    all_checked_ok = all(row["id_matches_expected"] and row["roundtrip_matches"] for row in rows)

    model_info = {}
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config["model"]["pretrained_model_path"], torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        original_embed_shape = tuple(model.get_input_embeddings().weight.shape)
        model.resize_token_embeddings(len(processor.tokenizer))
        resized_embed_shape = tuple(model.get_input_embeddings().weight.shape)
        lm_head_shape = tuple(model.lm_head.weight.shape)
        model_info = {
            "original_embed_shape": original_embed_shape,
            "resized_embed_shape": resized_embed_shape,
            "lm_head_shape": lm_head_shape,
            "tokenizer_len_matches_embedding": resized_embed_shape[0] == len(processor.tokenizer),
            "tokenizer_len_matches_lm_head": lm_head_shape[0] == len(processor.tokenizer),
        }
    except Exception as exc:
        model_info = {"error": repr(exc)}

    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "pretrained_model_path": config["model"]["pretrained_model_path"],
        "before_vocab_len": before_vocab,
        "after_vocab_len": after_vocab,
        "action_start_id": action_start_id,
        "n_bins": n_bins,
        "expected_after_vocab_min": action_start_id + n_bins,
        "all_checked_ok": all_checked_ok,
        "rows": rows,
        "model_info": model_info,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"before_vocab_len={before_vocab}")
    print(f"after_vocab_len={after_vocab}")
    print(f"action_start_id={action_start_id}")
    print(f"n_bins={n_bins}")
    print(f"all_checked_ok={all_checked_ok}")
    print("token checks:")
    for row in rows:
        print(
            f"  {row['token']}: encoded={row['encoded_ids']} expected={row['expected_id']} "
            f"single={row['single_token']} id_ok={row['id_matches_expected']} roundtrip={row['roundtrip_matches']}"
        )
    print(f"model_info={model_info}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
