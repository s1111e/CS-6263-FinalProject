#!/usr/bin/env python3
"""
Table 2 Reproduction Script: Runtime Analysis - Fast vs Slow Thinking

Measures inference time for:
- Fast Thinking: Direct action generation (no reasoning tokens)
- Slow Thinking: Chain-of-thought reasoning + action generation

Output: JSON file with runtime statistics and 9.8x ratio analysis
"""

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import random

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import AutoProcessor

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset_utils.sft_dataset import SFTDataset
from models.autovla import SFTAutoVLA


class RuntimeAnalyzer:
    """Analyze and compare fast vs slow thinking runtime."""
    
    def __init__(self):
        self.fast_times = []
        self.slow_times = []
        self.results = []
    
    def add_measurement(
        self,
        token: str,
        fast_time: float,
        slow_time: float,
        fast_tokens: int = None,
        slow_tokens: int = None
    ) -> None:
        """Record a runtime measurement."""
        self.fast_times.append(fast_time)
        self.slow_times.append(slow_time)
        
        result = {
            "token": token,
            "fast_thinking_time": fast_time,
            "slow_thinking_time": slow_time,
            "ratio": slow_time / fast_time if fast_time > 0 else 0,
        }
        
        if fast_tokens is not None:
            result["fast_tokens"] = fast_tokens
        if slow_tokens is not None:
            result["slow_tokens"] = slow_tokens
        
        self.results.append(result)
    
    def compute_summary(self) -> Dict:
        """Compute aggregate statistics."""
        if not self.fast_times or not self.slow_times:
            raise RuntimeError("No runtime measurements were collected; check model loading and per-sample errors.")

        fast = np.array(self.fast_times)
        slow = np.array(self.slow_times)
        ratios = slow / fast
        
        return {
            "fast_thinking": {
                "min": float(np.min(fast)),
                "max": float(np.max(fast)),
                "mean": float(np.mean(fast)),
                "std": float(np.std(fast)),
                "median": float(np.median(fast)),
                "samples": len(fast),
            },
            "slow_thinking": {
                "min": float(np.min(slow)),
                "max": float(np.max(slow)),
                "mean": float(np.mean(slow)),
                "std": float(np.std(slow)),
                "median": float(np.median(slow)),
                "samples": len(slow),
            },
            "ratio": {
                "min": float(np.min(ratios)),
                "max": float(np.max(ratios)),
                "mean": float(np.mean(ratios)),
                "std": float(np.std(ratios)),
                "median": float(np.median(ratios)),
            }
        }


def load_config(config_path: str) -> Dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Reproduce Table 2: Runtime Analysis - Fast vs Slow Thinking"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/training/qwen2.5-vl-3B-mix-sft.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/sft/2026-04-22_08-18-13/epoch=4-loss=1.7191.ckpt",
        help="Path to SFT checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="autovla-nuscenes-reproduction/evaluation_results/table_2_runtime.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--num_samples",
        "--num-samples",
        dest="num_samples",
        type=int,
        default=500,
        help="Number of samples to profile (default: 500)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (cuda:0, cpu, etc.)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum new tokens for each timed generation call"
    )
    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=1,
        help="Minimum new tokens for each timed generation call"
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding for more stable runtime profiling on older GPUs"
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip the untimed warmup generation"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    
    return parser.parse_args()


def update_results_md(summary: Dict, output_json: str) -> None:
    """Deprecated: results.md is updated by the paper-table collector."""
    print(f"Skipping direct results.md update; collector will read {output_json}.")


def main():
    args = parse_args()
    set_seed(SEED)

    print("=" * 60)
    print("Table 2: Runtime Analysis - Fast vs Slow Thinking")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Num Samples: {args.num_samples}")
    print(f"Output: {args.output}")
    print()

    if args.device.startswith("cuda") and torch.cuda.is_available():
        device_index = int(args.device.split(":")[-1]) if ":" in args.device else 0
        gpu_name = torch.cuda.get_device_name(device_index)
        print(f"GPU: {gpu_name}")
        if "V100" in gpu_name:
            os.environ.setdefault("AUTOVLA_TORCH_DTYPE", "float16")
            os.environ.setdefault("AUTOVLA_ATTN_IMPLEMENTATION", "eager")
            print("V100 detected: using AUTOVLA_TORCH_DTYPE=float16 and AUTOVLA_ATTN_IMPLEMENTATION=eager")
        print()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize processor
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        config['model']['pretrained_model_path'],
        use_fast=True
    )
    
    # Load validation dataset
    print("Loading validation dataset...")
    dataset = SFTDataset(
        config['data']['val'],
        config['model'],
        processor
    )
    print(f"✓ Loaded {len(dataset.scenes)} validation scenes")
    
    # Load model
    print("Loading model checkpoint...")
    model = SFTAutoVLA(config)
    model.autovla.vlm.resize_token_embeddings(len(processor.tokenizer))
    
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.autovla.load_state_dict(state_dict, strict=False)
    model.to(args.device)
    model.autovla.device = args.device
    model.eval()
    model.autovla.gen_conf["min_new_tokens"] = int(args.min_new_tokens)
    if args.greedy:
        model.autovla.gen_conf["do_sample"] = False
    print("✓ Model loaded and ready")
    print(f"Generation override: max_new_tokens={args.max_new_tokens}, min_new_tokens={args.min_new_tokens}, greedy={args.greedy}")
    print()
    
    # Initialize analyzer
    analyzer = RuntimeAnalyzer()
    
    # Determine number of samples
    num_samples = min(args.num_samples, len(dataset.scenes))
    
    print(f"Profiling {num_samples} random samples...")
    print(f"(This will take several minutes)")
    print()
    
    # Profile samples
    failures = []
    with torch.no_grad():
        indices = np.random.choice(len(dataset.scenes), num_samples, replace=False)
        
        for idx in tqdm(indices, desc="Runtime profiling"):
            scene_path, sensor_data_path = dataset.scenes[idx]
            
            # Load scene data
            with open(scene_path, 'r') as f:
                scene_data = json.load(f)
            
            try:
                # Extract features
                input_features = {}
                for builder in dataset._agent.get_feature_builders():
                    input_features.update(builder.compute_features(scene_data))
                input_features['sensor_data_path'] = sensor_data_path

                # Keep generation short enough for profiling on V100-class GPUs.
                # AutoVLA.predict uses max_length, so compute prompt length first
                # and convert the requested max_new_tokens into max_length.
                prompt_inputs = model.autovla.get_prompt(input_features)
                prompt_len = int(prompt_inputs.input_ids.shape[1])
                model.autovla.gen_conf["max_length"] = prompt_len + int(args.max_new_tokens)
                
                # Warm up GPU
                if not args.no_warmup:
                    _ = model.autovla.predict(input_features)
                
                # Time FAST thinking (direct action generation)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                fast_start = time.perf_counter()
                
                # For fast thinking, temporarily disable CoT
                original_use_cot = model.autovla.use_cot
                model.autovla.use_cot = False
                pred_fast, text_fast = model.autovla.predict(input_features)
                model.autovla.use_cot = original_use_cot
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                fast_end = time.perf_counter()
                fast_time = fast_end - fast_start
                
                # Time SLOW thinking (with CoT reasoning)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                slow_start = time.perf_counter()
                
                model.autovla.use_cot = True
                pred_slow, text_slow = model.autovla.predict(input_features)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                slow_end = time.perf_counter()
                slow_time = slow_end - slow_start
                
                # Count tokens
                fast_tokens = len(text_fast.split()) if text_fast else 0
                slow_tokens = len(text_slow.split()) if text_slow else 0
                
                token = scene_data.get('token', f"sample_{idx}")
                analyzer.add_measurement(
                    token=token,
                    fast_time=fast_time,
                    slow_time=slow_time,
                    fast_tokens=fast_tokens,
                    slow_tokens=slow_tokens
                )
                
                if args.verbose:
                    print(f"  Fast: {fast_time:.3f}s ({fast_tokens} tokens)")
                    print(f"  Slow: {slow_time:.3f}s ({slow_tokens} tokens)")
                    print(f"  Ratio: {slow_time/fast_time:.1f}x")
                
            except Exception as e:
                failures.append({
                    "idx": int(idx),
                    "scene_path": str(scene_path),
                    "error": repr(e),
                    "traceback": traceback.format_exc(limit=3),
                })
                if args.verbose or len(failures) <= 3:
                    print(f"Error processing sample {idx} ({scene_path}): {e}")
                    print(failures[-1]["traceback"])
                continue
    
    # Compute summary statistics
    print()
    print(f"Successful samples: {len(analyzer.results)} / {num_samples}")
    print(f"Failed samples: {len(failures)} / {num_samples}")
    if failures:
        print("First failure:")
        print(f"  scene: {failures[0]['scene_path']}")
        print(f"  error: {failures[0]['error']}")
    
    print("Computing summary statistics...")
    summary = analyzer.compute_summary()
    
    # Print results
    print()
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    
    print("\nFast Thinking (Direct Action Generation):")
    print(f"  Min:    {summary['fast_thinking']['min']:.3f}s")
    print(f"  Max:    {summary['fast_thinking']['max']:.3f}s")
    print(f"  Mean:   {summary['fast_thinking']['mean']:.3f}s (± {summary['fast_thinking']['std']:.3f})")
    print(f"  Median: {summary['fast_thinking']['median']:.3f}s")
    
    print("\nSlow Thinking (CoT + Action Generation):")
    print(f"  Min:    {summary['slow_thinking']['min']:.3f}s")
    print(f"  Max:    {summary['slow_thinking']['max']:.3f}s")
    print(f"  Mean:   {summary['slow_thinking']['mean']:.3f}s (± {summary['slow_thinking']['std']:.3f})")
    print(f"  Median: {summary['slow_thinking']['median']:.3f}s")
    
    print("\nSpeedup Ratio (Slow / Fast):")
    print(f"  Min:    {summary['ratio']['min']:.1f}x")
    print(f"  Max:    {summary['ratio']['max']:.1f}x")
    print(f"  Mean:   {summary['ratio']['mean']:.1f}x (± {summary['ratio']['std']:.1f})")
    print(f"  Median: {summary['ratio']['median']:.1f}x")
    
    print(f"\nSamples profiled: {summary['fast_thinking']['samples']}")
    print("=" * 60)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results_dict = {
        "num_samples": len(analyzer.results),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "runtime_units": "seconds",
        "summary": summary,
        "per_sample": analyzer.results,
        "failures": failures[:20],
    }
    
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"✓ Results saved to {output_path}")
    
    # Update results.md with findings
    update_results_md(summary, str(output_path))
    
    print()


if __name__ == "__main__":
    main()
