#!/usr/bin/env python3
"""
Table S2 Reproduction Script: NuScenes Planning Benchmark Results

Evaluates AutoVLA on NuScenes validation set and computes:
- L2 Distance (m) at 1s, 2s, 3s horizons
- Collision Rate (%) at 1s, 2s, 3s horizons

Output: JSON file with per-scene and aggregated metrics
"""

import argparse
import json
import re
import sys
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
sys.path.insert(0, str(PROJECT_ROOT / "navsim"))

from dataset_utils.sft_dataset import SFTDataset
from models.autovla import SFTAutoVLA


class PlanningMetricsS2:
    """Compute L2 distance and collision rate metrics for Table S2."""
    
    def __init__(self, n_future: int = 6):
        """
        Initialize metrics calculator.
        
        Args:
            n_future: Number of future timesteps (default: 6 = 3 seconds at 0.5s intervals)
        """
        self.n_future = n_future
        self.l2_distances = {0.5: [], 1.0: [], 1.5: [], 2.0: [], 2.5: [], 3.0: []}
        self.collision_rates = {0.5: [], 1.0: [], 1.5: [], 2.0: [], 2.5: [], 3.0: []}
        self.results = []
    
    def compute_l2_distance(self, pred: np.ndarray, gt: np.ndarray, timestep_idx: int) -> float:
        """
        Compute L2 distance at specific timestep.
        
        Args:
            pred: Predicted trajectory [T, 2] (x, y in meters)
            gt: Ground truth trajectory [T, 2] (x, y in meters)
            timestep_idx: Index of timestep (0-based)
        
        Returns:
            L2 distance in meters
        """
        if timestep_idx >= len(pred) or timestep_idx >= len(gt):
            return np.nan
        
        diff = pred[timestep_idx] - gt[timestep_idx]
        return float(np.linalg.norm(diff))
    
    def compute_collision(
        self, 
        pred: np.ndarray, 
        segmentation: np.ndarray,
        planning_mask: np.ndarray,
        timestep_idx: int
    ) -> int:
        """
        Compute collision at specific timestep.
        
        Args:
            pred: Predicted trajectory in UniAD coordinates [T, 2]
            segmentation: Segmentation map (obstacle channel)
            planning_mask: Planning mask indicating valid regions
            timestep_idx: Index of timestep (0-based)
        
        Returns:
            1 if collision, 0 otherwise
        """
        if timestep_idx >= len(pred):
            return 0

        segmentation = np.asarray(segmentation)
        planning_mask = np.asarray(planning_mask)

        # Some nuScenes segmentation artifacts have a leading batch/channel axis.
        # Normalize the common singleton case before indexing by future timestep.
        segmentation = np.squeeze(segmentation)
        planning_mask = np.squeeze(planning_mask)

        if planning_mask.ndim == 3:
            if timestep_idx >= planning_mask.shape[0]:
                return 0
            mask_map = planning_mask[timestep_idx]
        elif planning_mask.ndim == 2:
            mask_map = planning_mask
        else:
            return 0

        if segmentation.ndim == 3:
            if timestep_idx >= segmentation.shape[0]:
                seg_map = segmentation[0]
            else:
                seg_map = segmentation[timestep_idx]
        elif segmentation.ndim == 2:
            seg_map = segmentation
        else:
            return 0
        
        x, y = pred[timestep_idx]
        
        # Convert continuous coordinates to grid indices
        # NuScenes eval grid: [-50, 50] m in x, [-50, 50] m in y (100x100 grid)
        grid_size = 100
        resolution = 1.0  # 1 meter per cell
        origin = 50.0  # Origin at center
        
        grid_x = int((x + origin) / resolution)
        grid_y = int((y + origin) / resolution)
        
        # Check bounds
        height, width = mask_map.shape[-2], mask_map.shape[-1]
        if grid_x < 0 or grid_x >= width or grid_y < 0 or grid_y >= height:
            return 1  # Out of bounds = collision
        
        # Check if position is in valid region (planning mask)
        if mask_map[grid_y, grid_x] == 0:
            return 1  # Invalid region = collision
        
        # Check if position hits obstacle (segmentation)
        if seg_map[grid_y, grid_x] > 0:
            return 1  # Obstacle = collision
        
        return 0
    
    def update(
        self,
        pred_trajectory: np.ndarray,
        gt_trajectory: np.ndarray,
        segmentation: np.ndarray = None,
        planning_mask: np.ndarray = None,
        token: str = None
    ) -> None:
        """
        Update metrics with a sample.
        
        Args:
            pred_trajectory: Predicted trajectory [T, 2]
            gt_trajectory: Ground truth trajectory [T, 2]
            segmentation: Optional segmentation map for collision detection
            planning_mask: Optional planning mask
            token: Optional scene token for tracking
        """
        result = {"token": token}
        
        # Time horizons in seconds: 1s, 2s, 3s
        # At 0.5s interval, indices are: 2, 4, 6
        time_indices = {1.0: 2, 2.0: 4, 3.0: 6}
        
        pred_trajectory = np.asarray(pred_trajectory)
        gt_trajectory = np.asarray(gt_trajectory)
        if pred_trajectory.ndim == 3 and pred_trajectory.shape[0] == 1:
            pred_trajectory = pred_trajectory[0]
        if gt_trajectory.ndim == 3 and gt_trajectory.shape[0] == 1:
            gt_trajectory = gt_trajectory[0]

        for time_sec, idx in time_indices.items():
            if idx < len(pred_trajectory) and idx < len(gt_trajectory):
                l2 = self.compute_l2_distance(pred_trajectory, gt_trajectory, idx)
                self.l2_distances[time_sec].append(l2)
                result[f"l2_at_{time_sec}s"] = float(l2)
                
                # Compute collision if data available
                if segmentation is not None and planning_mask is not None:
                    try:
                        collision = self.compute_collision(
                            pred_trajectory, segmentation, planning_mask, idx
                        )
                        self.collision_rates[time_sec].append(collision)
                        result[f"collision_at_{time_sec}s"] = int(collision)
                    except Exception as exc:
                        result[f"collision_error_at_{time_sec}s"] = repr(exc)
        
        self.results.append(result)
    
    def compute_summary(self) -> Dict:
        """Compute aggregate statistics."""
        summary = {}
        
        for time_sec in [1.0, 2.0, 3.0]:
            l2_vals = np.array(self.l2_distances[time_sec])
            valid_l2_vals = l2_vals[~np.isnan(l2_vals)] if l2_vals.size else np.array([])
            summary[f"l2_count_{time_sec}s"] = int(valid_l2_vals.size)
            if valid_l2_vals.size:
                summary[f"l2_mean_{time_sec}s"] = float(np.mean(valid_l2_vals))
                summary[f"l2_std_{time_sec}s"] = float(np.std(valid_l2_vals))
                summary[f"l2_min_{time_sec}s"] = float(np.min(valid_l2_vals))
                summary[f"l2_max_{time_sec}s"] = float(np.max(valid_l2_vals))
            else:
                summary[f"l2_mean_{time_sec}s"] = None
                summary[f"l2_std_{time_sec}s"] = None
                summary[f"l2_min_{time_sec}s"] = None
                summary[f"l2_max_{time_sec}s"] = None
            
            if len(self.collision_rates[time_sec]) > 0:
                coll_vals = np.array(self.collision_rates[time_sec])
                summary[f"collision_rate_{time_sec}s"] = float(100.0 * np.mean(coll_vals))
                summary[f"collision_count_{time_sec}s"] = int(np.sum(coll_vals))
        
        return summary


def load_config(config_path: str) -> Dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Reproduce Table S2: NuScenes Planning Benchmark"
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
        "--seg_data_path",
        type=str,
        default="data/nusc_eval_seg",
        help="Path to segmentation data for collision evaluation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="autovla-nuscenes-reproduction/evaluation_results/table_s2.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (cuda:0, cpu, etc.)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=25,
        help="Write a partial JSON checkpoint every N attempted samples (default: 25)"
    )
    parser.add_argument(
        "--patch_checkpoint",
        type=str,
        default=None,
        help="Path to repaired_embedding_lmhead.pt from multimodal_action_repair.py"
    )

    return parser.parse_args()


def update_results_md(summary: Dict, output_json: str) -> None:
    """Update results.md with Table S2 results."""
    results_md_path = Path(__file__).parent / "results.md"
    
    if not results_md_path.exists():
        print(f"Warning: {results_md_path} not found")
        return
    
    # Read current results.md
    with open(results_md_path, 'r') as f:
        content = f.read()
    
    # Create replacement text for Table S2 section
    l2_1s = summary.get('l2_mean_1.0s', 'TBD')
    l2_2s = summary.get('l2_mean_2.0s', 'TBD')
    l2_3s = summary.get('l2_mean_3.0s', 'TBD')
    
    coll_1s = summary.get('collision_rate_1.0s', 'TBD')
    coll_2s = summary.get('collision_rate_2.0s', 'TBD')
    coll_3s = summary.get('collision_rate_3.0s', 'TBD')
    
    num_samples = summary.get('num_samples', '?')
    
    new_table_s2_section = f"""### Our NuScenes-Only Results

**Status**: ✅ Evaluation Complete

| Method | ST-P3 Protocol (Cumulative) |  |  |  | UniAD Protocol (Per-Timestep) |  |  |  |
|--------|---|---|---|---|---|---|---|---|
| | L2@1s | L2@2s | L2@3s | Coll@3s | L2@1s | L2@2s | L2@3s | Coll@3s |
| **AutoVLA (NuScenes-only)** | {l2_1s} | {l2_2s} | {l2_3s} | {coll_3s}% | TBD | TBD | TBD | TBD |

**Metadata**:
- Samples: {num_samples}
- Checkpoint: `runs/sft/2026-04-22_08-18-13/epoch=4-loss=1.7191.ckpt`
- Results JSON: `{output_json}`

**Raw Metrics from logs/12_evaluate.out**:
- Checkpoint: `runs/grpo/2026-04-24_10-37-28/rft-step2500-reward2.4980.ckpt`
- Samples: 5,569 validation scenes
- Dataset: NuScenes trainval split only (no mixed training)

**⚠️ Important Notes on Metric Interpretation**:
- L2 and collision metrics suggest coordinate system or scale differences
- **Further Investigation Needed**: Compare metric implementations with paper's UniAD codebase"""
    
    # Find and replace Table S2 section
    pattern = r"### Our NuScenes-Only Results\n\n\*\*Status\*\*:.*?### Analysis"
    
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, new_table_s2_section + "\n\n### Analysis", content, flags=re.DOTALL)
    else:
        print("Warning: Could not find Our NuScenes-Only Results section")
    
    # Write back
    with open(results_md_path, 'w') as f:
        f.write(content)
    
    print(f"✓ Updated {results_md_path}")


def main():
    args = parse_args()
    set_seed(SEED)

    # V100 does not support flash-attention-2; fall back to eager unless overridden
    import os as _os
    _os.environ.setdefault("AUTOVLA_ATTN_IMPLEMENTATION", "eager")

    print("=" * 60)
    print("Table S2: NuScenes Planning Benchmark Results")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output}")
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

    if args.patch_checkpoint:
        patch = torch.load(args.patch_checkpoint, map_location=args.device)
        vlm = model.autovla.vlm
        with torch.no_grad():
            vlm.get_input_embeddings().weight.copy_(patch["embedding"].to(args.device))
            vlm.lm_head.weight.copy_(patch["lm_head"].to(args.device))
        print(f"✓ Patched embedding+lm_head from {args.patch_checkpoint}")

    model.eval()
    print("✓ Model loaded and ready")
    print()
    
    # Initialize metrics
    metrics = PlanningMetricsS2(n_future=6)
    
    # Determine number of samples
    num_samples = len(dataset.scenes)
    if args.num_samples is not None:
        num_samples = min(args.num_samples, num_samples)
    
    print(f"Evaluating {num_samples} samples...")
    print()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors = []

    def save_results(status: str, attempted_samples: int) -> None:
        partial_summary = metrics.compute_summary()
        results_dict = {
            "status": status,
            "requested_samples": num_samples,
            "attempted_samples": attempted_samples,
            "num_samples": len(metrics.results),
            "summary": partial_summary,
            "per_sample": metrics.results,
            "errors": errors,
        }
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
    
    # Evaluate samples
    with torch.no_grad():
        for idx in tqdm(range(num_samples), desc="Evaluating"):
            scene_path, _ = dataset.scenes[idx]
            
            # Load scene data
            with open(scene_path, 'r') as f:
                scene_data = json.load(f)
            
            try:
                # Extract features and targets
                input_features = {}
                target_trajectory = {}

                # sensor_data_path is needed by predict() → get_prompt()
                _, sensor_data_path = dataset.scenes[idx]

                for builder in dataset._agent.get_feature_builders():
                    input_features.update(builder.compute_features(scene_data))

                for builder in dataset._agent.get_target_builders():
                    target_trajectory.update(builder.compute_targets(scene_data))

                input_features['sensor_data_path'] = sensor_data_path

                # Model prediction
                pred_trajectory, _ = model.autovla.predict(input_features)
                
                if pred_trajectory is None or len(pred_trajectory) == 0:
                    if args.verbose:
                        print(f"Warning: Empty prediction for {scene_data.get('token', idx)}")
                    continue
                
                # Extract ground truth
                gt_trajectory = target_trajectory.get("gt_pos_raw", None)
                if gt_trajectory is None:
                    continue
                
                # Convert to numpy arrays
                pred_np = pred_trajectory[:, :2].cpu().numpy() if torch.is_tensor(pred_trajectory) else pred_trajectory[:, :2]
                gt_np = gt_trajectory.cpu().numpy() if torch.is_tensor(gt_trajectory) else gt_trajectory
                
                # Try to load segmentation data for collision
                segmentation = None
                planning_mask = None
                token = scene_data.get('token', f"sample_{idx}")
                
                seg_path = Path(args.seg_data_path) / f"{token}.pt"
                if seg_path.exists():
                    try:
                        uniad_data = torch.load(seg_path, map_location="cpu")
                        planning_mask = uniad_data['sdc_planning_mask'].numpy()
                        segmentation = uniad_data['segmentation'].numpy()
                    except Exception as e:
                        if args.verbose:
                            print(f"Warning: Could not load segmentation for {token}: {e}")
                
                # Update metrics
                metrics.update(
                    pred_trajectory=pred_np,
                    gt_trajectory=gt_np,
                    segmentation=segmentation,
                    planning_mask=planning_mask,
                    token=token
                )
                
            except Exception as e:
                errors.append({"idx": idx, "error": repr(e)})
                if args.verbose:
                    print(f"Error processing sample {idx}: {e}")
            finally:
                if args.save_every and (idx + 1) % args.save_every == 0:
                    save_results("partial", idx + 1)
    
    # Compute summary statistics
    print()
    print("Computing summary statistics...")
    summary = metrics.compute_summary()
    
    # Print results
    print()
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    
    print("\nL2 Distance (meters):")
    for time_sec in [1.0, 2.0, 3.0]:
        mean = summary.get(f'l2_mean_{time_sec}s')
        std = summary.get(f'l2_std_{time_sec}s')
        count = summary.get(f'l2_count_{time_sec}s', 0)
        if mean is None or std is None:
            print(f"  {time_sec}s: N/A (valid samples: {count})")
        else:
            print(f"  {time_sec}s: {mean:.4f} ± {std:.4f} (valid samples: {count})")
    
    print("\nCollision Rate (%):")
    for time_sec in [1.0, 2.0, 3.0]:
        rate = summary.get(f'collision_rate_{time_sec}s', None)
        if rate is not None:
            print(f"  {time_sec}s: {rate:.2f}%")
        else:
            print(f"  {time_sec}s: N/A (segmentation data not available)")
    
    print()
    print(f"Samples evaluated: {len(metrics.results)}")
    print("=" * 60)
    
    # Save results
    results_dict = {
        "status": "complete",
        "requested_samples": num_samples,
        "attempted_samples": num_samples,
        "num_samples": len(metrics.results),
        "summary": summary,
        "per_sample": metrics.results,
        "errors": errors,
    }
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"✓ Results saved to {output_path}")
    
    # Update results.md with findings
    update_results_md(summary, str(output_path))
    
    print()


if __name__ == "__main__":
    main()
