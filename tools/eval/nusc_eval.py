"""
NuScenes Evaluation Script for AutoVLA.

This script evaluates the AutoVLA model on NuScenes validation data,
computing planning metrics such as L2 distance and collision rate.
"""
import argparse
import json
import sys
import time
from pathlib import Path

# Add project root and navsim to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "navsim"))

import yaml
import torch
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable
from transformers import AutoProcessor

from dataset_utils.sft_dataset import SFTDataset
from models.autovla import SFTAutoVLA
from tools.eval.planning_metrics import PlanningMetric



def load_config(file_path):
    """Load configuration from YAML file."""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate AutoVLA on NuScenes validation data")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint")
    parser.add_argument("--seg_data_path", type=str, required=True,
                        help="Path to the segmentation data directory for evaluation")
    parser.add_argument("--output", type=str, default="planning_table.txt",
                        help="Output file for results (default: planning_table.txt)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use (default: cuda:0)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to evaluate (default: all)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose output for each sample")
    parser.add_argument("--patch_checkpoint", type=str, default=None,
                        help="Path to repaired embedding+lm_head .pt file from multimodal_action_repair.py")
    return parser.parse_args()


def load_eval_checkpoint(checkpoint_file):
    """Load processed sample tokens from checkpoint file."""
    if Path(checkpoint_file).exists():
        with open(checkpoint_file, 'r') as f:
            return set(json.load(f).get('processed_tokens', []))
    return set()


def save_eval_checkpoint(checkpoint_file, processed_tokens):
    """Save processed sample tokens to checkpoint file."""
    with open(checkpoint_file, 'w') as f:
        json.dump({'processed_tokens': list(processed_tokens)}, f)


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize processor
    processor = AutoProcessor.from_pretrained(config['model']['pretrained_model_path'], use_fast=True)
    
    # Build data config for SFTDataset from config
    
    train_dataset = SFTDataset(config['data']['val'], config['model'], processor)

    # Load model
    checkpoint_path = Path(args.checkpoint)
    model = SFTAutoVLA(config)
    
    state_dict = torch.load(checkpoint_path, map_location=args.device)['state_dict']
    # Remove 'autovla.' prefix if it exists (for RFT checkpoints)
    state_dict = {k.replace('autovla.', ''): v for k, v in state_dict.items()}
    model.autovla.load_state_dict(state_dict, strict=False)

    # Move to device with float16 for memory efficiency
    model.to(args.device)
    model.autovla.device = args.device  # Update the device attribute for predict()
    model.eval()

    # Apply repaired embedding + lm_head if provided
    if args.patch_checkpoint:
        patch = torch.load(args.patch_checkpoint, map_location=args.device)
        vlm = model.autovla.vlm
        with torch.no_grad():
            vlm.get_input_embeddings().weight.copy_(patch["embedding"].to(args.device))
            vlm.lm_head.weight.copy_(patch["lm_head"].to(args.device))
        print(f"Patched embedding+lm_head from {args.patch_checkpoint}")

    # Set model to float16 to reduce memory usage during inference
    model.half()

    # Initialize planning metrics
    planning_metrics = PlanningMetric(n_future=6)
    
    # Determine number of samples to evaluate
    sample_num = len(train_dataset.scenes)
    if args.num_samples is not None:
        sample_num = min(args.num_samples, sample_num)
    
    print(f"Evaluating {sample_num} samples...")

    # Resume checkpoint for restarting from interruption
    checkpoint_file = Path("eval_checkpoint.json")
    processed_tokens = load_eval_checkpoint(checkpoint_file)
    print(f"Resuming from checkpoint: {len(processed_tokens)} samples already processed")

    # Pre-cache segmentation files to avoid repeated disk I/O
    seg_cache = {}
    if Path(args.seg_data_path).exists():
        seg_files = list(Path(args.seg_data_path).glob("*.pt"))
        print(f"Pre-loading {len(seg_files)} segmentation files into memory...")
        for seg_file in seg_files:
            token = seg_file.stem
            try:
                seg_cache[token] = torch.load(seg_file, map_location="cpu")
            except Exception as e:
                print(f"Warning: Failed to load {seg_file}: {e}")
        print(f"✓ Cached {len(seg_cache)} segmentation files")

    # Evaluate each sample
    for idx in tqdm(range(sample_num), desc="Processing samples"):
        # scenes is a list of tuples: (scene_path, sensor_data_path)
        scene_path, _ = train_dataset.scenes[idx]
        
        # Load scene data
        with open(scene_path, 'r') as f:
            scene_data = json.load(f)
        
        token = scene_data['token']
        
        # Skip if already processed
        if token in processed_tokens:
            if idx < 3:
                print(f"DEBUG[{idx}] SKIPPED: Already processed (resume from checkpoint)")
            continue
        
        # Get features and targets
        input_features = {}
        target_trajectory = {}
        
        start_time = time.time()

        for builder in train_dataset._agent.get_feature_builders():
            input_features.update(builder.compute_features(scene_data))
        for builder in train_dataset._agent.get_target_builders():
            target_trajectory.update(builder.compute_targets(scene_data))

        # Model prediction
        pred_trajectory, output_text = model.autovla.predict(input_features)
        if idx < 3:
            print(f"DEBUG[{idx}] Prediction result: type={type(pred_trajectory)}, pred={pred_trajectory if not isinstance(pred_trajectory, torch.Tensor) else f'tensor shape={pred_trajectory.shape}'}")
        if pred_trajectory == [] or len(pred_trajectory) == 0:
            if idx < 3:
                print(f"DEBUG[{idx}] SKIPPED: Empty prediction")
            continue
        
        if args.verbose:
            print(f"Output: {output_text}")
            print(f"Predicted trajectory: {pred_trajectory}")
            print(f"Time taken: {time.time() - start_time:.3f} seconds")
       
        # Get ground truth trajectory
        gt_raw_trajectory = target_trajectory["gt_pos_raw"]
        pred_xy = pred_trajectory[:, :2].to(gt_raw_trajectory.device)

        # Load segmentation data for collision evaluation
        token = scene_data['token']
        if token not in seg_cache:
            print(f"Warning: Segmentation file not found for token: {token}")
            continue
        
        uniad_data = seg_cache[token]
        sdc_planning_mask = uniad_data['sdc_planning_mask'].to(gt_raw_trajectory.dtype)
        segmentation = uniad_data['segmentation'].to(gt_raw_trajectory.dtype)

        # Transform GT trajectory to UniAD coordinate system
        gt_traj_uniadcoord = gt_raw_trajectory.unsqueeze(0).clone()
        gt_traj_uniadcoord[:, :, [0, 1]] = gt_traj_uniadcoord[:, :, [1, 0]]
        gt_traj_uniadcoord[:, :, 0] = -gt_traj_uniadcoord[:, :, 0]
        gt_traj_uniadcoord = gt_traj_uniadcoord.unsqueeze(0)

        # Transform predicted trajectory to UniAD coordinate system
        pred_traj_uniadcoord = pred_xy.unsqueeze(0).clone()
        pred_traj_uniadcoord[:, :, [0, 1]] = pred_traj_uniadcoord[:, :, [1, 0]]
        pred_traj_uniadcoord[:, :, 0] = -pred_traj_uniadcoord[:, :, 0]
        pred_traj_uniadcoord = pred_traj_uniadcoord.unsqueeze(0)

        # Validate future mask consistency
        cache_future_mask = torch.tensor(scene_data['future_mask'][:6])
        sdc_mask = sdc_planning_mask[0, 0, :, 0]
        if not torch.allclose(cache_future_mask, sdc_mask):
            # DEBUG: print mismatch details for first few samples
            if idx < 3:
                print(f"DEBUG[{idx}] Mask mismatch: cache={cache_future_mask.tolist()}, sdc={sdc_mask.tolist()}")
            continue

        # DEBUG: Track successfully processed samples
        if idx < 3:
            print(f"DEBUG[{idx}] Processing: pred_shape={pred_traj_uniadcoord.shape}, gt_shape={gt_traj_uniadcoord.shape}")

        # Compute planning metrics
        planning_metrics(
            pred_traj_uniadcoord[0, :, :6, :], 
            gt_traj_uniadcoord[0, :, :6, :], 
            sdc_planning_mask[0, :, :6, :2], 
            segmentation[:, [1, 2, 3, 4, 5, 6]]
        )
        
        if idx < 3:
            print(f"DEBUG[{idx}] Metric computed successfully")
        
        # Mark this sample as processed
        processed_tokens.add(token)
        
        # Periodically save checkpoint (every 100 samples)
        if (idx + 1) % 100 == 0:
            save_eval_checkpoint(checkpoint_file, processed_tokens)
    
    # Calculate and print overall statistics
    eval_result = planning_metrics.compute()
    
    # Save final checkpoint as evaluation is complete
    save_eval_checkpoint(checkpoint_file, processed_tokens)
    print(f"✓ Evaluation checkpoint saved: {len(processed_tokens)} samples processed")
    
    # Create table with STP3's definition (cumulative average)
    planning_tab_stp3 = PrettyTable()
    planning_tab_stp3.title = "STP3's Definition Planning Metrics (Cumulative Average)"
    planning_tab_stp3.field_names = ["metrics", "0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s"]
    
    for key, value in eval_result.items():
        row_value = [key]
        for i in range(min(len(value), 6)):
            row_value.append("%.4f" % float(value[:i + 1].mean()))
        planning_tab_stp3.add_row(row_value)
    print(planning_tab_stp3)

    # Create table with UniAD's definition (per-timestep)
    planning_tab_uniad = PrettyTable()
    planning_tab_uniad.title = "UniAD's Definition Planning Metrics (Per-Timestep)"
    planning_tab_uniad.field_names = ["metrics", "0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s"]
    
    for key, value in eval_result.items():
        row_value = [key]
        for i in range(min(len(value), 6)):
            row_value.append("%.4f" % float(value[i]))
        planning_tab_uniad.add_row(row_value)
    print(planning_tab_uniad)

    # Save results to file
    with open(args.output, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Evaluation Results - {sample_num} samples\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"{'='*60}\n\n")
        f.write(str(planning_tab_stp3) + "\n\n")
        f.write(str(planning_tab_uniad) + "\n")
    
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
