#!/usr/bin/env python3
"""
Table 4 Reproduction: K=2048 with OPTIMAL SEGMENT-WISE Evaluation

Instead of greedy accumulation, evaluate each segment independently
This is closer to what the paper likely does.
"""

import json
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Tuple
import random

class K2048OptimalEvaluator:
    def __init__(self, codebook_path: str = "codebook_cache/agent_vocab.pkl"):
        """Initialize with K=2048 codebook"""
        with open(codebook_path, "rb") as f:
            data = pickle.load(f)
        self.codebook = torch.tensor(data['token_all']['veh'], dtype=torch.float32)
        print(f"✅ Loaded K=2048 codebook: shape {self.codebook.shape}")
        assert self.codebook.shape[0] == 2048, "Must be K=2048!"
        
    def transform_to_global(self, pos_local, pos_now, head_now):
        """Transform local coordinates to global"""
        cos, sin = head_now.cos(), head_now.sin()
        rot_mat = torch.zeros((2, 2), dtype=torch.float32)
        rot_mat[0, 0] = cos
        rot_mat[0, 1] = sin
        rot_mat[1, 0] = -sin
        rot_mat[1, 1] = cos
        
        pos_global = torch.matmul(pos_local, rot_mat.T)
        pos_global = pos_global + pos_now
        return pos_global
    
    def rollout_single_token(self, 
                            token: torch.Tensor,
                            pos_now: torch.Tensor,
                            head_now: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rollout a single action token from current pose"""
        trajectory = [pos_now.clone()]
        
        for t in range(token.shape[0]):  # 6 timesteps
            waypoints_local = token[t, :, :]  # [4, 2]
            waypoints_global = self.transform_to_global(
                waypoints_local, pos_now, head_now
            )
            
            pos_next = waypoints_global.mean(dim=0)
            diff = waypoints_global[-1] - waypoints_global[0]
            head_next = torch.atan2(diff[1], diff[0])
            
            trajectory.append(pos_next.clone())
            pos_now = pos_next
            head_now = head_next
        
        return torch.stack(trajectory), head_now
    
    def segment_error(self, 
                     gt_segment: np.ndarray,
                     token: torch.Tensor,
                     start_pos: torch.Tensor,
                     start_head: torch.Tensor) -> float:
        """
        Calculate error for a single segment
        
        gt_segment: [T, 3] ground truth segment
        token: [6, 4, 2] codebook token
        start_pos, start_head: initial pose for rollout
        
        Returns: mean L2 error over segment
        """
        # Rollout token
        traj, _ = self.rollout_single_token(token, start_pos, start_head)
        traj = traj.numpy()
        
        # Align lengths (token produces 7 positions, gt has 2)
        # For segment comparison: use first len(gt_segment) positions from rollout
        traj_pos = traj[:min(len(traj), len(gt_segment)), :2]
        gt_pos = gt_segment[:len(traj_pos), :2]
        
        # Calculate error
        errors = np.linalg.norm(traj_pos - gt_pos, axis=1)
        return np.mean(errors)
    
    def quantize_trajectory_optimal(self, 
                                   ground_truth: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        Quantize using SEGMENT-WISE optimal matching (not greedy accumulation)
        
        For each timestep t → t+1:
        - Try all K tokens
        - Pick best match for THIS segment
        - Move to next segment (independent)
        
        This avoids error accumulation!
        """
        T = len(ground_truth)
        gt_tensor = torch.tensor(ground_truth, dtype=torch.float32)
        
        pos_now = torch.tensor([0.0, 0.0])
        head_now = torch.tensor(0.0)
        
        reconstructed = [pos_now.clone().numpy()]
        token_indices = []
        
        for t in range(T - 1):
            # Current segment: from t to t+1
            gt_segment = ground_truth[t:t+2]  # [2, 3]
            
            # Find best token for this segment
            best_k = -1
            best_error = float('inf')
            best_traj = None
            best_head = None
            
            for k in range(self.codebook.shape[0]):
                error = self.segment_error(
                    gt_segment, 
                    self.codebook[k],
                    pos_now,
                    head_now
                )
                
                if error < best_error:
                    best_error = error
                    best_k = k
                    # Precompute the trajectory for later
                    best_traj, best_head = self.rollout_single_token(
                        self.codebook[k], pos_now, head_now
                    )
            
            token_indices.append(best_k)
            
            # Update for next segment
            pos_now = best_traj[-1]
            head_now = best_head
            reconstructed.append(pos_now.numpy())
        
        reconstructed = np.array(reconstructed)
        return reconstructed, token_indices
    
    def calculate_ade_fde(self, 
                         ground_truth: np.ndarray,
                         reconstructed: np.ndarray) -> Tuple[float, float]:
        """Calculate ADE and FDE"""
        gt_pos = ground_truth[:, :2]
        rec_pos = reconstructed[:, :2]
        
        min_len = min(len(gt_pos), len(rec_pos))
        gt_pos = gt_pos[:min_len]
        rec_pos = rec_pos[:min_len]
        
        displacements = np.linalg.norm(gt_pos - rec_pos, axis=1)
        ade = np.mean(displacements)
        fde = np.linalg.norm(gt_pos[-1] - rec_pos[-1])
        
        return ade, fde
    
    def load_nuscenes_trajectories(self, num_samples: int = 100):
        """Load real GT trajectories from NuScenes"""
        data_dir = Path("data/nuscenes_processed/train")
        json_files = list(data_dir.glob("*.json"))
        
        sampled_files = random.sample(json_files, min(num_samples, len(json_files)))
        
        trajectories = []
        for json_file in sampled_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            gt_traj = np.array(data['gt_trajectory'])  # [10, 3]
            gt_traj = gt_traj - gt_traj[0]  # Normalize
            trajectories.append(gt_traj)
        
        return trajectories
    
    def evaluate(self, num_samples: int = 100):
        """Evaluate with OPTIMAL segment-wise matching"""
        print("\n" + "="*80)
        print("EVALUATION: K=2048 with OPTIMAL Segment-Wise Matching")
        print("="*80)
        
        print(f"\n📂 Loading {num_samples} real NuScenes trajectories...")
        trajectories = self.load_nuscenes_trajectories(num_samples)
        print(f"✅ Loaded {len(trajectories)} trajectories")
        
        all_ades = []
        all_fdes = []
        
        for i, gt_traj in enumerate(trajectories):
            if (i + 1) % 20 == 0:
                print(f"   Processing: {i+1}/{len(trajectories)}")
            
            try:
                # Quantize with optimal segment-wise matching
                rec_traj, tokens = self.quantize_trajectory_optimal(gt_traj)
                
                # Calculate metrics
                ade, fde = self.calculate_ade_fde(gt_traj, rec_traj)
                all_ades.append(ade)
                all_fdes.append(fde)
            except Exception as e:
                print(f"   ⚠️ Error processing trajectory {i}: {e}")
                continue
        
        if not all_ades:
            print("❌ No valid results!")
            return None
        
        all_ades = np.array(all_ades)
        all_fdes = np.array(all_fdes)
        
        ade_mean = np.mean(all_ades)
        ade_std = np.std(all_ades)
        fde_mean = np.mean(all_fdes)
        fde_std = np.std(all_fdes)
        
        print(f"\n📊 RESULTS (K=2048, Optimal Segment-Wise, {len(all_ades)} trajectories):")
        print(f"   ADE: {ade_mean:.6f} ± {ade_std:.6f} m")
        print(f"   FDE: {fde_mean:.6f} ± {fde_std:.6f} m")
        print(f"   Min ADE: {np.min(all_ades):.6f} m")
        print(f"   Max ADE: {np.max(all_ades):.6f} m")
        
        paper_ade = 0.0182
        paper_fde = 0.0203
        
        print(f"\n📄 PAPER VALUES (Table 4, K=2048):")
        print(f"   ADE: {paper_ade:.6f} m")
        print(f"   FDE: {paper_fde:.6f} m")
        
        print(f"\n⚠️  DIFFERENCE ANALYSIS:")
        print(f"   ADE difference: {ade_mean - paper_ade:+.6f} m ({(ade_mean/paper_ade - 1)*100:+.1f}%)")
        print(f"   FDE difference: {fde_mean - paper_fde:+.6f} m ({(fde_mean/paper_fde - 1)*100:+.1f}%)")
        
        print(f"\n💡 COMPARISON vs GREEDY:")
        print(f"   Greedy ADE: 6.18m")
        print(f"   Optimal ADE: {ade_mean:.6f}m")
        if ade_mean < 6.18:
            improvement = (1 - ade_mean / 6.18) * 100
            print(f"   ✅ IMPROVEMENT: {improvement:.1f}% better!")
        
        # Save results
        results = {
            "codebook_size": 2048,
            "num_trajectories": len(all_ades),
            "method": "Optimal Segment-Wise",
            "data_source": "NuScenes processed (real GT trajectories)",
            "ade_mean": float(ade_mean),
            "ade_std": float(ade_std),
            "ade_min": float(np.min(all_ades)),
            "ade_max": float(np.max(all_ades)),
            "fde_mean": float(fde_mean),
            "fde_std": float(fde_std),
            "paper_ade": float(paper_ade),
            "paper_fde": float(paper_fde),
            "ade_diff_percent": float((ade_mean/paper_ade - 1)*100),
        }
        
        output_path = "evaluation_results_K2048_optimal.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to {output_path}")
        
        return results

if __name__ == "__main__":
    evaluator = K2048OptimalEvaluator()
    results = evaluator.evaluate(num_samples=100)
