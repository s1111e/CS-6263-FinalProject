#!/usr/bin/env python3
"""
Table 4 Reproduction: K-disk Quantization Accuracy (K=2048 only)

This script evaluates action tokenization accuracy for codebook size K=2048.
We use the SAME codebook that was used during SFT/RFT training of the model.

Paper Reference: AutoVLA Table 4 - Action Tokenization Accuracy
K-disk Method with K=2048 baseline
"""

import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Tuple
import json

class K2048Evaluator:
    def __init__(self, codebook_path: str = "codebook_cache/agent_vocab.pkl"):
        """Initialize with K=2048 codebook (training codebook)"""
        with open(codebook_path, "rb") as f:
            data = pickle.load(f)
        self.codebook = torch.tensor(data['token_all']['veh'], dtype=torch.float32)
        
        print(f"✅ Loaded K=2048 codebook: shape {self.codebook.shape}")
        assert self.codebook.shape[0] == 2048, "Must be K=2048!"
        
    def transform_to_global(self, pos_local, pos_now, head_now):
        """Transform local coordinates to global (from action_tokenizer.py)"""
        cos, sin = head_now.cos(), head_now.sin()
        
        # Rotation matrix
        rot_mat = torch.zeros((2, 2), dtype=torch.float32)
        rot_mat[0, 0] = cos
        rot_mat[0, 1] = sin
        rot_mat[1, 0] = -sin
        rot_mat[1, 1] = cos
        
        # pos_local: [n_waypoints, 2]
        pos_global = torch.matmul(pos_local, rot_mat.T)
        pos_global = pos_global + pos_now
        
        return pos_global
    
    def rollout_single_token(self, 
                            token: torch.Tensor,
                            pos_now: torch.Tensor,
                            head_now: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rollout a single action token starting from current pose
        
        Args:
            token: [6, 4, 2] - 6 timesteps, 4 waypoints, 2D coords (local)
            pos_now: [2] - current position
            head_now: scalar - current heading
            
        Returns:
            (trajectory, final_heading) where trajectory is [6, 2] positions
        """
        trajectory = [pos_now.clone()]
        
        for t in range(token.shape[0]):  # 6 timesteps
            # Get waypoints for this timestep (local coords)
            waypoints_local = token[t, :, :]  # [4, 2]
            
            # Transform to global
            waypoints_global = self.transform_to_global(
                waypoints_local, pos_now, head_now
            )
            
            # Next position: average of waypoints
            pos_next = waypoints_global.mean(dim=0)
            
            # Next heading: from first to last waypoint
            diff = waypoints_global[-1] - waypoints_global[0]
            head_next = torch.atan2(diff[1], diff[0])
            
            trajectory.append(pos_next.clone())
            pos_now = pos_next
            head_now = head_next
        
        return torch.stack(trajectory), head_now
    
    def quantize_trajectory(self, 
                           ground_truth: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        Quantize continuous trajectory to discrete codebook tokens
        
        Args:
            ground_truth: [T, 3] trajectory (x, y, heading) at 2Hz
            
        Returns:
            (reconstructed_traj, token_indices)
        """
        T = len(ground_truth)
        gt_tensor = torch.tensor(ground_truth, dtype=torch.float32)
        
        # Start from origin
        pos_now = torch.tensor([0.0, 0.0])
        head_now = torch.tensor(0.0)
        
        reconstructed = [pos_now.clone().numpy()]
        token_indices = []
        
        for t in range(T):
            # Target position from ground truth
            target_pos = gt_tensor[t, :2]
            
            # Find nearest codebook token based on:
            # Distance from current position to target
            distances = torch.zeros(self.codebook.shape[0])
            
            for k in range(self.codebook.shape[0]):
                # Rollout token and see where it takes us
                traj_k, _ = self.rollout_single_token(
                    self.codebook[k], pos_now, head_now
                )
                # Final position after rollout
                final_pos_k = traj_k[-1]
                distances[k] = torch.norm(final_pos_k - target_pos)
            
            best_k = torch.argmin(distances).item()
            token_indices.append(best_k)
            
            # Rollout best token
            traj_best, head_next = self.rollout_single_token(
                self.codebook[best_k], pos_now, head_now
            )
            
            # Update state
            pos_now = traj_best[-1]
            head_now = head_next
            reconstructed.append(pos_now.numpy())
        
        reconstructed = np.array(reconstructed)
        return reconstructed, token_indices
    
    def calculate_ade_fde(self, 
                         ground_truth: np.ndarray,
                         reconstructed: np.ndarray) -> Tuple[float, float]:
        """Calculate ADE and FDE metrics"""
        # Use only position (x, y), ignore heading
        gt_pos = ground_truth[:, :2]
        rec_pos = reconstructed[:, :2]
        
        # Make same length
        min_len = min(len(gt_pos), len(rec_pos))
        gt_pos = gt_pos[:min_len]
        rec_pos = rec_pos[:min_len]
        
        # ADE: Average Displacement Error
        displacements = np.linalg.norm(gt_pos - rec_pos, axis=1)
        ade = np.mean(displacements)
        
        # FDE: Final Displacement Error
        fde = np.linalg.norm(gt_pos[-1] - rec_pos[-1])
        
        return ade, fde
    
    def generate_mock_gt_trajectory(self, T: int = 10, 
                                     dt: float = 0.5) -> np.ndarray:
        """Generate realistic mock ground truth trajectory"""
        np.random.seed(42)
        traj = np.zeros((T, 3))
        
        # Random walk with smooth heading
        pos = np.array([0.0, 0.0])
        heading = 0.0
        
        for t in range(T):
            # Smooth velocity (changes slowly)
            vel_x = 2.0 + 0.5 * np.sin(t * 0.3)
            vel_y = 0.5 + 0.3 * np.cos(t * 0.2)
            
            # Update position
            pos = pos + np.array([vel_x, vel_y]) * dt
            
            # Update heading
            heading = heading + np.random.normal(0, 0.1)
            
            traj[t] = np.array([pos[0], pos[1], heading])
        
        return traj
    
    def evaluate(self, num_trajectories: int = 100):
        """Evaluate quantization accuracy over multiple trajectories"""
        print("\n" + "="*80)
        print("EVALUATION: K=2048 Codebook Quantization Accuracy")
        print("="*80)
        
        all_ades = []
        all_fdes = []
        
        for i in range(num_trajectories):
            # Generate mock trajectory
            gt_traj = self.generate_mock_gt_trajectory(T=10)
            
            # Quantize and reconstruct
            rec_traj, tokens = self.quantize_trajectory(gt_traj)
            
            # Calculate metrics
            ade, fde = self.calculate_ade_fde(gt_traj, rec_traj)
            all_ades.append(ade)
            all_fdes.append(fde)
        
        all_ades = np.array(all_ades)
        all_fdes = np.array(all_fdes)
        
        # Statistics
        ade_mean = np.mean(all_ades)
        ade_std = np.std(all_ades)
        fde_mean = np.mean(all_fdes)
        fde_std = np.std(all_fdes)
        
        print(f"\n📊 RESULTS (K=2048, {num_trajectories} trajectories):")
        print(f"   ADE: {ade_mean:.6f} ± {ade_std:.6f} m")
        print(f"   FDE: {fde_mean:.6f} ± {fde_std:.6f} m")
        
        # Paper values for comparison
        paper_ade = 0.0182
        paper_fde = 0.0203
        
        print(f"\n📄 PAPER VALUES (Table 4, K=2048):")
        print(f"   ADE: {paper_ade:.6f} m")
        print(f"   FDE: {paper_fde:.6f} m")
        
        print(f"\n⚠️  DIFFERENCE ANALYSIS:")
        print(f"   ADE difference: {ade_mean - paper_ade:+.6f} m ({(ade_mean/paper_ade - 1)*100:+.1f}%)")
        print(f"   FDE difference: {fde_mean - paper_fde:+.6f} m ({(fde_mean/paper_fde - 1)*100:+.1f}%)")
        
        # Save results
        results = {
            "codebook_size": 2048,
            "num_trajectories": num_trajectories,
            "ade_mean": float(ade_mean),
            "ade_std": float(ade_std),
            "fde_mean": float(fde_mean),
            "fde_std": float(fde_std),
            "paper_ade": float(paper_ade),
            "paper_fde": float(paper_fde),
            "note": "Mock GT trajectories used (no real prediction/GT available)"
        }
        
        output_path = "evaluation_results_K2048.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to {output_path}")
        
        return results


if __name__ == "__main__":
    evaluator = K2048Evaluator()
    results = evaluator.evaluate(num_trajectories=100)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"""
✅ Evaluated K=2048 (SAME as training codebook)
✅ Used proper rollout algorithm (from action_tokenizer.py)
✅ Quantized trajectories via nearest token matching
⚠️  Mock GT trajectories used (no real data available)
    
See README.md for detailed explanation of methodology and limitations.
""")
