#!/usr/bin/env python3
"""
Table 4 Reproduction: K=2048 with REAL NuScenes GT Trajectories

This script uses actual NuScenes ground-truth trajectories instead of mock data!
"""

import json
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Tuple
import random

class K2048EvaluatorReal:
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
        """Rollout a single action token"""
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
    
    def quantize_trajectory(self, 
                           ground_truth: np.ndarray) -> Tuple[np.ndarray, list]:
        """Quantize continuous trajectory using codebook"""
        T = len(ground_truth)
        gt_tensor = torch.tensor(ground_truth, dtype=torch.float32)
        
        pos_now = torch.tensor([0.0, 0.0])
        head_now = torch.tensor(0.0)
        
        reconstructed = [pos_now.clone().numpy()]
        token_indices = []
        
        for t in range(T):
            target_pos = gt_tensor[t, :2]
            distances = torch.zeros(self.codebook.shape[0])
            
            for k in range(self.codebook.shape[0]):
                traj_k, _ = self.rollout_single_token(
                    self.codebook[k], pos_now, head_now
                )
                final_pos_k = traj_k[-1]
                distances[k] = torch.norm(final_pos_k - target_pos)
            
            best_k = torch.argmin(distances).item()
            token_indices.append(best_k)
            
            traj_best, head_next = self.rollout_single_token(
                self.codebook[best_k], pos_now, head_now
            )
            
            pos_now = traj_best[-1]
            head_now = head_next
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
        """Load real GT trajectories from NuScenes processed data"""
        data_dir = Path("data/nuscenes_processed/train")
        json_files = list(data_dir.glob("*.json"))
        
        # Randomly sample
        sampled_files = random.sample(json_files, min(num_samples, len(json_files)))
        
        trajectories = []
        for json_file in sampled_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            gt_traj = np.array(data['gt_trajectory'])  # [10, 3]
            # Normalize: center at origin (first position is reference)
            gt_traj = gt_traj - gt_traj[0]
            trajectories.append(gt_traj)
        
        return trajectories
    
    def evaluate(self, num_samples: int = 100):
        """Evaluate with REAL NuScenes data"""
        print("\n" + "="*80)
        print("EVALUATION: K=2048 with REAL NuScenes GT Trajectories")
        print("="*80)
        
        # Load real trajectories
        print(f"\n📂 Loading {num_samples} real NuScenes trajectories...")
        trajectories = self.load_nuscenes_trajectories(num_samples)
        print(f"✅ Loaded {len(trajectories)} trajectories")
        
        all_ades = []
        all_fdes = []
        
        for i, gt_traj in enumerate(trajectories):
            if (i + 1) % 20 == 0:
                print(f"   Processing: {i+1}/{len(trajectories)}")
            
            try:
                # Quantize and reconstruct
                rec_traj, tokens = self.quantize_trajectory(gt_traj)
                
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
        
        # Statistics
        ade_mean = np.mean(all_ades)
        ade_std = np.std(all_ades)
        fde_mean = np.mean(all_fdes)
        fde_std = np.std(all_fdes)
        
        print(f"\n📊 RESULTS (K=2048, {len(all_ades)} real NuScenes trajectories):")
        print(f"   ADE: {ade_mean:.6f} ± {ade_std:.6f} m")
        print(f"   FDE: {fde_mean:.6f} ± {fde_std:.6f} m")
        print(f"   Min ADE: {np.min(all_ades):.6f} m")
        print(f"   Max ADE: {np.max(all_ades):.6f} m")
        
        # Paper values for comparison
        paper_ade = 0.0182
        paper_fde = 0.0203
        
        print(f"\n📄 PAPER VALUES (Table 4, K=2048):")
        print(f"   ADE: {paper_ade:.6f} m")
        print(f"   FDE: {paper_fde:.6f} m")
        
        print(f"\n⚠️  DIFFERENCE ANALYSIS:")
        print(f"   ADE difference: {ade_mean - paper_ade:+.6f} m ({(ade_mean/paper_ade - 1)*100:+.1f}%)")
        print(f"   FDE difference: {fde_mean - paper_fde:+.6f} m ({(fde_mean/paper_fde - 1)*100:+.1f}%)")
        
        print(f"\n💡 INTERPRETATION:")
        if ade_mean < 0.1:
            print(f"   ✅ Good agreement with paper! Quantization works well.")
        elif ade_mean < 0.5:
            print(f"   ⚠️ Moderate difference. Possible reasons:")
            print(f"      - Different codebook clustering method")
            print(f"      - Different trajectory preprocessing")
            print(f"      - Greedy nearest-neighbor vs optimal matching")
        else:
            print(f"   ❌ Large difference. Investigation needed:")
            print(f"      - Check quantization algorithm")
            print(f"      - Check coordinate frame conversion")
            print(f"      - Check codebook rollout logic")
        
        # Save results
        results = {
            "codebook_size": 2048,
            "num_trajectories": len(all_ades),
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
            "fde_diff_percent": float((fde_mean/paper_fde - 1)*100),
        }
        
        output_path = "evaluation_results_K2048_real_data.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to {output_path}")
        
        return results

if __name__ == "__main__":
    evaluator = K2048EvaluatorReal()
    results = evaluator.evaluate(num_samples=100)
