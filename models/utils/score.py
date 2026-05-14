import os
import torch
import yaml
import lzma
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from omegaconf import OmegaConf
from hydra.utils import instantiate

from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.common.dataclasses import SensorConfig
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from pathlib import Path
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import Scene, Trajectory


class PDM_Reward:
    """
    A class that encapsulates the RL PDM reward calculation for gievn token.
    """
    def __init__(self, metric_cache_path):
        """
        Initialize the reward calculator with the given configuration.

        :param metric_cache_path: Path to the metric cache.
        """
        # Initialize metric cache loader only when cache layout exists.
        self.metric_cache_loader = None
        if metric_cache_path is not None:
            cache_path = Path(metric_cache_path)
            metadata_dir = cache_path / "metadata"
            if cache_path.exists() and metadata_dir.exists():
                try:
                    self.metric_cache_loader = MetricCacheLoader(cache_path)
                except Exception:
                    self.metric_cache_loader = None
        self.future_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        self.simulator = PDMSimulator(self.future_sampling)
        self.scorer= PDMScorer(self.future_sampling)

    def rl_pdm_score(self, trajectory, token, target_trajectory=None):
        """
        Compute the rl pdm reward for a given token using ADE-based or PDM-based reward.

        :param trajectory: model output (Trajectory object with .waypoints).
        :param token: The scene token.
        :param target_trajectory: Ground truth trajectory dict with 'gt_pos' key.
        """
        # If ground truth trajectory is provided, compute ADE-based reward
        if target_trajectory is not None and isinstance(target_trajectory, dict):
            try:
                # Extract predicted positions from Trajectory object
                pred_positions = None
                if hasattr(trajectory, 'poses'):
                    traj_data = trajectory.poses
                    if isinstance(traj_data, torch.Tensor):
                        traj_data = traj_data.detach().cpu().numpy()
                    else:
                        traj_data = np.asarray(traj_data)
                    if traj_data.ndim >= 2 and traj_data.shape[1] >= 2:
                        pred_positions = traj_data[:, :2]
                elif hasattr(trajectory, 'waypoints'):
                    traj_data = trajectory.waypoints
                    if isinstance(traj_data, torch.Tensor):
                        traj_data = traj_data.detach().cpu().numpy()
                    else:
                        traj_data = np.asarray(traj_data)
                    if traj_data.ndim >= 2 and traj_data.shape[1] >= 2:
                        pred_positions = traj_data[:, :2]
                
                if pred_positions is None:
                    try:
                        traj_data = np.asarray(trajectory)
                        pred_positions = traj_data[:, :2]
                    except Exception:
                        return 0.0
                
                # Extract ground truth positions from target_trajectory dict
                # Key is 'gt_pos' with shape (1, 10, 2) - take [0] for batch
                if 'gt_pos' in target_trajectory:
                    gt_traj = target_trajectory['gt_pos']
                    if isinstance(gt_traj, torch.Tensor):
                        gt_positions = gt_traj[0, :, :2].cpu().numpy()  # Remove batch dim
                    else:
                        gt_positions = gt_traj[0, :, :2]
                elif 'sampled_pos' in target_trajectory:
                    # Fallback to sampled_pos if gt_pos not available
                    gt_traj = target_trajectory['sampled_pos']
                    if isinstance(gt_traj, torch.Tensor):
                        gt_positions = gt_traj[0, :, :2].cpu().numpy()
                    else:
                        gt_positions = gt_traj[0, :, :2]
                else:
                    return 0.0
                
                # Ensure both are numpy arrays
                if isinstance(pred_positions, torch.Tensor):
                    pred_positions = pred_positions.cpu().numpy()
                if isinstance(gt_positions, torch.Tensor):
                    gt_positions = gt_positions.cpu().numpy()
                
                # Ensure same length
                min_len = min(len(pred_positions), len(gt_positions))
                pred_positions = pred_positions[:min_len]
                gt_positions = gt_positions[:min_len]
                
                # Compute Average Displacement Error (ADE)
                ade = np.mean(np.linalg.norm(pred_positions - gt_positions, axis=1))
                
                # Exponential shaping keeps reward in (0, 1] and avoids hard zero collapse.
                reward = float(np.exp(-ade / 8.0))
                
                return reward
            
            except Exception as e:
                print(f"ADE reward failed: {e}")
                return 0.0
        
        # PDM-based reward (when metric cache is available)
        if self.metric_cache_loader is None:
            return 0.0
        
        if token not in self.metric_cache_loader.metric_cache_paths:
            return 0.0

        metric_cache_path = self.metric_cache_loader.metric_cache_paths[token]
        with lzma.open(metric_cache_path, "rb") as f:
            metric_cache = pickle.load(f)

        try:
            # Compute the pdm score
            result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=self.future_sampling,
                simulator=self.simulator,
                scorer=self.scorer,
            )

            final_reward = result.score

            return final_reward

        except Exception as e:
            print(f"Reward calculation failed: {e}")

            return 0.0
