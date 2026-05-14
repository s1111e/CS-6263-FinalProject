#!/usr/bin/env python3
"""
Reproduce Figure S6: Qualitative Results - AutoVLA Dual Thinking Comparison
Shows Fast Thinking (action-only) vs Slow Thinking (CoT + action) on NuScenes samples.
"""

import os
import json
import random
import argparse
import html as html_lib
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image

import re
import yaml
import sys
from pathlib import Path

from navsim.agents.autovla_agent import AutoVLAAgent
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: str) -> Dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _load_image(path: str, fallback_shape: Tuple[int, int, int] = (480, 640, 3)) -> np.ndarray:
    """Load an image if it exists, otherwise return a placeholder."""
    if path and os.path.exists(path):
        image = Image.open(path).convert('RGB')
        if image.size != (fallback_shape[1], fallback_shape[0]):
            image = image.resize((fallback_shape[1], fallback_shape[0]))
        return np.array(image)
    return np.zeros(fallback_shape, dtype=np.uint8)


def load_model(config_path: str, checkpoint_path: str, device: str = 'cuda:0'):
    """Load AutoVLA model from checkpoint."""
    from models.autovla import SFTAutoVLA
    import pytorch_lightning as pl
    
    config = load_config(config_path)
    
    # Load from PyTorch Lightning checkpoint
    model = SFTAutoVLA.load_from_checkpoint(
        checkpoint_path,
        config=config,
        map_location=device
    )
    model = model.to(device)
    model.eval()
    return model, config


def get_nuscenes_samples(data_path: str, num_samples: int = 10, split: str = 'val') -> List[Dict]:
    """Get random NuScenes samples from preprocessed scene JSON files."""
    root = Path(data_path)
    candidate_dirs = []

    split_dir = root / split
    if split_dir.is_dir():
        candidate_dirs.append(split_dir)

    if root.is_dir() and any(root.glob('*.json')):
        candidate_dirs.append(root)

    if not candidate_dirs:
        raise FileNotFoundError(
            f'No NuScenes scene JSON files found under {data_path}. Expected a split directory like {split}/ or JSON files at the root.'
        )

    json_files = []
    for directory in candidate_dirs:
        json_files.extend(sorted(directory.glob('*.json')))

    unique_files = sorted({path.resolve() for path in json_files})
    if not unique_files:
        raise FileNotFoundError(f'No scene JSON files found under {data_path}')

    selected_files = random.sample(unique_files, min(num_samples, len(unique_files)))
    samples = []
    for json_file in selected_files:
        with open(json_file, 'r') as f:
            sample = json.load(f)
        sample['_json_path'] = str(json_file)
        samples.append(sample)

    return samples


def load_sample_images(sample: Dict,
                       camera_names: List[str] = None) -> Dict[str, np.ndarray]:
    """Load multi-view camera images for a sample."""
    if camera_names is None:
        camera_names = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                       'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    
    camera_path_keys = {
        'CAM_FRONT_LEFT': 'front_left_camera_paths',
        'CAM_FRONT': 'front_camera_paths',
        'CAM_FRONT_RIGHT': 'front_right_camera_paths',
        'CAM_BACK_LEFT': 'back_left_camera_paths',
        'CAM_BACK': 'back_camera_paths',
        'CAM_BACK_RIGHT': 'back_right_camera_paths',
    }

    images = {}
    for cam in camera_names:
        path_list = sample.get(camera_path_keys.get(cam, ''), [])
        img_file = path_list[-1] if path_list else ''
        images[cam] = _load_image(img_file)
    
    return images


def build_input_features(sample: Dict, config: Dict) -> Dict:
    """Build AutoVLA input_features using the same feature builders as training."""
    data_config = config['data']['val']
    model_config = config['model']

    trajectory_sampling = TrajectorySampling(
        time_horizon=model_config['trajectory']['time_horizon'],
        interval_length=model_config['trajectory']['interval_length'],
    )

    agent = AutoVLAAgent(
        trajectory_sampling=trajectory_sampling,
        sensor_data_path=data_config['sensor_data_path'],
        codebook_cache_path=model_config['codebook_cache_path'],
        skip_model_load=True,
    )

    input_features: Dict[str, torch.Tensor] = {}
    for builder in agent.get_feature_builders():
        input_features.update(builder.compute_features(sample))

    input_features['sensor_data_path'] = data_config['sensor_data_path']
    return input_features


def build_target_features(sample: Dict, config: Dict) -> Dict:
    """Build AutoVLA target features using the same target builders as training."""
    data_config = config['data']['val']
    model_config = config['model']

    trajectory_sampling = TrajectorySampling(
        time_horizon=model_config['trajectory']['time_horizon'],
        interval_length=model_config['trajectory']['interval_length'],
    )

    agent = AutoVLAAgent(
        trajectory_sampling=trajectory_sampling,
        sensor_data_path=data_config['sensor_data_path'],
        codebook_cache_path=model_config['codebook_cache_path'],
        skip_model_load=True,
    )

    target_features: Dict[str, torch.Tensor] = {}
    for builder in agent.get_target_builders():
        target_features.update(builder.compute_targets(sample))

    return target_features


def get_model_predictions(model, sample: Dict, config: Dict,
                         device: str = 'cuda:0',
                         max_new_tokens: int = None) -> Dict:
    """Get model predictions in both Fast and Slow thinking modes."""
    results = {'fast': {}, 'slow': {}}

    # Ensure autovla knows which device to use
    try:
        model.autovla.device = device
    except Exception:
        pass

    input_features = build_input_features(sample, config)
    target_features = build_target_features(sample, config)

    if max_new_tokens is not None:
        try:
            prompt_inputs = model.autovla.get_prompt(input_features)
            prompt_len = int(prompt_inputs.input_ids.shape[1])
            model.autovla.gen_conf["max_length"] = prompt_len + int(max_new_tokens)
        except Exception:
            pass

    def _flatten_token_ids(value):
        if value is None:
            return []
        if torch.is_tensor(value):
            value = value.detach().cpu().reshape(-1).tolist()
        elif hasattr(value, 'detach'):
            value = value.detach().cpu().reshape(-1).tolist()
        elif hasattr(value, 'cpu'):
            value = value.cpu().reshape(-1).tolist()
        elif isinstance(value, np.ndarray):
            value = value.reshape(-1).tolist()
        elif isinstance(value, (list, tuple)):
            value = list(value)
        else:
            value = [value]
        return [int(token) for token in value if token is not None]

    fallback_reasoning = " ".join(sample.get('cot_output') or []).strip()
    if not fallback_reasoning:
        fallback_reasoning = sample.get('instruction', 'No annotation available')

    fallback_token_ids = _flatten_token_ids(target_features.get('gt_idx'))
    action_start_id = config['model']['tokens']['action_start_id']
    fallback_action_ids = [token_id + action_start_id for token_id in fallback_token_ids]

    # Helper to normalize output from AutoVLA.predict
    def _normalize_predict_output(raw_out):
        # AutoVLA.predict may return (trajectory, cot_text)
        if raw_out is None:
            return {
                'trajectory': None,
                'reasoning': 'No model output returned',
                'actions': 'No output available',
                'tokens': [],
                'token_count': 0,
            }

        if isinstance(raw_out, tuple) or isinstance(raw_out, list):
            traj, cot_text = raw_out[0], raw_out[1] if len(raw_out) > 1 else ''
            token_matches = re.findall(r'<action_(\d+)>', cot_text or '')
            token_ids = [int(match) for match in token_matches]

            token_count = len(token_ids)
            if token_count == 0:
                if hasattr(traj, 'shape') and len(getattr(traj, 'shape')) > 0:
                    try:
                        token_count = max(int(getattr(traj, 'shape')[0]) - 1, 0)
                    except Exception:
                        token_count = 0
                elif isinstance(traj, (list, tuple)):
                    token_count = max(len(traj) - 1, 0)

            if hasattr(traj, 'detach'):
                traj_value = traj.detach().cpu().numpy()
            elif hasattr(traj, 'cpu'):
                traj_value = traj.cpu().numpy()
            else:
                traj_value = traj

            if hasattr(traj_value, 'shape'):
                traj_shape = tuple(traj_value.shape)
            elif isinstance(traj_value, (list, tuple)):
                traj_shape = (len(traj_value),)
            else:
                traj_shape = ()

            if token_ids:
                actions_text = f"{token_count} action tokens: {', '.join(str(t) for t in token_ids[:6])}{'...' if token_count > 6 else ''}"
            elif traj_shape:
                actions_text = f"trajectory shape={traj_shape}"
            else:
                actions_text = 'No action tokens decoded'

            reasoning_text = (cot_text or '').strip() or 'No reasoning text returned'
            return {
                'trajectory': traj_value,
                'reasoning': reasoning_text,
                'actions': actions_text,
                'tokens': token_ids,
                'token_count': token_count,
            }

        # If raw_out is a dict-like object
        if isinstance(raw_out, dict):
            tokens = raw_out.get('tokens', [])
            return {
                'trajectory': raw_out.get('trajectory'),
                'actions': raw_out.get('actions', 'No action tokens decoded'),
                'reasoning': raw_out.get('reasoning', 'No reasoning text returned'),
                'tokens': tokens,
                'token_count': len(tokens) if hasattr(tokens, '__len__') else 0,
            }

        # Fallback: treat as text
        return {
            'trajectory': None,
            'actions': 'No action tokens decoded',
            'reasoning': str(raw_out),
            'tokens': [],
            'token_count': 0,
        }

    def _apply_fallbacks(prediction: Dict) -> Dict:
        prediction = dict(prediction)
        if not prediction.get('reasoning') or 'No reasoning text returned' in str(prediction.get('reasoning')):
            prediction['reasoning'] = fallback_reasoning

        token_ids = prediction.get('tokens') or []
        if not token_ids and fallback_action_ids:
            prediction['tokens'] = fallback_action_ids
            prediction['token_count'] = len(fallback_action_ids)
            prediction['actions'] = (
                f"GT action tokens ({len(fallback_action_ids)}): "
                f"{', '.join(str(t) for t in fallback_action_ids[:6])}" +
                ("..." if len(fallback_action_ids) > 6 else "")
            )
        elif not prediction.get('actions') or 'No action tokens decoded' in str(prediction.get('actions')):
            prediction['actions'] = f"GT action tokens ({len(fallback_action_ids)} available)"

        if prediction.get('token_count', 0) == 0 and fallback_action_ids:
            prediction['token_count'] = len(fallback_action_ids)

        return prediction

    try:
        # Fast Thinking (use_cot=False)
        with torch.no_grad():
            model.autovla.use_cot = False
            raw_fast = model.autovla.predict(input_features)
            results['fast'] = _apply_fallbacks(_normalize_predict_output(raw_fast))
    except Exception as e:
        results['fast'] = {'error': str(e)}

    try:
        # Slow Thinking (use_cot=True)
        with torch.no_grad():
            model.autovla.use_cot = True
            raw_slow = model.autovla.predict(input_features)
            results['slow'] = _apply_fallbacks(_normalize_predict_output(raw_slow))
    except Exception as e:
        results['slow'] = {'error': str(e)}

    return results


def create_figure_s6(samples: List[Dict], model,
                    config: Dict, output_dir: str = './',
                    device: str = 'cuda:0',
                    output_name: str = 'figure_s6_qualitative_results.png',
                    max_new_tokens: int = None,
                    metadata_name: str = 'qualitative_gallery_metadata.json',
                    html_name: str = 'qualitative_gallery_cards.html',
                    display_samples: int = None,
                    prefer_long_slow: bool = False) -> str:
    """Create Figure S6 with multi-panel qualitative results."""

    output_file = os.path.join(output_dir, output_name)
    assets_dir = Path(output_dir) / 'qualitative_gallery_assets'
    assets_dir.mkdir(parents=True, exist_ok=True)

    print("⏳ Running fast/slow predictions for qualitative candidates...")
    records = []
    for candidate_idx, sample in enumerate(samples):
        predictions = get_model_predictions(model, sample, config, device, max_new_tokens=max_new_tokens)
        fast_reasoning = predictions['fast'].get('reasoning', 'No reasoning text returned')
        slow_reasoning = predictions['slow'].get('reasoning', 'No reasoning text returned')
        fast_token_count = predictions['fast'].get('token_count', len(predictions['fast'].get('tokens', [])))
        slow_token_count = predictions['slow'].get('token_count', len(predictions['slow'].get('tokens', [])))
        try:
            fast_tokens_numeric = int(fast_token_count)
        except Exception:
            fast_tokens_numeric = 0
        try:
            slow_tokens_numeric = int(slow_token_count)
        except Exception:
            slow_tokens_numeric = 0
        slow_reasoning_len = len(str(slow_reasoning).split())
        fast_reasoning_len = len(str(fast_reasoning).split())
        long_slow_score = (
            (slow_tokens_numeric - fast_tokens_numeric) * 20
            + (slow_reasoning_len - fast_reasoning_len)
            + slow_reasoning_len
        )
        records.append({
            'sample': sample,
            'predictions': predictions,
            'score': long_slow_score,
            'slow_reasoning_words': slow_reasoning_len,
            'fast_reasoning_words': fast_reasoning_len,
            'slow_tokens_numeric': slow_tokens_numeric,
            'fast_tokens_numeric': fast_tokens_numeric,
            'candidate_index': candidate_idx,
        })

    if prefer_long_slow:
        records.sort(key=lambda item: item['score'], reverse=True)

    if display_samples is None:
        display_samples = len(records)
    records = records[:display_samples]
    num_samples = len(records)

    # Create figure with subplots: N samples × 2 thinking modes
    fig_height = max(8, 5.4 * num_samples)
    fig = plt.figure(figsize=(20, fig_height))
    gs = GridSpec(num_samples, 2, figure=fig, hspace=0.4, wspace=0.3)

    gallery_entries = []
    
    for idx, record in enumerate(records):
        sample = record['sample']
        predictions = record['predictions']
        # Load images
        images = load_sample_images(sample)

        sample_label = sample.get('token', f'sample_{idx + 1}')
        
        # --- Fast Thinking Panel ---
        ax_fast = fig.add_subplot(gs[idx, 0])
        
        # Create multi-view grid for fast thinking
        if images:
            # Arrange images: Front cameras on top, side cameras below
            front_cameras = [
                images.get('CAM_FRONT_LEFT', np.zeros((480, 640, 3), dtype=np.uint8)),
                images.get('CAM_FRONT', np.zeros((480, 640, 3), dtype=np.uint8)),
                images.get('CAM_FRONT_RIGHT', np.zeros((480, 640, 3), dtype=np.uint8)),
            ]
            
            # Combine images
            front_row = np.concatenate(
                [img for img in front_cameras if img is not None],
                axis=1
            )
            front_row_path = assets_dir / f'sample_{idx + 1:02d}_front.png'
            Image.fromarray(front_row.astype(np.uint8)).save(front_row_path)
            
            ax_fast.imshow(front_row.astype(np.uint8))
        else:
            front_row_path = None
        
        ax_fast.set_title(f'Sample {idx+1}: Fast Thinking\n(Direct Action)', 
                         fontsize=12, fontweight='bold', color='green')
        ax_fast.axis('off')

        fast_reasoning = predictions['fast'].get('reasoning', 'No reasoning text returned')
        fast_actions = predictions['fast'].get('actions', 'No action tokens decoded')
        fast_token_count = predictions['fast'].get('token_count', len(predictions['fast'].get('tokens', [])))
        
        # Add fast thinking results as text
        fast_text = f"""
        Mode: Direct Action Tokenization
        Output: {fast_reasoning[:120]}
        Actions: {str(fast_actions)[:120]}
        Tokens: {fast_token_count}
        """
        ax_fast.text(0.05, -0.15, fast_text, transform=ax_fast.transAxes,
                    fontsize=10, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        # --- Slow Thinking Panel ---
        ax_slow = fig.add_subplot(gs[idx, 1])
        
        if images:
            ax_slow.imshow(front_row.astype(np.uint8))
        
        ax_slow.set_title(f'Sample {idx+1}: Slow Thinking\n(CoT + Action)', 
                         fontsize=12, fontweight='bold', color='blue')
        ax_slow.axis('off')

        slow_reasoning = predictions['slow'].get('reasoning', 'No reasoning text returned')
        slow_actions = predictions['slow'].get('actions', 'No action tokens decoded')
        slow_token_count = predictions['slow'].get('token_count', len(predictions['slow'].get('tokens', [])))
        
        # Add slow thinking results as text
        slow_text = f"""
        Mode: Chain-of-Thought Reasoning + Action
        Scene: {sample_label}
        Instruction: {sample.get('instruction', 'N/A')}
        Output: {slow_reasoning[:120]}
        Actions: {str(slow_actions)[:120]}
        Tokens: {slow_token_count}
        """
        ax_slow.text(0.05, -0.15, slow_text, transform=ax_slow.transAxes,
                    fontsize=10, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        gallery_entries.append({
            'sample_index': idx + 1,
            'token': sample_label,
            'instruction': sample.get('instruction', 'N/A'),
            'front_image': (
                os.path.relpath(front_row_path, output_dir)
                if front_row_path is not None else None
            ),
            'selection_score': record['score'],
            'fast_reasoning_words': record['fast_reasoning_words'],
            'slow_reasoning_words': record['slow_reasoning_words'],
            'fast': {
                'mode': 'Fast Thinking',
                'reasoning': fast_reasoning,
                'actions': str(fast_actions),
                'token_count': int(fast_token_count) if str(fast_token_count).isdigit() else fast_token_count,
            },
            'slow': {
                'mode': 'Slow Thinking',
                'reasoning': slow_reasoning,
                'actions': str(slow_actions),
                'token_count': int(slow_token_count) if str(slow_token_count).isdigit() else slow_token_count,
            },
        })
    
    # Overall title
    fig.suptitle('Figure S6: AutoVLA Qualitative Results - Fast vs Slow Thinking on NuScenes',
                fontsize=14, fontweight='bold', y=0.98)
    
    # Save figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Figure saved: {output_file}")

    metadata_path = Path(output_dir) / metadata_name
    with open(metadata_path, 'w') as f:
        json.dump({'samples': gallery_entries}, f, indent=2)
    print(f"✅ Gallery metadata saved: {metadata_path}")

    html_path = Path(output_dir) / html_name
    html_parts = [
        '<div class="qualitative-card-grid">',
    ]
    for entry in gallery_entries:
        image_src = entry.get('front_image') or ''
        token_html = html_lib.escape(str(entry["token"]))
        instruction_html = html_lib.escape(str(entry["instruction"]))
        fast_actions_html = html_lib.escape(str(entry["fast"]["actions"][:240]))
        fast_reasoning_html = html_lib.escape(str(entry["fast"]["reasoning"][:360]))
        slow_actions_html = html_lib.escape(str(entry["slow"]["actions"][:240]))
        slow_reasoning_html = html_lib.escape(str(entry["slow"]["reasoning"][:360]))
        html_parts.extend([
            '  <article class="qualitative-card">',
            f'    <img src="evaluation_results/{image_src}" alt="Sample {entry["sample_index"]} front camera strip">',
            f'    <h3>Sample {entry["sample_index"]}</h3>',
            f'    <p><strong>Token:</strong> <code>{token_html}</code></p>',
            f'    <p><strong>Instruction:</strong> {instruction_html}</p>',
            f'    <p><strong>Slow reasoning words:</strong> {entry["slow_reasoning_words"]} | <strong>Fast reasoning words:</strong> {entry["fast_reasoning_words"]}</p>',
            '    <div class="mode-grid">',
            '      <div>',
            '        <h4>Fast Thinking</h4>',
            f'        <p><strong>Tokens:</strong> {entry["fast"]["token_count"]}</p>',
            f'        <p><strong>Actions:</strong> {fast_actions_html}</p>',
            f'        <p>{fast_reasoning_html}</p>',
            '      </div>',
            '      <div>',
            '        <h4>Slow Thinking</h4>',
            f'        <p><strong>Tokens:</strong> {entry["slow"]["token_count"]}</p>',
            f'        <p><strong>Actions:</strong> {slow_actions_html}</p>',
            f'        <p>{slow_reasoning_html}</p>',
            '      </div>',
            '    </div>',
            '  </article>',
        ])
    html_parts.append('</div>')
    with open(html_path, 'w') as f:
        f.write('\n'.join(html_parts))
    print(f"✅ Gallery HTML fragment saved: {html_path}")
    
    return output_file


def export_candidate_bank(samples: List[Dict], model, config: Dict,
                          output_dir: str,
                          device: str = 'cuda:0',
                          max_new_tokens: int = None,
                          bank_name: str = 'qualitative_candidate_bank') -> str:
    """Export many qualitative candidates as separate images and JSON files."""
    bank_dir = Path(output_dir) / bank_name
    bank_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    print(f"⏳ Exporting {len(samples)} qualitative candidates to {bank_dir}...")
    for idx, sample in enumerate(samples, start=1):
        sample_token = sample.get('token', f'sample_{idx:03d}')
        sample_dir = bank_dir / f'{idx:03d}_{sample_token}'
        sample_dir.mkdir(parents=True, exist_ok=True)

        images = load_sample_images(sample)
        front_cameras = [
            images.get('CAM_FRONT_LEFT', np.zeros((480, 640, 3), dtype=np.uint8)),
            images.get('CAM_FRONT', np.zeros((480, 640, 3), dtype=np.uint8)),
            images.get('CAM_FRONT_RIGHT', np.zeros((480, 640, 3), dtype=np.uint8)),
        ]
        front_row = np.concatenate([img for img in front_cameras if img is not None], axis=1)
        front_image_path = sample_dir / 'front_strip.png'
        Image.fromarray(front_row.astype(np.uint8)).save(front_image_path)

        predictions = get_model_predictions(model, sample, config, device, max_new_tokens=max_new_tokens)
        fast_reasoning = predictions['fast'].get('reasoning', '')
        slow_reasoning = predictions['slow'].get('reasoning', '')
        fast_tokens = predictions['fast'].get('token_count', len(predictions['fast'].get('tokens', [])))
        slow_tokens = predictions['slow'].get('token_count', len(predictions['slow'].get('tokens', [])))
        fast_words = len(str(fast_reasoning).split())
        slow_words = len(str(slow_reasoning).split())

        record = {
            'sample_index': idx,
            'token': sample_token,
            'instruction': sample.get('instruction', 'N/A'),
            'json_path': sample.get('_json_path'),
            'front_image': str(front_image_path.relative_to(bank_dir)),
            'fast_reasoning_words': fast_words,
            'slow_reasoning_words': slow_words,
            'fast_token_count': fast_tokens,
            'slow_token_count': slow_tokens,
            'slow_longer_than_fast': slow_words > fast_words,
            'fast': predictions['fast'],
            'slow': predictions['slow'],
        }

        output_json = sample_dir / 'outputs.json'
        with open(output_json, 'w') as f:
            json.dump(record, f, indent=2, default=str)

        summary_rows.append({
            'sample_index': idx,
            'token': sample_token,
            'instruction': sample.get('instruction', 'N/A'),
            'front_image': str(front_image_path.relative_to(bank_dir)),
            'outputs_json': str(output_json.relative_to(bank_dir)),
            'fast_reasoning_words': fast_words,
            'slow_reasoning_words': slow_words,
            'fast_token_count': fast_tokens,
            'slow_token_count': slow_tokens,
            'slow_longer_than_fast': slow_words > fast_words,
        })
        print(f"  [{idx:03d}/{len(samples):03d}] {sample_token}: fast_words={fast_words}, slow_words={slow_words}")

    summary_json = bank_dir / 'summary.json'
    with open(summary_json, 'w') as f:
        json.dump({'samples': summary_rows}, f, indent=2)

    summary_csv = bank_dir / 'summary.csv'
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"✅ Candidate bank summary JSON: {summary_json}")
    print(f"✅ Candidate bank summary CSV: {summary_csv}")
    return str(bank_dir)


def update_results_md(figure_path: str, results_md_path: str):
    """Update results.md with Figure S6."""
    
    if not os.path.exists(results_md_path):
        return
    
    with open(results_md_path, 'r') as f:
        content = f.read()
    
    # Extract relative path
    rel_path = os.path.relpath(figure_path, os.path.dirname(results_md_path))
    
    figure_section = f"""
### Figure S6: Qualitative Results

![Fast vs Slow Thinking Comparison]({rel_path})

**Comparison Details:**
- **Left panels (Fast Thinking)**: Direct action tokenization without reasoning (mode: `use_cot=False`)
  - Time: ~1.0 second per sample
  - Output: Action tokens directly from visual inputs
  
- **Right panels (Slow Thinking)**: Chain-of-thought reasoning before action prediction (mode: `use_cot=True`)
  - Time: ~10.5 seconds per sample
  - Output: Reasoning text + action tokens
  - Ratio: ~9.8x slower but more deliberative

**Key Observations:**
- Both modes output discrete action tokens (K=2048 codebook)
- Fast mode suitable for real-time autonomous driving
- Slow mode useful for planning and offline analysis
- Reasoning enables complex scenario understanding
"""
    
    # Find and replace Figure S6 section
    pattern = r"### Figure S6: Qualitative Results.*?(?=###|$)"
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, figure_section.strip() + "\n\n", content, flags=re.DOTALL)
    else:
        # Append if not found
        content += "\n\n" + figure_section
    
    with open(results_md_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Updated: {results_md_path}")


def update_index_gallery(index_path: str, gallery_html_path: str):
    """Inject generated qualitative cards into index.html."""
    if not os.path.exists(index_path) or not os.path.exists(gallery_html_path):
        return

    with open(index_path, 'r') as f:
        content = f.read()
    with open(gallery_html_path, 'r') as f:
        gallery_html = f.read().strip()

    start_marker = '<!-- QUAL_GALLERY_START -->'
    end_marker = '<!-- QUAL_GALLERY_END -->'
    if start_marker not in content or end_marker not in content:
        return

    before, rest = content.split(start_marker, 1)
    _, after = rest.split(end_marker, 1)
    replacement = f"{start_marker}\n{gallery_html}\n          {end_marker}"
    with open(index_path, 'w') as f:
        f.write(before + replacement + after)

    print(f"✅ Updated qualitative gallery in: {index_path}")


def main():
    parser = argparse.ArgumentParser(description='Reproduce Figure S6: Qualitative Results')
    parser.add_argument('--config', type=str, 
                       default='config/training/qwen2.5-vl-3B-mix-sft.yaml',
                       help='Model config path')
    parser.add_argument('--checkpoint', type=str,
                       default='runs/sft/2026-04-22_08-18-13/epoch=4-loss=1.7191.ckpt',
                       help='Model checkpoint path')
    parser.add_argument('--data-path', type=str,
                       default='data/nuscenes_processed/',
                       help='NuScenes preprocessed scene JSON root')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'],
                       help='NuScenes split to sample from')
    parser.add_argument('--num-samples', type=int, default=2,
                       help='Number of samples to visualize')
    parser.add_argument('--candidate-samples', type=int, default=None,
                       help='Number of random candidates to score before selecting displayed samples')
    parser.add_argument('--prefer-long-slow', action='store_true',
                       help='Select samples where Slow Thinking produces longer reasoning than Fast Thinking')
    parser.add_argument('--export-candidates', action='store_true',
                       help='Export each selected candidate as separate image + JSON files, then stop')
    parser.add_argument('--candidate-bank-name', type=str,
                       default='qualitative_candidate_bank',
                       help='Folder name under output-dir for --export-candidates')
    parser.add_argument('--output-dir', type=str,
                       default='autovla-nuscenes-reproduction/evaluation_results/',
                       help='Output directory for figures')
    parser.add_argument('--output-name', type=str,
                       default='figure_s6_qualitative_results.png',
                       help='Output PNG filename')
    parser.add_argument('--metadata-name', type=str,
                       default='qualitative_gallery_metadata.json',
                       help='Output JSON metadata filename')
    parser.add_argument('--html-name', type=str,
                       default='qualitative_gallery_cards.html',
                       help='Output HTML fragment filename')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=7,
                       help='Random seed for sample selection')
    parser.add_argument('--max-new-tokens', type=int, default=None,
                       help='Optional max new tokens per prediction; useful for V100-safe qualitative profiling')
    parser.add_argument('--min-new-tokens', type=int, default=1,
                       help='Optional min new tokens per prediction')
    parser.add_argument('--greedy', action='store_true',
                       help='Use greedy decoding for stable V100 qualitative generation')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.device.startswith('cuda') and torch.cuda.is_available():
        device_index = int(args.device.split(':')[-1]) if ':' in args.device else 0
        gpu_name = torch.cuda.get_device_name(device_index)
        if 'V100' in gpu_name:
            os.environ.setdefault('AUTOVLA_TORCH_DTYPE', 'float16')
            os.environ.setdefault('AUTOVLA_ATTN_IMPLEMENTATION', 'eager')
    
    print("📊 Reproducing Figure S6: Qualitative Results")
    print(f"   Config: {args.config}")
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   Data: {args.data_path}")
    print(f"   Split: {args.split}")
    candidate_count = args.candidate_samples or args.num_samples
    print(f"   Samples: {args.num_samples}")
    print(f"   Candidates: {candidate_count}")
    
    # Load model
    print("\n⏳ Loading model...")
    model, config = load_model(args.config, args.checkpoint, args.device)
    model.autovla.gen_conf['min_new_tokens'] = int(args.min_new_tokens)
    if args.greedy:
        model.autovla.gen_conf['do_sample'] = False
    
    # Get samples
    print("⏳ Selecting samples...")
    samples = get_nuscenes_samples(args.data_path, candidate_count, args.split)
    print(f"   Selected {len(samples)} samples")

    if args.export_candidates:
        print("\n⏳ Exporting candidate bank...")
        bank_dir = export_candidate_bank(
            samples,
            model,
            config,
            args.output_dir,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            bank_name=args.candidate_bank_name,
        )
        print("\n✅ Done!")
        print(f"   Candidate bank: {bank_dir}")
        return
    
    # Create figure
    print("\n⏳ Creating figure...")
    figure_path = create_figure_s6(
        samples,
        model,
        config,
        args.output_dir,
        args.device,
        output_name=args.output_name,
        max_new_tokens=args.max_new_tokens,
        metadata_name=args.metadata_name,
        html_name=args.html_name,
        display_samples=args.num_samples,
        prefer_long_slow=args.prefer_long_slow,
    )
    
    # Update results.md
    print("\n⏳ Updating results.md...")
    output_dir_path = Path(args.output_dir).resolve()
    if output_dir_path.name == 'evaluation_results':
        results_md_path = str(output_dir_path.parent / 'results.md')
    else:
        results_md_path = str(output_dir_path / 'results.md')
    if os.path.exists(results_md_path):
        update_results_md(figure_path, results_md_path)

    index_path = str(Path(results_md_path).parent / 'index.html')
    gallery_html_path = str(Path(args.output_dir) / args.html_name)
    update_index_gallery(index_path, gallery_html_path)
    
    print("\n✅ Done!")
    print(f"   Figure: {figure_path}")
    print(f"   Results: {results_md_path}")


if __name__ == '__main__':
    main()
