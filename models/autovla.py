import torch
import os
import re
from tqdm import tqdm
from typing import Dict, Any
import pytorch_lightning as pl
from pathlib import Path
import torch.nn.functional as F
import numpy as np
from typing import List
from torch.distributed.fsdp import StateDictType
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from models.action_tokenizer import ActionTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from models.utils.score import PDM_Reward, TrajectorySampling, Trajectory


class GRPOAutoVLA(pl.LightningModule):
    def __init__(self, config: dict, inference=False):
        super().__init__()
        self.cfg = config
        self.use_cot = config['model']['use_cot']
        self.save_hyperparameters()

        # Load trajectory sampling from config or use default
        traj_conf = config['model']['trajectory']
        self.trajectory_sampling = TrajectorySampling(
            num_poses=traj_conf['num_poses'],
            interval_length=traj_conf['interval_length']
        )
        
        # Load token configs
        token_conf = config['model']['tokens']
        self.action_start_id = token_conf['action_start_id']
        self.assistant_id = torch.tensor(token_conf['assistant_id'])

        # Training model (wrapped by Lightning FSDPStrategy)
        self.autovla = AutoVLA(config)

        self.autovla.train()
        self._train_vision_backbone = config['model']['train_vision_backbone']
        self._train_llm_backbone = config['model']['train_lm_backbone']

        # online reference model.
        if not inference:
            self.reference_model = AutoVLA(config, inference=True)
            state_dict = torch.load(config['model']['sft_model_path'])["state_dict"]
            state_dict = {k.replace("autovla.", "").replace("drivevla.", ""): v for k, v in state_dict.items()}
            self.reference_model.load_state_dict(state_dict, strict=False)
            self.reference_model.eval()  
            print(f"Using online reference model from {config['model']['sft_model_path']}")

        # sample generation config
        sample_conf = config['training']['sample']
        self._sample_generation_temperature = {
            "max_length": sample_conf['max_length'],
            "temperature": sample_conf['temperature'],
            "top_k": sample_conf['top_k'],
            "top_p": sample_conf['top_p'],
        }

        # reward function
        self.train_critic = PDM_Reward(Path(config['data']['train']['metric_cache_path']))
        self.val_critic = PDM_Reward(Path(config['data']['val']['metric_cache_path']))

        # sliding window for training reward
        if not inference:
            self.window_size = config['rl']['reward'].get("sliding_window_size", 100)
            self.register_buffer("training_reward_buffer", torch.zeros(self.window_size))
            self.register_buffer("sliding_idx",   torch.zeros(1, dtype=torch.long))
            self.register_buffer("window_count",  torch.zeros(1, dtype=torch.long))

    def training_step(self, batch):
        # Generate a sample from the model.
        self.autovla.train()
        with torch.no_grad():
            sample = self.generate_sample(
                batch, model=self.autovla, device=next(self.parameters()).device)
        
            # Compute the reward for the generated sample.
            reward = self.reward_function(sample)
            reward_scale = self.cfg['rl']['reward'].get("scale", 1.0)
            reward = reward * reward_scale
            
            # Normalize rewards across distributed workers when available.
            # In single-process training, use reward directly to avoid zero advantage.
            if self.trainer.world_size > 1:
                groupped_rewards = self.all_gather(reward)
                advantage = (reward - groupped_rewards.mean()) / (groupped_rewards.std(unbiased=False) + 1e-4)
            else:
                advantage = reward

        # Compute the per-token log probabilities.
        per_token_logps = self.get_per_token_logps(
            self.autovla.vlm, 
            sample['input_ids'], 
            sample['attention_mask'], 
            sample['pixel_values_videos'], 
            sample['video_grid_thw']
        )
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, sample['prompt_length']-1:]
        completion_mask = sample['completion_mask']

        # reference model
        with torch.no_grad():
            ref_per_token_logps = self.get_per_token_logps(
                self.reference_model.vlm, 
                sample["input_ids"], 
                sample["attention_mask"], 
                sample["pixel_values_videos"], 
                sample["video_grid_thw"]
            )
            ref_per_token_logps = ref_per_token_logps[:, sample["prompt_length"]-1:]

        # Compute the policy loss
        per_policy_loss = \
            torch.exp(per_token_logps - per_token_logps.detach()) * advantage.unsqueeze(-1)

        # Compute the kl loss
        kl_beta = self.cfg['rl'].get("kl_beta", 0.0)
        per_token_kl = \
            torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        per_kl_loss = kl_beta * per_token_kl

        per_token_loss = -(per_policy_loss - per_kl_loss)
        token_counts = completion_mask.sum(dim=1).clamp_min(1)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / token_counts).mean()
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

        # Log metrics
        self.log("loss", loss, sync_dist=True, prog_bar=True)
        per_kl_loss = ((per_kl_loss * completion_mask).sum(dim=1) / token_counts).mean()
        self.log("kl_divergence", per_kl_loss, sync_dist=True)

        # record training reward
        self.training_buffer_record(reward.mean())
        return loss
    
    def training_buffer_record(self, step_reward):
        idx = self.sliding_idx.item()
        self.training_reward_buffer[idx] = step_reward

        new_idx = (idx + 1) % self.window_size
        self.sliding_idx.fill_(new_idx)
        new_count = min(self.window_count.item() + 1, self.window_size)
        self.window_count.fill_(new_count)

        if new_count >= self.window_size:
            sliding_avg = self.training_reward_buffer.mean()
            self.log(
                "avg_train_reward",
                sliding_avg,
                sync_dist=False, 
                prog_bar=True
            )

    def on_after_backward(self):
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.log("grad_norm", total_norm, sync_dist=True)
    
    def reward_function(self, sample):
        device = next(self.parameters()).device

        # Compute ADE-based reward (ground truth trajectory available in dataset)
        if sample.get('target_trajectory') is not None:
            reward = self.train_critic.rl_pdm_score(
                sample['trajectory'], 
                sample['token'],
                target_trajectory=sample['target_trajectory']
            )
        else:
            # Fallback to PDM score if no ground truth available
            reward = self.train_critic.rl_pdm_score(sample['trajectory'], sample['token'])
        reward = torch.tensor(reward).to(device)

        # Add chain-of-thought penalty (if "need cot" is found in the generated text).
        if self.use_cot:
            cot_conf = self.cfg['rl']['cot_penalty']
            cot_penalty_coef = cot_conf['coef']
            center = cot_conf['center']
            cot_penalty_weight = cot_conf['weight']

            cot_penalties = torch.stack([
                torch.sigmoid(torch.tensor(
                    (len(text) - center) * cot_penalty_coef,
                    device=device,
                    dtype=reward.dtype
                ))
                if "complex scenario" in text.lower() else torch.tensor(
                    0.0, device=device, dtype=reward.dtype
                )
                for text in sample['completion_texts']
            ])
            reward = reward - cot_penalty_weight * cot_penalties
        else:
            cot_penalties = torch.tensor(0.0, device=device, dtype=reward.dtype)

        self.log("train_reward", reward, sync_dist=True, prog_bar=True, on_step=True, on_epoch=False)
        self.log("cot_penalty", cot_penalties.mean(), sync_dist=True, prog_bar=True, on_step=True, on_epoch=False)

        return reward
    
    def get_per_token_logps(self, model, input_ids, attention_mask, pixel_values_videos, video_grid_thw):
        # Get the per-token log probabilities for the completions for the model and the reference model
        logits = model(input_ids, attention_mask=attention_mask, 
                       pixel_values_videos=pixel_values_videos, 
                       video_grid_thw=video_grid_thw).logits  # (B, L, V)
        
        logits = logits[:, :-1, :].float()  # (B, L-1, V), exclude the last logit and compute in fp32 for stability
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it

        # Compute the log probabilities for the input tokens.
        log_probs = torch.log_softmax(logits, dim=-1)  # (B, L-1, V)
        per_token_logps = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)  # (B, L-1)
        per_token_logps = torch.nan_to_num(per_token_logps, nan=-1e4, posinf=-1e4, neginf=-1e4)
        return per_token_logps

    def generate_sample(self, data, model, device):

        # Get the model inputs
        inputs = model.get_prompt(data['input_features'])
        model_inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        # set seed
        torch.manual_seed(int(str(device).split(':')[-1]))

        # Generate completion
        with torch.no_grad():
            try:
                prompt_completion_ids = model.vlm.generate(
                    **model_inputs,
                    do_sample=True,
                    max_length=self._sample_generation_temperature['max_length'],
                    temperature=self._sample_generation_temperature['temperature'],
                    top_k=int(self._sample_generation_temperature['top_k']),
                    top_p=float(self._sample_generation_temperature['top_p']),
                    renormalize_logits=True,
                    remove_invalid_values=True,
                )
            except RuntimeError as e:
                # Fallback to greedy decoding if sampling becomes numerically unstable.
                if "probability tensor contains either `inf`, `nan` or element < 0" not in str(e):
                    raise
                prompt_completion_ids = model.vlm.generate(
                    **model_inputs,
                    do_sample=False,
                    max_length=self._sample_generation_temperature['max_length'],
                    renormalize_logits=True,
                    remove_invalid_values=True,
                )

            prompt_length = inputs.input_ids.size(1)
            prompt_mask = model_inputs['attention_mask']
            completion_ids = prompt_completion_ids[:, prompt_length:]

            # Extract action tokens and trajectory (! batch size = 1)
            actions_tokens = completion_ids[0][completion_ids[0] >= self.action_start_id]

            if len(actions_tokens) > self.trajectory_sampling.num_poses:
                actions_tokens = actions_tokens[:self.trajectory_sampling.num_poses]
            elif len(actions_tokens) < self.trajectory_sampling.num_poses:
                actions_tokens = torch.cat([actions_tokens, torch.zeros(self.trajectory_sampling.num_poses - len(actions_tokens)).to(device)])
                actions_tokens = actions_tokens.long()
            else:
                pass

            trajectory = self.autovla.action_tokenizer.decode_token_ids_to_trajectory(actions_tokens.cpu())[0, 1:]
            trajectory = Trajectory(trajectory.cpu().numpy(), self.trajectory_sampling)

            # Create completion mask
            is_eos = completion_ids == model.processor.tokenizer.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

            # Concatenate prompt_mask with completion_mask for logit computation
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1) 

            completion_texts = model.processor.batch_decode(completion_ids)

            # Create outputs
            outputs = {'trajectory': trajectory, 
                       'token': data['token'], 
                       'completion_texts': completion_texts,
                       'prompt_length': prompt_length,
                       'input_ids': prompt_completion_ids, 
                       'completion_ids': completion_ids,
                       'attention_mask': attention_mask,
                       'completion_mask': completion_mask,
                       'pixel_values_videos': model_inputs['pixel_values_videos'], 
                       'video_grid_thw': model_inputs['video_grid_thw'],
                       'target_trajectory': data.get('target_trajectory'),
                        }
        
        # clean up
        torch.cuda.empty_cache()

        return outputs
    
    def configure_optimizers(self):
        if not self._train_vision_backbone:
            for param in self.autovla.vlm.visual.parameters():
                param.requires_grad = False

        if not self._train_llm_backbone:
            for param in self.autovla.vlm.model.parameters():
                param.requires_grad = False

        params_to_update = []
        for param in self.autovla.vlm.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

        assert len(params_to_update) > 0, 'No parameters to update'

        lr = float(self.cfg['training']['learning_rate'])
        wd = float(self.cfg['training'].get('weight_decay', 0.0))
        optimizer = torch.optim.AdamW(
            params_to_update,
            lr=lr,
            weight_decay=wd
        )

        return optimizer
    
    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        # Filter out parameters with no gradient to avoid empty tensor lists
        params_with_grad = [p for p in self.parameters() if p.grad is not None]
        if params_with_grad:
            torch.nn.utils.clip_grad_value_(params_with_grad, clip_value=gradient_clip_val)

    def on_save_checkpoint(self, checkpoint: dict):
        # only save main model
        sd = checkpoint.get("state_dict", {})
        for k in list(sd):
            if k.startswith("reference_model."):
                sd.pop(k)

class SFTAutoVLA(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.cfg = config
        self.save_hyperparameters()

        self.autovla = AutoVLA(config)
        self.autovla.train()

        self._train_vision_backbone = config['model']['train_vision_backbone']
        self._train_llm_backbone = config['model']['train_lm_backbone']

    def training_step(self, batch):
        hascot = batch['has_cot']
        gt_trajectory = batch["gt_trajectory"]
        gt_action = batch["gt_action"]
        output = self.autovla(batch)

        # Compute loss in float32. Vision encoder can overflow in float16 on V100,
        # producing Inf logits → NaN loss. Clamp before casting to prevent this.
        labels = batch['labels']
        logits_f32 = torch.nan_to_num(
            output.logits, nan=0.0, posinf=1e4, neginf=-1e4
        ).float()  # (B, T, V) in float32
        shift_logits = logits_f32[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        logits_flat = shift_logits.view(-1, logits_f32.size(-1))
        labels_flat = shift_labels.view(-1)

        loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100)

        # Action token auxiliary loss (also in float32)
        action_mask = (labels_flat != -100) & (labels_flat >= self.autovla.action_start_id)
        if action_mask.any():
            action_loss = F.cross_entropy(
                logits_flat[action_mask], labels_flat[action_mask]
            )
            if hascot[0]:
                loss = loss * 40 + action_loss
            else:
                loss = loss + action_loss

        self.log("train_loss", loss.item(),
                 batch_size=gt_action.shape[0],
                 sync_dist=True,
                 prog_bar=True)

        return loss
    
    def validation_step(self, batch):
        gt_trajectory = batch["gt_trajectory"]
        gt_action = batch["gt_action"]

        labels = batch['labels']
        output = self.autovla(batch)
        logits_f32 = torch.nan_to_num(
            output.logits, nan=0.0, posinf=1e4, neginf=-1e4
        ).float()
        shift_logits = logits_f32[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, logits_f32.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        self.log("val_loss", loss.item(),
                 batch_size=gt_action.shape[0],
                 sync_dist=True, prog_bar=True)

        return loss
    
    def configure_optimizers(self):
        if not self._train_vision_backbone:
            for param in self.autovla.vlm.visual.parameters():
                param.requires_grad = False

        if not self._train_llm_backbone:
            for param in self.autovla.vlm.model.parameters():
                param.requires_grad = False

        params_to_update = []
        for param in self.autovla.vlm.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

        assert len(params_to_update) > 0, 'No parameters to update'

        optimizer = torch.optim.AdamW(
            params_to_update,
            lr=self.cfg['training']['learning_rate'],
            weight_decay=self.cfg['training'].get('weight_decay', 0.0)
        )
        lr_warmpup_step = self.cfg['training']['lr_warmup_step']
        lr_step_freq = self.cfg['training']['lr_step_frequency']
        lr_step_gamma = self.cfg['training']['lr_step_gamma']

        def lr_update(step, warmup_step, step_size, gamma):
            if step < warmup_step:
                # warm up lr
                lr_scale = 1 - (warmup_step - step) / warmup_step * 0.95
            else:
                n = (step - warmup_step) // step_size
                lr_scale = gamma ** n

            if lr_scale < 1e-2:
                lr_scale = 1e-2
            elif lr_scale > 1:
                lr_scale = 1

            return lr_scale
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: lr_update(
                step,
                lr_warmpup_step,
                lr_step_freq,
                lr_step_gamma,
            )
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    @torch.no_grad()
    def calculate_metrics(self, logits, labels, gt_trajectory):
        # Find start index for ground truth sequence
        gt_start_idx = self.find_assistant_start_idx(labels[0])
        gt_tokens = labels[0, gt_start_idx+1:] # shifted
        pred_tokens = logits[0, gt_start_idx:-1].argmax(dim=-1)

        # Find action tokens in ground truth and predicted sequences
        gt_action_idx = gt_tokens >= self.autovla.action_start_id
        pred_action_idx = pred_tokens >= self.autovla.action_start_id

        if len(pred_tokens[pred_action_idx]) != len(gt_tokens[gt_action_idx]):
            pred_action_idx = gt_action_idx
            
        gt_action_tokens = gt_tokens[gt_action_idx]
        pred_action_tokens = pred_tokens[pred_action_idx]

        # Decode predicted trajectory
        # pred_trajectory = self.autovla.action_tokenizer.decode_token_ids_to_trajectory(pred_action_tokens.cpu())
        # action_acc = (pred_action_tokens == gt_action_tokens).float().mean()
        # traj_mse = torch.norm(pred_trajectory[0, 1:, :2] - gt_trajectory[0].cpu(), dim=-1).mean()
        # traj_mse = traj_mse.to(logits.device)

        # return {
        #     'action_acc': action_acc,
        #     'traj_mse': traj_mse
        # }
    
    @staticmethod
    def find_assistant_start_idx(labels):
        assistant_id = torch.tensor(ASSISTANT_ID).to(labels.device)
        
        for j in range(len(labels) - len(assistant_id) + 1):
            if torch.equal(labels[j:j + len(assistant_id)], assistant_id):
                start_idx = j
                break

        return start_idx


class AutoVLA(torch.nn.Module):
    def __init__(self, config, inference=False, device='cpu'):
        super().__init__()
        self.device = device

        model_path = config['model']['pretrained_model_path']
        dtype_name = os.environ.get("AUTOVLA_TORCH_DTYPE", "float16").lower()
        torch_dtype = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }.get(dtype_name, torch.bfloat16)
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device,
        }
        attn_implementation = os.environ.get("AUTOVLA_ATTN_IMPLEMENTATION")
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            **model_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer, 
                                                model_config=config['model'])
        self.vlm.resize_token_embeddings(len(self.processor.tokenizer))

        self.video_conf = config['model']['video']
        self.action_start_id = config['model']['tokens']['action_start_id']

        self.use_cot = config['model']['use_cot']
        self.traj_num_poses = int(config['model']['trajectory'].get('num_poses', 10))
        inference_conf = config.get('inference', {})
        sample_conf = dict(inference_conf.get('sample', {}))
        if 'max_length' not in sample_conf:
            sample_conf['max_length'] = inference_conf.get(
                'max_length',
                config.get('training', {}).get('sample', {}).get('max_length', 2048)
            )
        sample_conf.setdefault('min_new_tokens', 64)
        sample_conf.setdefault('temperature', 0.2)
        sample_conf.setdefault('top_k', 0)
        sample_conf.setdefault('top_p', 1.0)
        sample_conf.setdefault('do_sample', True)
        self.gen_conf = sample_conf

    def predict(self, input_features):
        torch.backends.cudnn.benchmark = True  # V100: prevents conv3d resource error
        inputs = self.get_prompt(input_features)
        model_inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        do_sample = bool(self.gen_conf.get('do_sample', False))
        generate_kwargs = {
            'max_length': int(self.gen_conf['max_length']),
            'do_sample': do_sample,
            'min_new_tokens': int(self.gen_conf.get('min_new_tokens', 64)),
        }
        if do_sample:
            generate_kwargs.update({
                'temperature': float(self.gen_conf['temperature']),
                'top_k': int(self.gen_conf['top_k']),
                'top_p': float(self.gen_conf['top_p']),
            })

        outputs = self.vlm.generate(
            **model_inputs,
            **generate_kwargs,
        )

        outputs_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]

        outputs_trimmed = outputs_trimmed[0][:-1].cpu() # remove end token
        cot_results = self.processor.decode(outputs_trimmed)
        
        # DEBUG: Always log action token extraction to find issue
        if outputs_trimmed.numel() > 0:
            print(f"[EVAL_DEBUG] outputs_trimmed len={len(outputs_trimmed)}, range=[{outputs_trimmed.min().item()}, {outputs_trimmed.max().item()}]")
        else:
            print("[EVAL_DEBUG] outputs_trimmed is empty")
        print(f"[EVAL_DEBUG] action_start_id={self.action_start_id}")
        print(f"[EVAL_DEBUG] first 50 tokens: {outputs_trimmed[:50].tolist() if outputs_trimmed.numel() > 0 else []}")

        decoded_action_matches = re.findall(r"<action_(\d+)>", cot_results)
        if decoded_action_matches:
            actions_tokens = torch.tensor([int(match) + self.action_start_id for match in decoded_action_matches], device=outputs_trimmed.device)
            print(f"[EVAL_DEBUG] valid actions found via text: {len(actions_tokens)}")
        else:
            actions_tokens = outputs_trimmed[outputs_trimmed >= self.action_start_id]
            print(f"[EVAL_DEBUG] valid actions found via token ids: {len(actions_tokens)}")

        if actions_tokens.numel() == 0:
            with torch.no_grad():
                fallback_mode = os.environ.get("AUTOVLA_ACTION_FALLBACK_MODE", "topk").lower()
                if fallback_mode in {"autoregressive", "constrained", "constrained_autoregressive"}:
                    generated_action_ids = []
                    action_context = outputs
                    for _ in range(self.traj_num_poses):
                        model_outputs = self.vlm(
                            input_ids=action_context,
                            attention_mask=torch.ones_like(action_context),
                            pixel_values_videos=model_inputs.get('pixel_values_videos'),
                            video_grid_thw=model_inputs.get('video_grid_thw'),
                        )
                        last_logits = model_outputs.logits[0, -1]
                        action_logits = last_logits[self.action_start_id:self.action_start_id + self.action_tokenizer.n_bins]
                        if action_logits.numel() == 0:
                            break
                        next_action_id = torch.argmax(action_logits).to(action_context.device) + self.action_start_id
                        generated_action_ids.append(next_action_id)
                        action_context = torch.cat([action_context, next_action_id.view(1, 1)], dim=1)
                    if generated_action_ids:
                        actions_tokens = torch.stack(generated_action_ids).to(outputs_trimmed.device)
                        print(f"[EVAL_DEBUG] valid actions found via constrained autoregressive fallback: {len(actions_tokens)}")
                else:
                    model_outputs = self.vlm(
                        input_ids=outputs,
                        attention_mask=torch.ones_like(outputs),
                        pixel_values_videos=model_inputs.get('pixel_values_videos'),
                        video_grid_thw=model_inputs.get('video_grid_thw'),
                    )
                    last_logits = model_outputs.logits[0, -1]
                    action_logits = last_logits[self.action_start_id:self.action_start_id + self.action_tokenizer.n_bins]
                    top_k = min(self.traj_num_poses, action_logits.shape[-1])
                    if top_k > 0:
                        fallback_action_ids = torch.topk(action_logits, k=top_k).indices + self.action_start_id
                        actions_tokens = fallback_action_ids.to(outputs_trimmed.device)
                        print(f"[EVAL_DEBUG] valid actions found via logits fallback: {len(actions_tokens)}")

        if actions_tokens.numel() == 0:
            return torch.empty((0, 3)), cot_results

        trajectory = self.action_tokenizer.decode_token_ids_to_trajectory(actions_tokens)[0, 1:]

        return trajectory, cot_results
    
    def get_prompt(self, input_features, image_mode="video"):
        # image sensor
        images = input_features['images']

        min_pixels = self.video_conf.get("min_pixels", 28 * 28 * 128)
        max_pixels = self.video_conf.get("max_pixels", 28 * 28 * 128)

        camera_images = {}
        
        # List of camera types to load
        camera_types = ['front_camera', 'front_left_camera', 'front_right_camera']
        
        if input_features['sensor_data_path']:
            for camera_type in camera_types:
                camera_images[camera_type] = []
                for i in range(4):
                    img = images[camera_type][i]
                    camera_images[camera_type].append(
                        os.path.join(input_features['sensor_data_path'], img))

        # Assign to individual variables for message formatting
        front_camera_1, front_camera_2, front_camera_3, front_camera_4 = camera_images['front_camera']
        front_left_camera_1, front_left_camera_2, front_left_camera_3, front_left_camera_4 = camera_images['front_left_camera']
        front_right_camera_1, front_right_camera_2, front_right_camera_3, front_right_camera_4 = camera_images['front_right_camera']


        # vehicle state
        velocity = input_features["vehicle_velocity"]

        if isinstance(velocity, list) or isinstance(velocity, np.ndarray):
            velocity_x = velocity[0]
            velocity_y = velocity[1]
            velocity = np.sqrt(velocity_x**2 + velocity_y**2)
    
        acceleration = input_features["vehicle_acceleration"]
        if isinstance(acceleration, list) or isinstance(acceleration, np.ndarray):
            acceleration_x = acceleration[0]
            acceleration_y = acceleration[1]
            acceleration = np.sqrt(acceleration_x**2 + acceleration_y**2)

        instruction = input_features["driving_command"].lower()
    
        user_content = [
            {
                "type": "text",
                "text": (
                    "The autonomous vehicle is equipped with three cameras mounted at the front, left, and right, enabling a comprehensive perception of the surrounding environment."
                )
            },
            {
                "type": "text",
                "text": "The first video presents the front view of the vehicle, comprising four sequential frames sampled at 2 Hz."
            },
            {
                "type": "video",
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "video": [
                    f"file://{front_camera_1}",
                    f"file://{front_camera_2}",
                    f"file://{front_camera_3}",
                    f"file://{front_camera_4}",
                ]
            },
            {
                "type": "text",
                "text": "The second video presents the front-left view of the vehicle, comprising four sequential frames sampled at 2 Hz."
            },
            {
                "type": "video",
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "video": [
                    f"file://{front_left_camera_1}",
                    f"file://{front_left_camera_2}",
                    f"file://{front_left_camera_3}",
                    f"file://{front_left_camera_4}",
                ]
            },
            {
                "type": "text",
                "text": "The third video presents the front-right view of the vehicle, comprising four sequential frames sampled at 2 Hz."
            },
            {
                "type": "video",
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "video": [
                    f"file://{front_right_camera_1}",
                    f"file://{front_right_camera_2}",
                    f"file://{front_right_camera_3}",
                    f"file://{front_right_camera_4}",
                ]
            },
            {
                "type": "text",
                "text": (
                    f"The current velocity of the vehicle is {velocity:.3f} m/s, and the current acceleration is {acceleration:.3f} m/s². "
                    f"The driving instruction is: {instruction}. Based on this information, plan the action trajectory for the autonomous vehicle over the next five seconds."
                )
            },
        ]

        if self.use_cot:
            messages = [
                {   
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text":
                            "You are an Advanced Driver Assistance and Full Self-Driving System. "
                            "You will receive visual observations from the ego vehicle's cameras and dynamic information about the vehicle's current state. "
                            "Your task is to predict the optimal driving action for the next five seconds.\n\n"
                            "First, carefully analyze the surrounding environment by considering traffic lights, the movements of other vehicles and pedestrians, lane markings, and any other relevant factors.\n\n"
                            "If necessary, use step-by-step reasoning (Chain-of-Thought) to arrive at the best driving action. Otherwise, you may directly predict the final driving action.\n\n"
                            "Present the final action clearly after your reasoning steps.\n"
                            "In the <answer> section, output action trajectory tokens in this exact prefix format: "
                            "The final output action is: <action_...>"
                        }
                    ]
                },

                {
                    "role": "user",
                    "content": user_content
                },
            ]
        else:
            messages = [
                {   
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text":
                            "You are an Advanced Driver Assistance and Full Self-Driving System. "
                            "You will be provided with video observations from the ego vehicle's surrounding cameras, along with the vehicle's current dynamic states. "
                            "Your task is to predict the most appropriate driving action for the next five seconds.\n"
                            "Output action trajectory tokens using this exact prefix format: "
                            "The final output action is: <action_...>"
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": user_content
                },
            ]

        image_inputs, video_inputs = process_vision_info(messages)
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
        )
        text += "<answer>\nThe final output action is: "

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        return inputs
    
    def forward(self, inputs):
        inputs.pop('gt_trajectory')
        inputs.pop('gt_action')
        inputs.pop('has_cot')
        outputs: CausalLMOutputWithPast = self.vlm(**inputs)

        return outputs
