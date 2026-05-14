# Project Guide — AutoVLA NuScenes Reproduction

Quick-reference for code structure, execution order, and requirements.
For detailed analysis see [results.md](results.md). For the visual report see [index.html](index.html).

---

## Requirements

**Hardware**
- Training: 4× NVIDIA V100 (32 GB each), SLURM cluster
- Evaluation / repair: 1× V100
- RAM: ≥ 64 GB

**Software**
```bash
conda env create -f environment.yml              # main env (autovla)
conda env create -f environment_nusc_preprocess.yml  # NuScenes preprocessing env
pip install -e .                                 # install AutoVLA package
cd navsim && pip install -e . && cd ..           # install NavSim package
```

**Data**
- NuScenes full dataset (trainval, ~300 GB) → `data/nuscenes/`
- Processed records → `data/nuscenes_processed/`  (produced by slurm/09)
- Evaluation segments → `data/nusc_eval_seg/nusc_eval_seg_6s/`

**Model weights**
```bash
bash scripts/download_qwen.sh   # downloads Qwen2.5-VL-3B-Instruct
```

---

## Project Tree

```
autovla-nuscenes-reproduction/
│
├── models/
│   ├── autovla.py              # Main model class (SFTAutoVLA, PyTorch Lightning)
│   ├── action_tokenizer.py     # K=2048 codebook, manages <action_0>…<action_2047>
│   └── utils/score.py          # L2 distance and collision metric helpers
│
├── dataset_utils/
│   ├── sft_dataset.py          # SFT dataset (CoT and no-CoT modes)
│   └── rft_dataset.py          # RFT / GRPO dataset
│
├── navsim/                     # NavSim library (copied package)
│   ├── common/dataloader.py
│   ├── agents/autovla_agent.py
│   └── evaluate/
│
├── tools/
│   ├── run_sft.py              # Launches SFT training (FSDP, 4-GPU)
│   ├── run_rft.py              # Launches RFT / GRPO training
│   ├── eval/
│   │   ├── nusc_eval.py        # NuScenes L2 + collision evaluation engine
│   │   └── planning_metrics.py
│   └── preprocessing/
│       ├── nusc_sample_generation.py   # raw NuScenes → JSON records
│       ├── cot_sample_generation.py    # adds chain-of-thought annotations
│       └── nocot_sample_generation.py  # no-CoT variant
│
├── config/
│   ├── training/
│   │   ├── qwen2.5-vl-3B-mix-sft.yaml        # main SFT config (used)
│   │   └── qwen2.5-vl-3B-nusc-nocot-sft.yaml # no-CoT alternative
│   └── eval/
│       └── qwen2.5-vl-3B-nusc-sft-eval.yaml  # evaluation config
│
├── slurm/                      # SLURM job scripts (run in order)
│   ├── 01_create_env.slurm
│   ├── 02_install_packages.slurm
│   ├── 03_download_models.slurm
│   ├── 04_check_setup.slurm
│   ├── 06_preprocess_data.slurm         # Waymo preprocessing (optional)
│   ├── 08_setup_nusc_preprocess_env.slurm
│   ├── 09_preprocess_nuscenes.slurm
│   ├── 10_sft_training_full_gpu4v100.slurm
│   ├── 11_rft_training.slurm
│   ├── 12_evaluate.slurm
│   ├── 13_reproduce_table_s2.slurm
│   ├── 14_reproduce_table_2.slurm
│   ├── 15_reproduce_quicktest.slurm
│   ├── 16_reproduce_figure_s6.slurm
│   └── 19_reproduce_paper_tables_nuscenes.slurm
│
├── scripts/
│   ├── download_qwen.sh
│   ├── run_sft.sh                 # thin wrapper around tools/run_sft.py
│   ├── run_rft.sh
│   ├── run_nuscenes_preprocessing.sh
│   └── generate_all_tables.py     # orchestrates all reproduction steps
│
├── codebook_cache/             # K=2048 codebook cache (prebuilt, included in git)
├── images/                     # logos and figures
├── logs/                       # SLURM stdout/stderr (git-ignored)
│
├── evaluation_results/
│   ├── table_s2_baseline.json  # L2 / collision before repair
│   ├── table_s2_step20.json    # L2 / collision after repair
│   ├── table_2_runtime.json    # fast vs slow thinking latency
│   ├── paper_tables_nuscenes.json
│   └── multimodal_repair_v2/
│       └── metrics.json        # repair training history (before/after action rate)
│
│  ── Reproduction scripts ──
│
├── reproduce_table_s2_nuscenes.py     # evaluates model → L2 + collision JSON
├── reproduce_table_2_runtime.py       # measures inference latency → JSON
├── reproduce_figure_s6_qualitative.py # generates qualitative comparison PNG
├── reproduce_paper_tables_nuscenes.py # collects paper numbers into JSON
├── multimodal_action_repair.py        # proposed improvement: two-phase repair
├── print_results.py                   # prints all tables from JSON (no GPU needed)
│
│  ── Diagnostic / debug scripts ──
│
├── audit_action_token_mapping.py      # checks tokenizer mapping is correct
├── audit_action_token_labels.py       # checks action labels reach the dataset
├── audit_action_token_fallback.py     # detects logit-fallback usage in eval
├── audit_action_token_grad_flow.py    # checks gradients flow to action embeddings
├── audit_multimodal_action_grad_flow.py
└── action_token_only_mini_sft.py      # text-only overfit test (no images)
```

---

## Execution Order

### Step 1 — Environment setup
```bash
sbatch slurm/01_create_env.slurm
sbatch slurm/02_install_packages.slurm
sbatch slurm/03_download_models.slurm
sbatch slurm/04_check_setup.slurm
```

### Step 2 — Data preprocessing
```bash
sbatch slurm/08_setup_nusc_preprocess_env.slurm
sbatch slurm/09_preprocess_nuscenes.slurm   # → data/nuscenes_processed/
```

### Step 3 — Training
```bash
sbatch slurm/10_sft_training_full_gpu4v100.slurm
# output: runs/sft/<date>/epoch=N-loss=X.ckpt  (~48 hours, 4×V100)
```

### Step 4 — Reproduce paper results
```bash
# All at once:
python scripts/generate_all_tables.py \
    --config config/training/qwen2.5-vl-3B-mix-sft.yaml \
    --checkpoint runs/sft/2026-04-22_08-18-13/epoch=4-loss=1.7191.ckpt \
    --seg-data data/nusc_eval_seg/nusc_eval_seg_6s

# Or individually:
python reproduce_table_s2_nuscenes.py --config ... --checkpoint ...
python reproduce_table_2_runtime.py   --config ... --checkpoint ...
python reproduce_figure_s6_qualitative.py --config ... --checkpoint ...
```

### Step 5 — Proposed improvement (action-token repair)
```bash
python multimodal_action_repair.py \
    --config config/training/qwen2.5-vl-3B-mix-sft.yaml \
    --base-checkpoint runs/sft/.../epoch=4-loss=1.7191.ckpt \
    --device cuda:0
```

### Step 6 — View results (no GPU needed)
```bash
python print_results.py
```

---

## Script Reference

| Script | Category | What it does | GPU |
|--------|----------|-------------|-----|
| `tools/run_sft.py` | Training | SFT training with FSDP | 4× |
| `tools/run_rft.py` | Training | RFT / GRPO training | 4× |
| `reproduce_table_s2_nuscenes.py` | Reproduction | L2@1s, L2@2s, collision rate | 1× |
| `reproduce_table_2_runtime.py` | Reproduction | Fast vs slow thinking latency | 1× |
| `reproduce_figure_s6_qualitative.py` | Reproduction | Qualitative comparison PNG | 1× |
| `reproduce_paper_tables_nuscenes.py` | Reproduction | Collects paper numbers to JSON | ✗ |
| `scripts/generate_all_tables.py` | Orchestrator | Runs all reproduction steps in order | 1× |
| `multimodal_action_repair.py` | Improvement | Two-phase action-token fine-tuning | 1× |
| `print_results.py` | Viewer | Prints tables from JSON files | ✗ |
| `audit_action_token_mapping.py` | Diagnostic | Validates tokenizer mapping | ✗ |
| `audit_action_token_labels.py` | Diagnostic | Checks action labels in dataset | ✗ |
| `audit_action_token_fallback.py` | Diagnostic | Detects fallback mechanism usage | ✗ |
| `audit_action_token_grad_flow.py` | Diagnostic | Checks gradient flow to embeddings | 1× |
| `audit_multimodal_action_grad_flow.py` | Diagnostic | Multimodal gradient flow check | 1× |
| `action_token_only_mini_sft.py` | Diagnostic | Text-only overfit sanity test | 1× |

---

## Key Parameters

| Parameter | Value |
|-----------|-------|
| Base model | Qwen2.5-VL-3B-Instruct |
| Action codebook size | K = 2048 |
| Action token ID range | 151665 – 153712 |
| SFT batch size | 4 (1 per GPU) |
| SFT learning rate | 1e-4 |
| Training epochs | 5 |
| Slow thinking latency | ~6 s / sample |
| Fast thinking latency | ~0.4 s / sample |

---

## V100 Compatibility Notes

These fixes were needed to run on V100 GPUs (not required on A100/H100):

- `torch.backends.cudnn.benchmark = True` — prevents float16 Conv3d overflow
- `attn_implementation = "eager"` — Flash Attention is not supported on V100
- FSDP `reduce_dtype = float32` — keeps gradient reduction stable
- `bfloat16` is not supported on V100; use `float16`
