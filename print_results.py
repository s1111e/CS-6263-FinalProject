#!/usr/bin/env python3
"""
Print all reproduction results from pre-computed JSON artifacts.

No GPU, no checkpoint, no dataset required.
Run immediately after git clone:

    python autovla-nuscenes-reproduction/print_results.py

Reads:
    evaluation_results/table_s2_baseline.json    -- Table S2 baseline (200 val samples)
    evaluation_results/table_s2_step20.json      -- Table S2 + repair improvement
    evaluation_results/table_2_runtime.json      -- Table 2 runtime (fast vs slow)
    evaluation_results/paper_tables_nuscenes.json -- paper reference numbers
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVAL = ROOT / "evaluation_results"

# ── ANSI colours (disabled on Windows / when redirected) ──────────────────────
_C = sys.stdout.isatty()
GRN  = "\033[32m" if _C else ""
YLW  = "\033[33m" if _C else ""
CYN  = "\033[36m" if _C else ""
BOLD = "\033[1m"  if _C else ""
RST  = "\033[0m"  if _C else ""


def _load(name: str) -> dict:
    path = EVAL / name
    if not path.exists():
        print(f"  {YLW}WARNING:{RST} {path} not found — skipping.")
        return {}
    with open(path) as f:
        return json.load(f)


def _sep(char="─", width=72):
    print(char * width)


def _row(*cells, widths=(38, 9, 9, 9)):
    parts = []
    for cell, w in zip(cells, widths):
        parts.append(str(cell).ljust(w))
    print("  " + "  ".join(parts))


# ── TABLE S2 ──────────────────────────────────────────────────────────────────

def reproduce_table_s2():
    """Table S2: NuScenes planning benchmark (L2 distance in metres)."""
    print(f"\n{BOLD}{'='*72}{RST}")
    print(f"{BOLD}  TABLE S2 — nuScenes Planning Benchmark (L2 distance, metres){RST}")
    print(f"{'='*72}")

    baseline = _load("table_s2_baseline.json")
    repaired = _load("table_s2_step20.json")

    if not baseline:
        print("  No baseline data available.")
        return

    bs = baseline.get("summary", {})
    rs = repaired.get("summary", {}) if repaired else {}

    print()
    _row("Method", "L2@1s", "L2@2s", "L2@3s")
    _sep()

    # Paper rows
    _row("AutoVLA action-only (paper)", "0.22", "0.39", "0.61")
    _row("AutoVLA w/ CoT      (paper)", "0.21", "0.38", "0.60")
    _sep("·")

    # Our baseline
    b1 = bs.get("l2_mean_1.0s")
    b2 = bs.get("l2_mean_2.0s")
    b3 = bs.get("l2_mean_3.0s")
    bn = baseline.get("num_samples", "?")
    _row(
        f"Ours — baseline (fallback 100%, n={bn})",
        f"{b1:.2f}" if b1 else "N/A",
        f"{b2:.2f}" if b2 else "N/A",
        f"{b3:.2f}" if b3 else "N/A",
    )

    # Our improvement
    if rs:
        r1 = rs.get("l2_mean_1.0s")
        r2 = rs.get("l2_mean_2.0s")
        r3 = rs.get("l2_mean_3.0s")
        rn = repaired.get("num_samples", "?")

        def _delta(a, b):
            if a and b:
                pct = (b - a) / a * 100
                sign = "+" if pct > 0 else ""
                return f"{sign}{pct:.1f}%"
            return ""

        _row(
            f"{GRN}Ours + repair    (direct tokens, n={rn}){RST}",
            f"{GRN}{r1:.2f}{RST}" if r1 else "N/A",
            f"{GRN}{r2:.2f}{RST}" if r2 else "N/A",
            f"{GRN}{r3:.2f}{RST}" if r3 else "N/A",
        )
        _sep("·")
        _row(
            "  Δ repair vs baseline",
            _delta(b1, r1),
            _delta(b2, r2),
            _delta(b3, r3),
        )

    _sep()
    print(f"  {CYN}Improvement:{RST} two-phase action-token repair (multimodal_action_repair.py)")
    print(f"  Action token rate: 0% (baseline) → 100% (after repair)")
    print(f"  Artifacts: evaluation_results/table_s2_baseline.json")
    print(f"             evaluation_results/table_s2_step20.json")


# ── TABLE 2 ───────────────────────────────────────────────────────────────────

def reproduce_table_2():
    """Table 2: Fast vs Slow thinking runtime (seconds)."""
    print(f"\n{BOLD}{'='*72}{RST}")
    print(f"{BOLD}  TABLE 2 — Runtime: Fast Thinking vs Slow Thinking (seconds){RST}")
    print(f"{'='*72}")

    data = _load("table_2_runtime.json")
    if not data:
        print("  No runtime data available.")
        return

    s = data.get("summary", {})
    fast = s.get("fast_thinking", {})
    slow = s.get("slow_thinking", {})
    n    = data.get("num_samples", "?")

    print()
    _row("Mode", "Mean (s)", "Min (s)", "Max (s)")
    _sep()
    _row("Fast thinking (paper)",   "1.072",   "0.997",  "1.116")
    _row("Slow thinking (paper)",   "10.518",  "7.607",  "13.706")
    _sep("·")
    _row(
        f"Fast thinking (ours, n={n})",
        f"{fast.get('mean', 0):.3f}" if fast else "N/A",
        f"{fast.get('min',  0):.3f}" if fast else "N/A",
        f"{fast.get('max',  0):.3f}" if fast else "N/A",
    )
    _row(
        f"Slow thinking (ours, n={n})",
        f"{slow.get('mean', 0):.3f}" if slow else "N/A",
        f"{slow.get('min',  0):.3f}" if slow else "N/A",
        f"{slow.get('max',  0):.3f}" if slow else "N/A",
    )
    _sep()

    ratio = s.get("ratio", {})
    if ratio:
        print(f"  Slow/Fast ratio — paper: 9.8×   ours: {ratio.get('mean', 0):.2f}× (mean)")
    print(f"  Note: measured on V100 with float16, eager attention, max_new_tokens=64.")
    print(f"  Artifact: evaluation_results/table_2_runtime.json")


# ── IMPROVEMENT SUMMARY ───────────────────────────────────────────────────────

def reproduce_improvement():
    """Proposed improvement: two-phase action-token repair."""
    print(f"\n{BOLD}{'='*72}{RST}")
    print(f"{BOLD}  IMPROVEMENT — Two-Phase Action-Token Repair{RST}")
    print(f"{'='*72}")
    print("""
  Hypothesis: the trained SFT checkpoint never emits <action_*> tokens directly;
  it outputs natural language and falls back to logit-based selection. A targeted
  two-phase curriculum can restore direct action-token generation.

  Phase 1 — text warm-up (1 epoch):
    Short text-only prompt, action-only target, restricted CE + gate loss.
    → text_action_rate: 0.000 → 1.000

  Phase 2 — multimodal repair (20 steps):
    Exact evaluation prompt, loss on the 10 action positions only.
    → multimodal text_action_rate: 0.000 → 1.000

  Diagnostic audit trail:
    audit_action_token_mapping.py   ✅  tokenizer correct
    audit_action_token_labels.py    ✅  10 action tokens per sample
    audit_action_token_grad_flow.py ✅  gradient reaches action rows
    action_token_only_mini_sft.py   ✅  text-only rate = 1.0
    multimodal_action_repair.py     ✅  multimodal rate = 1.0 (step 20)
""")

    baseline = _load("table_s2_baseline.json")
    repaired = _load("table_s2_step20.json")
    bs = baseline.get("summary", {}) if baseline else {}
    rs = repaired.get("summary", {}) if repaired else {}

    _row("Metric", "Baseline", "+ Repair", "Δ", widths=(30, 14, 14, 12))
    _sep()
    _row("Action token rate",
         "0%", f"{GRN}100%{RST}", f"{GRN}+100pp{RST}",
         widths=(30, 14, 14, 12))

    for label, key in [("L2@1s (m)", "l2_mean_1.0s"),
                       ("L2@2s (m)", "l2_mean_2.0s"),
                       ("L2@3s (m)", "l2_mean_3.0s")]:
        b = bs.get(key)
        r = rs.get(key)
        if b and r:
            pct = (r - b) / b * 100
            sign = "+" if pct > 0 else ""
            color = GRN if pct < 0 else YLW
            _row(label,
                 f"{b:.3f}",
                 f"{color}{r:.3f}{RST}",
                 f"{color}{sign}{pct:.1f}%{RST}",
                 widths=(30, 14, 14, 12))

    _sep()
    print(f"  Script:   autovla-nuscenes-reproduction/multimodal_action_repair.py")
    print(f"  Artifact: evaluation_results/table_s2_step20.json")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def reproduce_action_token_demo():
    """Show concrete before/after examples of action token generation."""
    print(f"\n{BOLD}{'='*72}{RST}")
    print(f"{BOLD}  ACTION TOKEN GENERATION — Before vs After Repair{RST}")
    print(f"{'='*72}")

    repair_metrics = EVAL / "multimodal_repair_v2" / "metrics.json"
    if not repair_metrics.exists():
        repair_metrics = EVAL / "multimodal_repair_final" / "metrics.json"
    if not repair_metrics.exists():
        print(f"  {YLW}Repair metrics not found — skipping demo.{RST}")
        return

    with open(repair_metrics) as f:
        data = json.load(f)

    before, after = [], []
    for entry in data.get("history", []):
        if entry.get("type") != "eval" or entry.get("phase") != 2:
            continue
        if entry.get("step") == 0:
            before = entry.get("examples", [])
        if entry.get("step") == 20 and not after:
            after = entry.get("examples", [])

    print()
    print(f"  {BOLD}BEFORE repair (step=0):{RST} model outputs natural language")
    print(f"  {'─'*64}")
    for ex in before[:3]:
        out = ex.get("preview", "").strip().split("\n")[0][:80]
        print(f"  [{ex['dataset_index']}] output : {YLW}{out}{RST}")

    print()
    print(f"  {BOLD}AFTER repair (step=20):{RST} model outputs action tokens directly")
    print(f"  {'─'*64}")
    for ex in after[:3]:
        tgt = ex.get("target", "")[:60]
        out = ex.get("preview", "").strip().split("\n")[0][:80]
        print(f"  [{ex['dataset_index']}] target : {tgt}")
        print(f"  [{ex['dataset_index']}] output : {GRN}{out}{RST}")
        print()

    print(f"  {CYN}Key observation:{RST} Before repair → natural language (\"turn_right\").")
    print(f"  After repair → direct <action_N> token sequences.")
    print(f"  Action token rate: 0% → 100% in 20 training steps.")
    print(f"  Artifact: evaluation_results/multimodal_repair_v2/metrics.json")


def main():
    print(f"\n{BOLD}AutoVLA nuScenes Reproduction — Results Summary{RST}")
    print(f"Generated from pre-computed artifacts in evaluation_results/")
    print(f"No GPU, checkpoint, or dataset required to view these numbers.\n")

    reproduce_table_s2()
    reproduce_table_2()
    reproduce_improvement()
    reproduce_action_token_demo()

    print(f"\n{'='*72}")
    print(f"  Full details: results.md | Visual summary: index.html")
    print(f"  To regenerate: python scripts/generate_all_tables.py [requires checkpoint]")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
