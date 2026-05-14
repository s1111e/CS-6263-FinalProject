# Action-Token Fallback Audit

Source log: `logs/12_evaluate.out`

## Baseline Evaluation Results (5,569 validation samples)

| Metric | Value |
| --- | ---: |
| Samples evaluated | 5,569 |
| Direct `<action_*>` tokens from decoded text | 0 (0.0%) |
| Direct action tokens from generated token ids | 0 (0.0%) |
| Logits fallback used | 5,569 (100.0%) |

The baseline SFT checkpoint generates natural-language output ("keep going straight", "turn_right") and relies on logit-based fallback to select action tokens. This is the primary reason the baseline L2 values are higher than the paper.

## Resolution: Two-Phase Repair

The two-phase action-token repair (`multimodal_action_repair.py`) resolves this. After 20 training steps:

| Metric | Before repair | After repair |
| --- | ---: | ---: |
| Direct action token generation rate | 0.0% | 100.0% |
| UniAD L2@1s (200-sample eval) | 6.29 m | 5.55 m |

The repair consists of:
1. **Phase 1** — Text-only warm-up (1 epoch): re-establishes the action-token output mode
2. **Phase 2** — Multimodal fine-tuning (20 steps): transfers direct generation to the full evaluation prompt

See `evaluation_results/table_s2_step20.json` for the full repair evaluation results.

## Token Distribution (baseline)

```json
{
  "text": {},
  "token_ids": { "0": 5569 },
  "logits_fallback": { "10": 5569 }
}
```

All 5,569 samples used the logits fallback path (10 action tokens each). No samples produced direct `<action_*>` tokens in the text or token-id output of the baseline checkpoint.
