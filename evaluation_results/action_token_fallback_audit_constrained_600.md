# Action-Token Fallback Audit

Source log: `autovla-nuscenes-reproduction/evaluation_results/table_s2_constrained_action_600.log`

| Metric | Value |
| --- | ---: |
| Samples observed | 600 |
| Direct action tokens from decoded text | 0 (0.0%) |
| Direct action tokens from generated token ids | 0 (0.0%) |
| Logits fallback used | 0 (0.0%) |
| Constrained autoregressive fallback used | 600 (100.0%) |
| No action recovered estimate | 0 |

## Event Count Distributions

These distributions show how many action tokens were found by each extraction path.

```json
{
  "text": {},
  "token_ids": {
    "0": 600
  },
  "logits_fallback": {},
  "constrained_autoregressive_fallback": {
    "10": 600
  }
}
```

Interpretation: a high logits-fallback rate means the model did not directly generate clean `<action_*>` tokens in the normal output sequence. This is a likely source of the Table S2 reproduction gap.
