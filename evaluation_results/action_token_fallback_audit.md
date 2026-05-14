# Action-Token Fallback Audit

Source log: `logs/12_evaluate.out`

| Metric | Value |
| --- | ---: |
| Samples observed | 5569 |
| Direct action tokens from decoded text | 0 (0.0%) |
| Direct action tokens from generated token ids | 0 (0.0%) |
| Logits fallback used | 5569 (100.0%) |
| Constrained autoregressive fallback used | 0 (0.0%) |
| No action recovered estimate | 0 |

## Event Count Distributions

These distributions show how many action tokens were found by each extraction path.

```json
{
  "text": {},
  "token_ids": {
    "0": 5569
  },
  "logits_fallback": {
    "10": 5569
  }
}
```

Interpretation: a high logits-fallback rate means the model did not directly generate clean `<action_*>` tokens in the normal output sequence. This is a likely source of the Table S2 reproduction gap.
