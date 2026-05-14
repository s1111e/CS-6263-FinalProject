# Action-Token Label Audit

Config: `config/training/qwen2.5-vl-3B-mix-sft.yaml`
Split: `train`
Action start id: `151665`

| Sample | Raw `<action_*>` | Input action ids | Label action ids | Visible label tokens |
|---:|---:|---:|---:|---:|
| 0 | 10 | 10 | 10 | 298 |
| 1 | 10 | 10 | 10 | 52 |
| 2 | 10 | 10 | 10 | 52 |
| 3 | 10 | 10 | 10 | 52 |
| 4 | 10 | 10 | 10 | 52 |

## Interpretation

If `Label action ids` is zero, the supervised loss cannot teach direct action-token generation.
If it is non-zero, action tokens reach the labels and the problem is likely model/training/generation strength rather than collator masking.

## First Sample Preview

```text
ationary. The moving status of **Person wearing a blue shirt** is keep going straight. The moving status of **White and red truck** is stationary.

### Reasoning on Intent:
Firstly notice **A letter AHEAD on the ground**. It is a traffic sign, so the ego vehicle should continue at the same speed. Secondly notice **White commercial vehicle**. It is stationary, so the ego vehicle should continue at the same speed. Thirdly notice **White and red truck**. It is also stationary, so the ego vehicle should continue at the same speed.

### Best Driving Action:
turn right with a deceleration
</think>
<answer>
The final output action is: <action_828><action_419><action_700><action_700><action_1559><action_828><action_72><action_641><action_166><action_1324>
</answer><|im_end|>
<|im_start|>assistant

```
