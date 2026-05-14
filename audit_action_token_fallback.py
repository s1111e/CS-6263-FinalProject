#!/usr/bin/env python3
"""Audit action-token extraction and logits fallback usage from evaluation logs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


EVENT_PATTERNS = {
    "text": re.compile(r"valid actions found via text:\s*(\d+)"),
    "token_ids": re.compile(r"valid actions found via token ids:\s*(\d+)"),
    "logits_fallback": re.compile(r"valid actions found via logits fallback:\s*(\d+)"),
    "constrained_autoregressive_fallback": re.compile(r"valid actions found via constrained autoregressive fallback:\s*(\d+)"),
}


def pct(value: int, total: int) -> float:
    return round((100.0 * value / total), 2) if total else 0.0


def counter_dict(values: Iterable[int]) -> Dict[str, int]:
    return {str(k): v for k, v in sorted(Counter(values).items())}


def audit_log(log_path: Path) -> Dict[str, object]:
    text = log_path.read_text(errors="replace")
    event_counts: Dict[str, List[int]] = {
        name: [int(match) for match in pattern.findall(text)]
        for name, pattern in EVENT_PATTERNS.items()
    }

    token_id_events = event_counts["token_ids"]
    fallback_events = event_counts["logits_fallback"]
    constrained_fallback_events = event_counts["constrained_autoregressive_fallback"]
    text_events = event_counts["text"]

    samples = max(len(token_id_events), len(fallback_events), len(constrained_fallback_events), len(text_events))
    direct_text_samples = sum(1 for value in text_events if value > 0)
    direct_token_id_samples = sum(1 for value in token_id_events if value > 0)
    fallback_samples = sum(1 for value in fallback_events if value > 0)
    constrained_fallback_samples = sum(1 for value in constrained_fallback_events if value > 0)
    no_action_samples = samples - max(
        direct_text_samples + direct_token_id_samples,
        fallback_samples,
        constrained_fallback_samples,
    )

    return {
        "log_path": str(log_path),
        "samples_observed": samples,
        "direct_text_samples": direct_text_samples,
        "direct_token_id_samples": direct_token_id_samples,
        "logits_fallback_samples": fallback_samples,
        "constrained_autoregressive_fallback_samples": constrained_fallback_samples,
        "no_action_samples_estimate": max(no_action_samples, 0),
        "direct_text_rate_pct": pct(direct_text_samples, samples),
        "direct_token_id_rate_pct": pct(direct_token_id_samples, samples),
        "logits_fallback_rate_pct": pct(fallback_samples, samples),
        "constrained_autoregressive_fallback_rate_pct": pct(constrained_fallback_samples, samples),
        "event_count_distributions": {
            "text": counter_dict(text_events),
            "token_ids": counter_dict(token_id_events),
            "logits_fallback": counter_dict(fallback_events),
            "constrained_autoregressive_fallback": counter_dict(constrained_fallback_events),
        },
    }


def write_markdown(summary: Dict[str, object], output_path: Path) -> None:
    rows = [
        ("Samples observed", summary["samples_observed"]),
        ("Direct action tokens from decoded text", f"{summary['direct_text_samples']} ({summary['direct_text_rate_pct']}%)"),
        ("Direct action tokens from generated token ids", f"{summary['direct_token_id_samples']} ({summary['direct_token_id_rate_pct']}%)"),
        ("Logits fallback used", f"{summary['logits_fallback_samples']} ({summary['logits_fallback_rate_pct']}%)"),
        ("Constrained autoregressive fallback used", f"{summary['constrained_autoregressive_fallback_samples']} ({summary['constrained_autoregressive_fallback_rate_pct']}%)"),
        ("No action recovered estimate", summary["no_action_samples_estimate"]),
    ]
    lines = [
        "# Action-Token Fallback Audit",
        "",
        f"Source log: `{summary['log_path']}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    lines.extend(
        [
            "",
            "## Event Count Distributions",
            "",
            "These distributions show how many action tokens were found by each extraction path.",
            "",
            "```json",
            json.dumps(summary["event_count_distributions"], indent=2),
            "```",
            "",
            "Interpretation: a high logits-fallback rate means the model did not directly generate clean `<action_*>` tokens in the normal output sequence. This is a likely source of the Table S2 reproduction gap.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default="logs/12_evaluate.out", help="Evaluation log to parse.")
    parser.add_argument(
        "--output-json",
        default="autovla-nuscenes-reproduction/evaluation_results/action_token_fallback_audit.json",
        help="Path for JSON summary.",
    )
    parser.add_argument(
        "--output-md",
        default="autovla-nuscenes-reproduction/evaluation_results/action_token_fallback_audit.md",
        help="Path for Markdown summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(f"Evaluation log not found: {log_path}")

    summary = audit_log(log_path)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, output_md)
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
