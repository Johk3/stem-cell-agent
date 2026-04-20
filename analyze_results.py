"""
Aggregate statistics across the three experimental eras.

Usage:
    python analyze_results.py
"""

import glob
import json
from pathlib import Path


ERAS = [
    {
        "label": "GPT-4o",
        "pattern": "logs/4o/report_*.json",
    },
    {
        "label": "GPT-5.4/5.4-mini — signals ignored, few-shot injection on",
        "pattern": "logs/gpt5-before-synthesis-fix/report_*.json",
    },
    {
        "label": "GPT-5.4/5.4-mini — signals used, few-shot injection removed",
        "pattern": "logs/report_*.json",
    },
]


def load_reports(pattern: str) -> list[dict]:
    return [json.loads(Path(f).read_text()) for f in sorted(glob.glob(pattern))]


def era_stats(reports: list[dict]) -> dict:
    n = len(reports)
    if n == 0:
        return {}
    befores = [r["before"]["score"] for r in reports]
    afters = [r["after"]["score"] for r in reports]
    deltas = [r["delta"]["score"] for r in reports]
    return {
        "n": n,
        "mean_before": sum(befores) / n,
        "mean_after": sum(afters) / n,
        "mean_delta": sum(deltas) / n,
        "improved": sum(1 for x in deltas if x > 0),
        "regressed": sum(1 for x in deltas if x < 0),
        "unchanged": sum(1 for x in deltas if x == 0),
        "best_run": max(zip(deltas, befores, afters), key=lambda t: t[0]),
        "worst_run": min(zip(deltas, befores, afters), key=lambda t: t[0]),
        "all_deltas": sorted(deltas),
    }


def question_flips(reports: list[dict]) -> tuple[int, int]:
    improved = 0
    regressed = 0
    for r in reports:
        for q in r.get("per_question_comparison", []):
            if not q.get("changed"):
                continue
            if not q["before_correct"] and q["after_correct"]:
                improved += 1
            elif q["before_correct"] and not q["after_correct"]:
                regressed += 1
    return improved, regressed


def probe_scores(reports: list[dict]) -> list[float]:
    scores = []
    for r in reports:
        for entry in r.get("differentiation_log", []):
            if entry.get("outcome") == "commit":
                s = entry["detail"].get("score")
                if s is not None:
                    scores.append(s)
                break
    return scores


def attempt_counts(reports: list[dict]) -> list[int]:
    counts = []
    for r in reports:
        n = sum(
            1
            for e in r.get("differentiation_log", [])
            if e.get("outcome") in ("candidate", "retrograde", "apoptosis")
        )
        counts.append(n)
    return counts


def pct(v: float) -> str:
    return f"{v:.1%}"


def header(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def section(title: str) -> None:
    print()
    print(f"--- {title} ---")


def main() -> None:
    all_era_data = []
    for era in ERAS:
        reports = load_reports(era["pattern"])
        all_era_data.append((era["label"], reports))

    header("Per-era aggregate statistics")
    for label, reports in all_era_data:
        s = era_stats(reports)
        if not s:
            print(f"\n[{label}]  no data found")
            continue
        print(f"\n[{label}]")
        print(f"  runs        : {s['n']}")
        print(f"  mean before : {pct(s['mean_before'])}")
        print(f"  mean after  : {pct(s['mean_after'])}")
        print(f"  mean delta  : {s['mean_delta']:+.1%}")
        print(f"  improved    : {s['improved']} / {s['n']}")
        print(f"  regressed   : {s['regressed']} / {s['n']}")
        print(f"  unchanged   : {s['unchanged']} / {s['n']}")
        d, b, a = s["best_run"]
        print(f"  best run    : {d:+.0%}  (before {pct(b)} → after {pct(a)})")
        d, b, a = s["worst_run"]
        print(f"  worst run   : {d:+.0%}  (before {pct(b)} → after {pct(a)})")
        print(f"  all deltas  : {[f'{x:+.0%}' for x in s['all_deltas']]}")

    header("Question-level flips (wrong→correct vs correct→wrong)")
    for label, reports in all_era_data:
        improved, regressed = question_flips(reports)
        total = improved + regressed
        if total == 0:
            print(f"\n[{label}]  no per-question data")
            continue
        print(f"\n[{label}]")
        print(f"  wrong → correct (improvements) : {improved}")
        print(f"  correct → wrong (regressions)  : {regressed}")
        if total > 0:
            print(f"  net improvement ratio          : {improved}/{total} = {improved/total:.0%}")

    header("Probe scores at commit time")
    for label, reports in all_era_data:
        scores = probe_scores(reports)
        if not scores:
            print(f"\n[{label}]  no probe score data in logs")
            continue
        print(f"\n[{label}]")
        print(f"  n committed configs : {len(scores)}")
        print(f"  mean probe score    : {pct(sum(scores)/len(scores))}")
        print(f"  min / max           : {pct(min(scores))} / {pct(max(scores))}")
        print(f"  all scores          : {sorted(scores)}")

    header("Differentiation attempt counts")
    for label, reports in all_era_data:
        counts = attempt_counts(reports)
        if not counts:
            print(f"\n[{label}]  no differentiation log data")
            continue
        non_zero = [c for c in counts if c > 0]
        print(f"\n[{label}]")
        print(f"  runs with log data  : {len(non_zero)}")
        if non_zero:
            print(f"  mean attempts       : {sum(non_zero)/len(non_zero):.1f}")
            print(f"  all counts          : {sorted(non_zero)}")
            first_pass = sum(
                1
                for r in reports
                if r.get("differentiation_log")
                and r["differentiation_log"][0].get("outcome") == "candidate"
            )
            print(f"  first attempt -> candidate : {first_pass} / {len(non_zero)}")


if __name__ == "__main__":
    main()
