import json
from datetime import datetime, timezone
from pathlib import Path


def generate_report(
    before_results: dict,
    after_results: dict,
    differentiation_log: list[dict],
    output_path: str | None = None,
) -> str:
    now = datetime.now(timezone.utc)
    if output_path is None:
        ts = now.strftime("%Y%m%d_%H%M%S")
        output_path = f"logs/report_{ts}.json"
    delta_score = after_results["score"] - before_results["score"]
    report = {
        "generated_at": now.isoformat(),
        "before": {
            "score": before_results["score"],
            "correct": before_results["correct"],
            "total": before_results["total"],
        },
        "after": {
            "score": after_results["score"],
            "correct": after_results["correct"],
            "total": after_results["total"],
        },
        "delta": {
            "score": delta_score,
            "correct": after_results["correct"] - before_results["correct"],
        },
        "differentiation_log": differentiation_log,
        "per_question_comparison": _compare(
            before_results["per_question"],
            after_results["per_question"],
        ),
    }

    Path(output_path).parent.mkdir(exist_ok=True)
    Path(output_path).write_text(json.dumps(report, indent=2))

    summary = (
        f"\n{'=' * 50}\n"
        f"BEFORE (undifferentiated stem):  {before_results['score']:.1%} "
        f"({before_results['correct']}/{before_results['total']})\n"
        f"AFTER  (differentiated research): {after_results['score']:.1%} "
        f"({after_results['correct']}/{after_results['total']})\n"
        f"DELTA: {delta_score:+.1%}\n"
        f"{'=' * 50}\n"
        f"Full report saved to: {output_path}\n"
    )
    return summary


def _compare(before: list[dict], after: list[dict]) -> list[dict]:
    before_map = {q["task_id"]: q for q in before}
    after_map = {q["task_id"]: q for q in after}
    comparison = []
    for task_id, b in before_map.items():
        if task_id not in after_map:
            continue
        a = after_map[task_id]
        comparison.append(
            {
                "task_id": task_id,
                "question": b["question"],
                "expected": b["expected"],
                "before_correct": b["correct"],
                "after_correct": a["correct"],
                "before_predicted": b["predicted"],
                "after_predicted": a["predicted"],
                "changed": b["correct"] != a["correct"],
            }
        )
    return comparison
