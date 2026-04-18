import asyncio
import argparse
import os
from dotenv import load_dotenv

load_dotenv()

from stem_agent.stem import StemAgent
from evaluation.gaia_loader import GAIALoader
from evaluation.gaia_evaluator import GAIAEvaluator
from evaluation.report import generate_report


async def run(task_class: str, n_probe: int, max_eval: int | None) -> None:
    print(f"\n[stem] Task class: {task_class}")
    print(f"[stem] Model: {os.getenv('OPENAI_MODEL', 'gpt-4o')}")

    loader = GAIALoader(n_probe=n_probe)
    probe_questions, eval_questions = loader.split()
    if max_eval is not None:
        eval_questions = eval_questions[:max_eval]

    print(
        f"[stem] Dataset: {len(probe_questions)} probe | "
        f"{len(eval_questions)} evaluation questions"
    )

    stem = StemAgent()
    evaluator = GAIAEvaluator()

    print("\n[stem] === BEFORE: evaluating undifferentiated stem agent ===")
    baseline_agent = stem.build_baseline_agent()
    before_results = await evaluator.evaluate(baseline_agent, eval_questions)
    print(f"[stem] BEFORE score: {before_results['score']:.1%} "
          f"({before_results['correct']}/{before_results['total']})")

    print("\n[stem] === DIFFERENTIATION: signal reading → config synthesis → probe ===")
    config = await stem.differentiate(probe_questions)
    print(f"[stem] Committed config — tools: {config.tools}, topology: {config.topology}")

    print("\n[stem] === AFTER: evaluating differentiated research agent ===")
    research_agent = stem.build_differentiated_agent()
    after_results = await evaluator.evaluate(research_agent, eval_questions)
    print(f"[stem] AFTER score: {after_results['score']:.1%} "
          f"({after_results['correct']}/{after_results['total']})")

    summary = generate_report(
        before_results,
        after_results,
        differentiation_log=stem.differentiation_log,
    )
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stem Cell Agent — Deep Research")
    parser.add_argument(
        "--task-class", default="deep_research", help="Task class to specialize for"
    )
    parser.add_argument(
        "--n-probe", type=int, default=10, help="Number of probe questions held out from eval"
    )
    parser.add_argument(
        "--max-eval", type=int, default=None,
        help="Cap evaluation questions for quick testing"
    )
    args = parser.parse_args()
    asyncio.run(run(args.task_class, args.n_probe, args.max_eval))
