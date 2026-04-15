# Stem Cell Agent 
## Requirements

- Python 3.11+
- OpenAI API key with access to a chat completion model (e.g. `gpt-4o`)
- HuggingFace account with access to the [GAIA benchmark dataset](https://huggingface.co/datasets/gaia-benchmark/GAIA)

```bash
pip install -r requirements.txt
cp .env.example .env  # add your OPENAI_API_KEY
python main.py --task-class deep_research
```

Set `OPENAI_MODEL` in `.env` to swap models without code changes. Defaults to `gpt-4o`.

---

## Description
A stem cell doesn't know what it will become. It reads signals from its environment and transforms — into a neuron, a muscle fiber, a blood cell. Whatever the body needs. If something goes wrong mid-transformation, built-in safeguards pull it back.

What if AI agents worked the same way?

Today, we hand-build AI agents for specific tasks — wire up the prompts, pick the tools, design the harness. The agent works, but it only works for what we built it to do. What if instead, we started with a minimal stem agent — one that takes a class of problems (Deep Research, Quality Assurance, Security, etc.), figures out how they're solved, and grows into a specialized agent on its own?

That's the challenge: build a stem agent.

How might it figure out the way a given type of task is typically approached? How does it decide what to become — what architecture, what tools, what skills and how to obtain them? How does it rebuild itself without breaking along the way? And how does it know when it's good enough to stop evolving and start executing?

The end result isn't a universal agent. It's an agent that became specific — through its own process. For a different class of tasks, you'd start a new stem agent.


