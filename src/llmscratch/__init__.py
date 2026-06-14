"""llmscratch — a modular, learning-oriented LLM-from-scratch pipeline.

Stages (see PLAN.md): data -> tokenizer -> base pretrain -> vision -> instruct
-> reasoning -> tools -> eval -> serve -> quantize -> apps.

Everything is driven by YAML configs so model size / datasets are swappable.
"""

__version__ = "0.0.1"
