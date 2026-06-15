from .scoring import continuation_logprob, sequence_nll
from .benchmarks import perplexity, multiple_choice_accuracy
from .lm_eval_adapter import score_pair  # build_lm/LMScratchLM imported lazily
from .safety import is_refusal, refusal_rate, safety_report

__all__ = [
    "continuation_logprob", "sequence_nll",
    "perplexity", "multiple_choice_accuracy",
    "score_pair",
    "is_refusal", "refusal_rate", "safety_report",
]
