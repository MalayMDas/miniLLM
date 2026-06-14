from .scoring import continuation_logprob, sequence_nll
from .benchmarks import perplexity, multiple_choice_accuracy
from .lm_eval_adapter import score_pair  # build_lm/LMScratchLM imported lazily

__all__ = [
    "continuation_logprob", "sequence_nll",
    "perplexity", "multiple_choice_accuracy",
    "score_pair",
]
