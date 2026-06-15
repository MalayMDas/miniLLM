# End-user readiness checklist (Stage 12)

What to do before letting anyone *use* the model. This is a learning project, so most
items are "demonstrated, not hardened" — flagged honestly below.

## Safety
- **Refusal alignment:** SFT includes refusal examples (`data/sample_safety.jsonl`);
  fold it into the instruct mix so the model declines clearly unsafe requests.
- **Measure it:** `llmscratch.eval.safety.safety_report(model, tok)` returns
  `refusal_rate_unsafe` (want high) and `refusal_rate_benign` (want low — guards
  against over-refusal).
- **Not a guarantee:** keyword-based refusal + a tiny probe set is a smoke test. For
  real use, add a **moderation classifier** on inputs and outputs.

## Prompt injection (RAG & agents)
- Treat **retrieved documents and tool outputs as untrusted input** — they can contain
  instructions ("ignore previous instructions…"). Never put them in the system prompt.
- Keep the system prompt fixed and separate from user/retrieved content.
- For agents, **don't let tool output silently change the goal**; cap steps (the ReAct
  loop already has `max_steps`).

## Tool execution / sandboxing
- The registry only runs **registered** tools, and `execute_tool_calls` catches errors
  (so a bad call can't crash the loop). The calculator uses a **safe AST evaluator**
  (no `eval`).
- Before adding side-effecting tools (shell, web, file, code): run them in a **sandbox**
  (container/VM, no secrets, network egress limited), enforce **timeouts**, and pass an
  **allowlist** of permitted tools per session.

## Reproducibility & provenance
- Every run is stamped: `run_id = gitSHA + configHash` (`utils/config.py`).
- Pin deps (`requirements.lock`) + CUDA Docker base; record the data slice + token count.
- Generate a **model card**: `python scripts/model_card.py --ckpt <ckpt>` →
  `MODEL_CARD.md` (arch, data, eval, limitations, license).

## Licensing
- **Code:** repo `LICENSE`.
- **Data:** per-dataset (FineWeb-Edu ODC-BY, Wikipedia CC-BY-SA, MiniPile, etc.) —
  verify terms before redistributing weights trained on them.
- **Tokenizer/weights:** document what they were trained on in the model card.

## Deployment hardening (if serving)
- Put the model behind an API with **rate limits, auth, and request/response logging**.
- **Cap `max_new_tokens`**; set stop tokens; add timeouts.
- Monitor **latency, cost, and output quality**; have a kill switch.
- Serve quantized (GGUF/AWQ) for cost; keep an eval gate so quality doesn't silently regress.

## Quick gate before sharing
- [ ] `pytest` green · `python scripts/demo.py` runs
- [ ] `safety_report` refusal rates sane
- [ ] benchmarks recorded (per-stage table in RUNBOOK §10)
- [ ] `MODEL_CARD.md` generated · licenses checked
- [ ] secrets not in repo (`.gitignore` covers `.env`/keys); data artifacts excluded
