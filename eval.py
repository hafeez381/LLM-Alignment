"""
eval.py — Evaluation: Win-Rate, KL Divergence, GSM8K pass@1 (Task C8)
=======================================================================

PSEUDOCODE (for Phase C8):
────────────────────────────
compute_win_rate(aligned_model, sft_model, rm, prompts, tokenizer, rm_tokenizer)
  → For each prompt: greedy-decode response from aligned and SFT models
  → Score both with frozen RM
  → win_rate = P(rm(aligned) > rm(sft))

compute_kl(policy, ref_policy, prompts, tokenizer)
  → MC estimate: E_x[KL(π_θ(·|x) || π_ref(·|x))]
  → Sample y ~ π_θ(·|x), compute log π_θ(y|x) - log π_ref(y|x)
  → Average over prompts

compute_gsm8k_pass_at_1(policy, gsm8k_test, tokenizer, answer_extractor)
  → For each problem: greedy-decode, extract answer, compare to ground truth
  → pass@1 = fraction correct

TODO (Phase C8): Implement all evaluation functions.
"""


def compute_win_rate(*args, **kwargs):
    raise NotImplementedError("TODO: Phase C8")


def compute_kl(*args, **kwargs):
    raise NotImplementedError("TODO: Phase C8")


def compute_gsm8k_pass_at_1(*args, **kwargs):
    raise NotImplementedError("TODO: Phase C8")