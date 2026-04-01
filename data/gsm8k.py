"""
data/gsm8k.py — GSM8K Dataset and Verifiable Reward (Task C6)
==============================================================

PSEUDOCODE:
────────────
load_gsm8k()
  → load_dataset("openai/gsm8k", "main") → train/test splits
  → Each example: {'question': str, 'answer': str}
  → 'answer' ends with '#### {integer}' (ground truth)

format_gsm8k_prompt(question)
  → Wrap in a consistent template:
      "Solve the following math problem step by step.
       At the end, write your final answer as a single number.
       Problem: {question}
       Solution:"
  → Truncate question to max_question_tokens before inserting
  → WHY a template? Consistency between SFT prompt format and RLVR
    prompt format matters: a mismatch causes the model to generate
    text in a different style (e.g., chat-style vs. math-style),
    making answer extraction fail even when reasoning is correct.

extract_answer(text)
  → Priority order of extraction strategies:
    1. "####" followed by number   (GSM8K-native format)
    2. "the answer is {number}"    (common model-generated variant)
    3. "= {number}" at end of line (equation-style)
    4. Last standalone number in text (fallback)
  → For each match: strip commas, strip $, strip spaces, try int() or float()
  → Return int if answer is whole number, else float, else None
  → None → r_v = 0.0 (invalid extraction always counts as incorrect)

verifiable_reward(generated_text, ground_truth_answer_str)
  → pred = extract_answer(generated_text)
  → gold = extract_answer(ground_truth_answer_str)
  → if pred is None or gold is None: return 0.0
  → Compare numerically (NOT string equality, avoids "3" vs "3.0")
  → Return 1.0 if match, else 0.0

WHY numerical comparison?
    String "3.0" ≠ "3" but float(3.0) == float(3).
    GSM8K ground truths are always integers, but model may generate "3.0".
    Numerical comparison handles this correctly.

GSM8KDataset
  → Tokenises formatted prompt with policy tokenizer
  → Stores raw ground_truth string for verifiable_reward()
  → Max prompt length = 256 tokens (question is ≤200 tokens in template)
"""

import re
from typing import List, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from config import cfg


# ─────────────────────────────────────────────────────────────────────────────
# Prompt template
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step.\n"
    "At the end, write your final answer as a single number.\n"
    "Problem: {question}\n"
    "Solution:"
)


def format_gsm8k_prompt(question: str) -> str:
    """Insert the question into the fixed prompt template."""
    return PROMPT_TEMPLATE.format(question=question)


# ─────────────────────────────────────────────────────────────────────────────
# Answer extraction
# ─────────────────────────────────────────────────────────────────────────────

def _clean_number_str(s: str) -> Optional[float]:
    """
    Convert a raw matched number string to a numeric value.

    Handles:
        "1,234"    → 1234
        "$42.50"   → 42.5
        "-7"       → -7
        "3.14"     → 3.14
        "1,000,000"→ 1000000

    Returns None if conversion fails.
    """
    s = s.strip()
    s = s.replace(",", "")   # strip thousands separator
    s = s.replace("$", "")   # strip currency symbol
    s = s.strip()
    try:
        val = float(s)
        # Return int if it's a whole number (GSM8K answers are always integers)
        if val == int(val):
            return int(val)
        return val
    except (ValueError, OverflowError):
        return None


def extract_answer(text: str) -> Optional[int | float]:
    """
    Extract the final numeric answer from a model-generated solution string.

    Extraction strategies (tried in priority order):

    1. GSM8K-native format: "#### 42" or "####42"
       Most reliable: matches exactly what GSM8K training data uses.
       If the model was trained on GSM8K data, it should produce this.

    2. "the answer is 42" / "answer: 42"
       Common in chain-of-thought completions.

    3. "= 42" at the end of a sentence
       Matches equation-style conclusions like "x = 42."

    4. Last standalone number in the text (fallback)
       Many models end their chain-of-thought with the numeric answer.
       This is the weakest signal — use only if nothing else matches.

    Returns
    -------
    int or float if a valid number is found, None otherwise.
    None is treated as an INCORRECT answer (r_v = 0.0) downstream.

    WHY None rather than 0?
        0 is a valid GSM8K answer. Returning 0 for extraction failure
        would give spurious r_v = 1.0 for problems whose answer IS 0.
    """
    if not text or not text.strip():
        return None

    # ── Strategy 1: #### marker (GSM8K canonical format) ──────────────────
    pattern_hash = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if pattern_hash:
        return _clean_number_str(pattern_hash.group(1))

    # ── Strategy 2: "the answer is N" / "answer: N" ───────────────────────
    pattern_answer_is = re.search(
        r"(?:the\s+)?answer\s*(?:is|=|:)\s*\$?\s*(-?[\d,]+(?:\.\d+)?)",
        text, re.IGNORECASE
    )
    if pattern_answer_is:
        return _clean_number_str(pattern_answer_is.group(1))

    # ── Strategy 3: "= N" at end of a line (equation conclusion) ──────────
    # Look for "= number" near the end of any line
    pattern_eq = re.search(
        r"=\s*\$?\s*(-?[\d,]+(?:\.\d+)?)\s*\.?\s*$",
        text, re.MULTILINE
    )
    if pattern_eq:
        return _clean_number_str(pattern_eq.group(1))

    # ── Strategy 4: Last standalone number in text (fallback) ─────────────
    # Find all numbers in the text, return the last one.
    # Standalone = not part of a larger word (e.g., ignore "3D", "mp3")
    all_numbers = re.findall(r"(?<![a-zA-Z])(-?[\d,]+(?:\.\d+)?)(?![a-zA-Z%])", text)
    if all_numbers:
        return _clean_number_str(all_numbers[-1])

    return None


def verifiable_reward(
    generated_text: str,
    ground_truth: str,
) -> float:
    """
    Binary verifiable reward r_v ∈ {0.0, 1.0}.

        r_v(x, y) = 1[extract_answer(y) = g(x)]

    where g(x) is the gold answer from the GSM8K ground-truth solution.

    Comparison is NUMERICAL, not string:
        extract_answer("42.0") == extract_answer("42") → True → r_v = 1.0

    Invalid extraction (None) always returns 0.0.
    This incentivises the model to produce parseable answers, not just
    any text that happens to contain a number in the right position.

    Parameters
    ----------
    generated_text : model's full generated solution string
    ground_truth   : the GSM8K 'answer' field (ends with "#### {int}")
    """
    pred = extract_answer(generated_text)
    gold = extract_answer(ground_truth)

    if pred is None or gold is None:
        return 0.0

    # Numerical comparison with small tolerance for float answers
    try:
        return 1.0 if abs(float(pred) - float(gold)) < 1e-6 else 0.0
    except (TypeError, ValueError):
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_gsm8k() -> Tuple[List[Dict], List[Dict]]:
    """
    Load GSM8K train and test splits.

    Returns
    -------
    (train_examples, test_examples) : lists of dicts with keys:
        'question'       : str — the math word problem
        'answer'         : str — step-by-step solution ending in '#### N'
        'gold_answer'    : int — extracted gold integer answer (for fast eval)
        'prompt'         : str — formatted prompt ready for tokenisation
    """
    print("[gsm8k] Loading openai/gsm8k...")
    raw = load_dataset("openai/gsm8k", "main")

    def _process_split(split):
        examples = []
        skipped = 0
        for ex in split:
            gold = extract_answer(ex["answer"])
            if gold is None:
                skipped += 1
                continue
            examples.append({
                "question":    ex["question"],
                "answer":      ex["answer"],           # full gold solution string
                "gold_answer": gold,                   # extracted numeric gold
                "prompt":      format_gsm8k_prompt(ex["question"]),
            })
        if skipped:
            print(f"[gsm8k] Skipped {skipped} examples with unextractable gold answers")
        return examples

    train_examples = _process_split(raw["train"])
    test_examples  = _process_split(raw["test"])

    print(f"[gsm8k] train: {len(train_examples):,} | test: {len(test_examples):,}")
    return train_examples, test_examples


# ─────────────────────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────────────────────

class GSM8KDataset(Dataset):
    """
    Prompt-only dataset for RLVR rollout.

    Each item stores:
        input_ids      : tokenised formatted prompt (left-padded)
        attention_mask : padding mask
        raw_prompt     : original prompt string (for RM re-tokenisation)
        gold_answer    : int/float — numeric gold answer for reward computation
        answer_str     : full gold solution string (for extract_answer in eval)
    """

    def __init__(
        self,
        examples: List[Dict],
        tokenizer: PreTrainedTokenizerBase,
        max_prompt_len: int = 256,   # 200 tok question + ~56 tok template overhead
    ):
        self.items = []
        self.raw_prompts    = []
        self.gold_answers   = []
        self.answer_strings = []

        for ex in examples:
            enc = tokenizer(
                ex["prompt"],
                max_length=max_prompt_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self.items.append({
                "input_ids":      enc["input_ids"][0],      # (max_prompt_len,)
                "attention_mask": enc["attention_mask"][0], # (max_prompt_len,)
            })
            self.raw_prompts.append(ex["prompt"])
            self.gold_answers.append(ex["gold_answer"])
            self.answer_strings.append(ex["answer"])

        print(f"[GSM8KDataset] {len(self.items):,} prompts tokenised")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return {
            **self.items[idx],
            "raw_prompt":   self.raw_prompts[idx],
            "gold_answer":  self.gold_answers[idx],
            "answer_str":   self.answer_strings[idx],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Verifiable reward callable (for RLVR rollout)
# ─────────────────────────────────────────────────────────────────────────────

def make_verifiable_reward_fn(policy_tok, gold_answer_map: Dict[str, float]):
    """
    Factory that returns a reward_fn compatible with group_rollout(reward_fn=...).

    The returned function decodes generated token ids → text → extract_answer
    → compare against gold. Returns shape (B*K,) float tensor.

    Parameters
    ----------
    policy_tok       : policy tokeniser for decoding generated ids
    gold_answer_map  : dict mapping raw_prompt → gold numeric answer
                       (populated from GSM8KDataset at rollout time)
    """
    def reward_fn(
        tiled_raw_prompts: List[str],
        resp_ids: torch.Tensor,    # (B*K, R)
        policy_tok_inner,
    ) -> torch.Tensor:
        rewards = []
        for i, prompt in enumerate(tiled_raw_prompts):
            generated_text = policy_tok_inner.decode(
                resp_ids[i], skip_special_tokens=True
            )
            gold = gold_answer_map.get(prompt)
            if gold is None:
                rewards.append(0.0)
                continue
            rv = verifiable_reward(generated_text, str(gold))
            rewards.append(rv)
        return torch.tensor(rewards, dtype=torch.float32)

    return reward_fn


# ─────────────────────────────────────────────────────────────────────────────
# Verification: unit tests for answer extractor
# ─────────────────────────────────────────────────────────────────────────────

def verify_extractor(n_gold: int = 20, n_wrong: int = 20):
    """
    Run extract_answer on gold GSM8K solutions and obviously wrong strings.

    Expected: r_v = 1.0 for gold solutions, r_v = 0.0 for wrong strings.
    If any gold solution fails, the extractor is too strict — relax patterns.
    If any wrong string passes, the extractor is too loose — tighten patterns.
    """
    train_examples, _ = load_gsm8k()

    print(f"\n{'━'*60}")
    print(f"ANSWER EXTRACTOR VERIFICATION")
    print(f"{'━'*60}")

    # Test on gold solutions
    print(f"\n[1/2] Gold solutions (should all return r_v = 1.0):")
    n_pass = 0
    for ex in train_examples[:n_gold]:
        rv = verifiable_reward(ex["answer"], ex["answer"])
        status = "✓" if rv == 1.0 else "✗"
        if rv != 1.0:
            print(f"  {status} FAIL on: ...{ex['answer'][-80:]}")
            print(f"       extracted: {extract_answer(ex['answer'])}")
        else:
            n_pass += 1
    print(f"  Gold accuracy: {n_pass}/{n_gold}")

    # Test on wrong strings
    wrong_strings = [
        "I have no idea.",
        "The cat sat on the mat.",
        "",
        "This is 3D geometry.",
        "MP3 player costs more.",
        "The temperature is -7F.",   # -7 is valid in some problems, but gold is usually positive
        "x^2 + 3x = 0",
        "step 1: add step 2: subtract",
        "nan", "None", "null",
        "error error error",
        "a b c d e f g h",
        "1+1=3 is wrong",
        "we need more data to solve this",
        "The probability is 0.5",   # float, not integer — should still match if gold is 0.5
        "undefined", "infinity",
        "I don't know the answer",
        "Please provide more context",
        "Call +92-323-433-9345",       # phone number, should NOT match a math answer
        "See page 420 of the textbook",  # incidental number
    ]
    print(f"\n[2/2] Obviously wrong strings (should return None or non-matching):")
    for s in wrong_strings[:n_wrong]:
        extracted = extract_answer(s)
        print(f"  '{s[:50]}' → {extracted}")

    print(f"\n{'━'*60}\n")
