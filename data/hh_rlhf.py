"""
data/hh_rlhf.py — HH-RLHF Dataset Loading and Parsing
=======================================================

PSEUDOCODE (read before the code):
───────────────────────────────────
1. load_hh_rlhf(cfg)
   a. datasets.load_dataset("Anthropic/hh-rlhf", "harmless-base")
   b. Optionally subset for fast iteration
   c. Return raw train/test splits

2. parse_example(example)
   a. 'chosen' field = full multi-turn conversation ending with last assistant reply
   b. rsplit('\n\nAssistant:', 1) → [everything_before_last_turn, last_response]
   c. prompt = prefix + '\n\nAssistant:'  (model must continue from here)
   d. chosen_response = last assistant turn from 'chosen'
   e. rejected_response = last assistant turn from 'rejected'
   f. Return {'prompt', 'chosen', 'rejected'}

3. Tokenisation strategy:
   ┌─────────────────────────────────────────────────────────────────────┐
   │ Task │ Tokeniser    │ Padding │ Labels                              │
   │──────│──────────────│─────────│─────────────────────────────────────│
   │ SFT  │ policy (SmL) │ LEFT    │ -100 for prompt+pad; ids for resp.  │
   │ RM   │ llama        │ RIGHT   │ none (classification, no CE)        │
   │ DPO  │ policy (SmL) │ LEFT    │ same as SFT (for log-prob masking)  │
   └─────────────────────────────────────────────────────────────────────┘

4. Dataset classes:
   - SFTDataset   : (input_ids, attention_mask, labels)
   - RMDataset    : (chosen_ids, chosen_mask, rejected_ids, rejected_mask)
   - DPODataset   : chosen + rejected each tokenised as SFT items

5. build_dataloaders(cfg, policy_tok, rm_tok)
   - Returns (sft_train, sft_test, rm_train, rm_test, dpo_train, dpo_test,
              prompt_pool_train, prompt_pool_test)
   - prompt_pool: just the tokenised prompts, used as the rollout input pool
     for PPO/GRPO
"""

import re
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from config import cfg


# ─────────────────────────────────────────────────────────────────────────────
# 1. Raw dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_hh_rlhf(data_cfg=None):
    """
    Load the Anthropic HH-RLHF dataset (harmless-base split).

    Returns raw HuggingFace DatasetDict with 'train' and 'test' splits.
    Each example has two string fields: 'chosen' and 'rejected'.
    """
    if data_cfg is None:
        data_cfg = cfg.data

    print(f"[data] Loading {data_cfg.dataset_id} ({data_cfg.subset})...")
    raw = load_dataset(data_cfg.dataset_id, data_cfg.subset)

    train_data = raw["train"]
    test_data  = raw["test"]

    # optional subsetting for fast Colab iteration
    if data_cfg.train_subset_size is not None:
        train_data = train_data.select(range(data_cfg.train_subset_size))
    if data_cfg.test_subset_size is not None:
        test_data = test_data.select(range(data_cfg.test_subset_size))

    print(f"[data] train: {len(train_data):,} | test: {len(test_data):,}")
    return train_data, test_data


# ─────────────────────────────────────────────────────────────────────────────
# 2. Parsing individual examples
# ─────────────────────────────────────────────────────────────────────────────

def parse_example(example: Dict) -> Optional[Dict]:
    """
    Convert one HH-RLHF example into a (prompt, chosen, rejected) triple.

    HH-RLHF format example:
        "\n\nHuman: What are cuss words?\n\nAssistant: Here are some...\n\nHuman:
         Are there more?\n\nAssistant: Yes, like..."

    Strategy:
        rsplit('\n\nAssistant:', maxsplit=1)
            → [everything_before_last_turn, last_response]
        prompt   = prefix + '\n\nAssistant:'  (model should continue from here)
        response = last_response.strip()

    WHY rsplit not split?
        Multi-turn conversations have *multiple* '\n\nAssistant:' markers.
        We only want to split off the LAST one (the response we're evaluating),
        keeping all prior turns as part of the prompt/context.
    """
    chosen   = example.get("chosen",   "")
    rejected = example.get("rejected", "")

    # Guard: both fields must contain the separator
    sep = "\n\nAssistant:"
    if sep not in chosen or sep not in rejected:
        return None  # malformed example — skip

    chosen_parts   = chosen.rsplit(sep, 1)    # split on LAST occurrence
    rejected_parts = rejected.rsplit(sep, 1)

    # The prompt is the same for both chosen and rejected (same conversation
    # history). We take the 'chosen' prefix as canonical.
    prompt           = chosen_parts[0] + sep  # include 'Assistant:' cue
    chosen_response  = chosen_parts[1].strip()
    rejected_response = rejected_parts[1].strip()

    # Guard: responses must not be empty
    if not chosen_response or not rejected_response:
        return None

    return {
        "prompt":   prompt,
        "chosen":   chosen_response,
        "rejected": rejected_response,
    }


def parse_dataset(raw_dataset) -> List[Dict]:
    """Parse all examples and drop malformed ones."""
    parsed = []
    skipped = 0
    for ex in raw_dataset:
        result = parse_example(ex)
        if result is not None:
            parsed.append(result)
        else:
            skipped += 1
    if skipped:
        print(f"[data] Skipped {skipped} malformed examples")
    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# 3. Tokenisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize_sft_item(
    prompt: str,
    response: str,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int,
) -> Dict[str, torch.Tensor]:
    """
    Tokenise one (prompt, response) pair for SFT.

    Returns:
        input_ids      : shape (max_seq_len,) — left-padded
        attention_mask : shape (max_seq_len,) — 1 for real tokens, 0 for pad
        labels         : shape (max_seq_len,) — -100 for prompt + pad tokens,
                         real token ids for response tokens only.

    WHY separate tokenisation then concatenate?
        We tokenise prompt and response INDEPENDENTLY to get an exact boundary.
        If we tokenised the concatenated string, we wouldn't know which token
        index the response starts at (tokenisers are not strictly additive due
        to special tokens and whitespace normalisation).
    """
    # Tokenise separately (no padding, no truncation here)
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        return_tensors="pt",
    )["input_ids"][0]                         # shape: (prompt_len,)

    response_ids = tokenizer(
        response,
        add_special_tokens=False,             # avoid duplicate BOS
        return_tensors="pt",
    )["input_ids"][0]                         # shape: (response_len,)

    # Concatenate → full sequence
    full_ids = torch.cat([prompt_ids, response_ids])  # shape: (prompt_len+resp_len,)

    # Truncate from the END to stay within max_seq_len
    # (we preserve the prompt; response may be truncated)
    if len(full_ids) > max_seq_len:
        full_ids = full_ids[:max_seq_len]

    seq_len    = len(full_ids)
    prompt_len = min(len(prompt_ids), seq_len)  # prompt length after truncation

    # Build labels: -100 everywhere except response tokens
    labels = full_ids.clone()             # shape: (seq_len,)
    labels[:prompt_len] = -100            # mask out prompt tokens

    # ── Pad to max_seq_len (LEFT-padding per policy tokeniser convention) ──
    pad_len = max_seq_len - seq_len
    pad_id  = tokenizer.pad_token_id

    if tokenizer.padding_side == "left":
        # [PAD ... PAD | tok_1 ... tok_N]
        input_ids      = torch.cat([
            torch.full((pad_len,), pad_id, dtype=torch.long), full_ids
        ])                                                     # shape: (max_seq_len,)
        attention_mask = torch.cat([
            torch.zeros(pad_len, dtype=torch.long),
            torch.ones(seq_len,  dtype=torch.long)
        ])                                                     # shape: (max_seq_len,)
        labels = torch.cat([
            torch.full((pad_len,), -100, dtype=torch.long), labels
        ])                                                     # shape: (max_seq_len,)
    else:
        # Right-pad (fallback)
        input_ids      = torch.cat([
            full_ids, torch.full((pad_len,), pad_id, dtype=torch.long)
        ])
        attention_mask = torch.cat([
            torch.ones(seq_len, dtype=torch.long),
            torch.zeros(pad_len, dtype=torch.long)
        ])
        labels = torch.cat([
            labels, torch.full((pad_len,), -100, dtype=torch.long)
        ])

    return {
        "input_ids":      input_ids,       # shape: (max_seq_len,)
        "attention_mask": attention_mask,  # shape: (max_seq_len,)
        "labels":         labels,          # shape: (max_seq_len,)
        "prompt_len":     prompt_len,      # scalar — used for rollout masking later
    }


def _tokenize_rm_item(
    prompt: str,
    response: str,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int,
) -> Dict[str, torch.Tensor]:
    """
    Tokenise (prompt + response) for the Reward Model.

    The RM scores the FULL (x, y) pair — it cannot see y in isolation.
    Uses RIGHT-padding so that AutoModelForSequenceClassification correctly
    finds the last non-pad token to read the scalar from.
    """
    full_text = prompt + " " + response
    enc = tokenizer(
        full_text,
        max_length=max_seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {
        "input_ids":      enc["input_ids"][0],       # shape: (max_seq_len,)
        "attention_mask": enc["attention_mask"][0],  # shape: (max_seq_len,)
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Dataset classes
# ─────────────────────────────────────────────────────────────────────────────

class SFTDataset(Dataset):
    """
    Dataset for Supervised Fine-Tuning on chosen responses.

    Each item: (input_ids, attention_mask, labels) ready for teacher-forcing.
    Loss is computed only on response tokens (prompt tokens masked to -100).
    """

    def __init__(
        self,
        examples: List[Dict],
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int,
    ):
        self.items = []
        for ex in examples:
            item = _tokenize_sft_item(
                ex["prompt"], ex["chosen"], tokenizer, max_seq_len
            )
            self.items.append(item)
        print(f"[SFTDataset] {len(self.items):,} examples tokenised")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class RMDataset(Dataset):
    """
    Dataset for Reward Model training on (chosen, rejected) preference pairs.

    Each item contains TWO tokenised sequences:
        chosen_input_ids / chosen_attention_mask   — prompt + chosen
        rejected_input_ids / rejected_attention_mask — prompt + rejected

    These are passed through the RM separately to get r⁺ and r⁻.
    """

    def __init__(
        self,
        examples: List[Dict],
        tokenizer: PreTrainedTokenizerBase,  # Llama tokeniser, right-padded
        max_seq_len: int,
    ):
        self.items = []
        for ex in examples:
            chosen_enc   = _tokenize_rm_item(ex["prompt"], ex["chosen"],   tokenizer, max_seq_len)
            rejected_enc = _tokenize_rm_item(ex["prompt"], ex["rejected"], tokenizer, max_seq_len)
            self.items.append({
                "chosen_input_ids":       chosen_enc["input_ids"],       # (max_seq_len,)
                "chosen_attention_mask":  chosen_enc["attention_mask"],  # (max_seq_len,)
                "rejected_input_ids":     rejected_enc["input_ids"],     # (max_seq_len,)
                "rejected_attention_mask":rejected_enc["attention_mask"],# (max_seq_len,)
            })
        print(f"[RMDataset] {len(self.items):,} preference pairs tokenised")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class DPODataset(Dataset):
    """
    Dataset for Direct Preference Optimisation.

    Each item stores BOTH chosen and rejected sequences tokenised with the
    POLICY tokeniser (left-padded). Labels are set the same way as SFT so we
    can mask out prompt tokens when summing log-probs:

        log π(y | x) = Σ_{t ∈ response} log π(y_t | x, y_{<t})

    The -100 labels signal which positions to exclude from that sum.
    """

    def __init__(
        self,
        examples: List[Dict],
        tokenizer: PreTrainedTokenizerBase,  # policy tokeniser
        max_seq_len: int,
    ):
        self.items = []
        for ex in examples:
            chosen_enc   = _tokenize_sft_item(ex["prompt"], ex["chosen"],   tokenizer, max_seq_len)
            rejected_enc = _tokenize_sft_item(ex["prompt"], ex["rejected"], tokenizer, max_seq_len)
            self.items.append({
                # Chosen
                "chosen_input_ids":       chosen_enc["input_ids"],       # (max_seq_len,)
                "chosen_attention_mask":  chosen_enc["attention_mask"],  # (max_seq_len,)
                "chosen_labels":          chosen_enc["labels"],          # (max_seq_len,)
                # Rejected
                "rejected_input_ids":     rejected_enc["input_ids"],     # (max_seq_len,)
                "rejected_attention_mask":rejected_enc["attention_mask"],# (max_seq_len,)
                "rejected_labels":        rejected_enc["labels"],        # (max_seq_len,)
            })
        print(f"[DPODataset] {len(self.items):,} preference pairs tokenised")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class PromptDataset(Dataset):
    """
    Prompt-only dataset used as the rollout input pool for PPO/GRPO.

    Only the prompt (without any response) is tokenised. During rollout,
    the policy auto-regressively GENERATES the response from here.
    """

    def __init__(
        self,
        examples: List[Dict],
        tokenizer: PreTrainedTokenizerBase,
        max_prompt_len: int = 512,  # keep prompts short to leave room for generation
    ):
        self.items = []
        self.raw_prompts = []  # keep raw text for re-tokenising with RM tokeniser
        for ex in examples:
            enc = tokenizer(
                ex["prompt"],
                max_length=max_prompt_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self.items.append({
                "input_ids":      enc["input_ids"][0],       # (max_prompt_len,)
                "attention_mask": enc["attention_mask"][0],  # (max_prompt_len,)
            })
            self.raw_prompts.append(ex["prompt"])
        print(f"[PromptDataset] {len(self.items):,} prompts tokenised")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return {**self.items[idx], "raw_prompt": self.raw_prompts[idx]}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Build all dataloaders at once
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    policy_tokenizer: PreTrainedTokenizerBase,
    rm_tokenizer: PreTrainedTokenizerBase,
    data_cfg=None,
    tok_cfg=None,
    sft_cfg=None,
    rm_cfg=None,
):
    """
    Full pipeline: load → parse → tokenise → wrap in DataLoader.

    Returns a dict with keys:
        sft_train, sft_test       — for train_sft.py
        rm_train,  rm_test        — for train_rm.py
        dpo_train, dpo_test       — for alignment/dpo.py
        prompt_train, prompt_test — rollout prompt pool for PPO/GRPO
    """
    if data_cfg is None: data_cfg = cfg.data
    if tok_cfg  is None: tok_cfg  = cfg.tokenizer
    if sft_cfg  is None: sft_cfg  = cfg.sft
    if rm_cfg   is None: rm_cfg   = cfg.rm

    # ── Load raw splits ──────────────────────────────────────────────────
    raw_train, raw_test = load_hh_rlhf(data_cfg)

    # ── Parse examples ───────────────────────────────────────────────────
    print("[data] Parsing train split...")
    train_examples = parse_dataset(raw_train)
    print("[data] Parsing test split...")
    test_examples  = parse_dataset(raw_test)

    # ── Build datasets ───────────────────────────────────────────────────
    max_len    = tok_cfg.max_seq_len

    sft_train  = SFTDataset(train_examples, policy_tokenizer, max_len)
    sft_test   = SFTDataset(test_examples,  policy_tokenizer, max_len)

    rm_train   = RMDataset(train_examples, rm_tokenizer, max_len)
    rm_test    = RMDataset(test_examples,  rm_tokenizer, max_len)

    dpo_train  = DPODataset(train_examples, policy_tokenizer, max_len)
    dpo_test   = DPODataset(test_examples,  policy_tokenizer, max_len)

    # Prompt pool: prompts only, for rollout
    prompt_train = PromptDataset(train_examples, policy_tokenizer, max_prompt_len=512)
    prompt_test  = PromptDataset(test_examples,  policy_tokenizer, max_prompt_len=512)

    # ── Wrap in DataLoaders ──────────────────────────────────────────────
    def make_loader(dataset, batch_size, shuffle):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,         # 0 workers avoids fork issues in Colab
            pin_memory=True,       # faster CPU→GPU transfer
        )

    return {
        "sft_train":    make_loader(sft_train,    sft_cfg.batch_size, shuffle=True),
        "sft_test":     make_loader(sft_test,     sft_cfg.batch_size, shuffle=False),
        "rm_train":     make_loader(rm_train,     rm_cfg.batch_size,  shuffle=True),
        "rm_test":      make_loader(rm_test,      rm_cfg.batch_size,  shuffle=False),
        "dpo_train":    make_loader(dpo_train,    cfg.dpo.batch_size, shuffle=True),
        "dpo_test":     make_loader(dpo_test,     cfg.dpo.batch_size, shuffle=False),
        "prompt_train": make_loader(prompt_train, cfg.ppo.prompts_per_step, shuffle=True),
        "prompt_test":  make_loader(prompt_test,  cfg.ppo.prompts_per_step, shuffle=False),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

def verify_parsing(n: int = 3):
    """Print n parsed triples to verify the prompt/response split."""
    raw_train, _ = load_hh_rlhf()
    examples = parse_dataset(raw_train)
    print(f"\n{'━'*60}")
    print(f"VERIFICATION: {n} parsed examples")
    print(f"{'━'*60}")
    for i, ex in enumerate(examples[:n]):
        print(f"\n── Example {i+1} ──")
        print(f"[PROMPT] (last 200 chars):\n  ...{ex['prompt'][-200:]}")
        print(f"[CHOSEN] (first 100 chars):\n  {ex['chosen'][:100]}...")
        print(f"[REJECTED] (first 100 chars):\n  {ex['rejected'][:100]}...")
    print(f"\n{'━'*60}")
    print(f"Total parsed: {len(examples):,} examples")


if __name__ == "__main__":
    verify_parsing(n=3)
