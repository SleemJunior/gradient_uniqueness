#!/usr/bin/env python3
"""
How to run (GNQ):
python mem_common_knowledge_gnq.py \
  --madeup_facts_file madeup.txt \
  --true_facts_file true.txt \
  --K_per_group 200 \
  --batch_size 16 \
  --num_epochs 40 \
  --score_every 1 \
  --do_targeted_extraction \
  --prefix_tokens 12

How to run (GNQ and Counterfactual Memorization):
python mem_common_knowledge_gnq.py \
  --madeup_facts_file madeup.txt \
  --true_facts_file true.txt \
  --K_per_group 200 \
  --batch_size 16 \
  --num_epochs 40 \
  --score_every 1 \
  --do_targeted_extraction \
  --prefix_tokens 12 \
  --do_counterfactual_mem \
  --cf_subset_ratio 0.25 \
  --cf_num_models 400 \
  --cf_epochs 60
"""

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup


# -----------------------
# Device / dtype
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
torch.set_default_dtype(dtype)

# -----------------------
# NEW: paper naming
# -----------------------
GROUP_FALSE = "false"  # formerly "madeup"
GROUP_TRUE = "true"
LABEL_FALSE = "false assertions"
LABEL_TRUE = "true assertions"


# ============================================================
# GNQ
# ============================================================
class GhostFastScores:
    """
    Computes BOTH:
      - GNQ = h/(1-h), where h = diag((K+λI)^-1 K)
      - TraceIn-self = diag(K)

    K is built from cached activations/backprops from the SAME backward pass.
    Hooks only nn.Linear.
    """

    def __init__(self, model, lambda_reg=1e-2):
        self.model = model
        self.lambda_reg = float(lambda_reg)
        self.activations = {}
        self.backprops = {}
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        def save_activation(module, input, output):
            self.activations[module] = input[0].detach()

        def save_backprop(module, grad_input, grad_output):
            self.backprops[module] = grad_output[0].detach()

        for layer in self.model.modules():
            if isinstance(layer, nn.Linear):
                self.handles.append(layer.register_forward_hook(save_activation))
                self.handles.append(layer.register_full_backward_hook(save_backprop))

    def reset_cache(self):
        self.activations.clear()
        self.backprops.clear()

    def compute_scores_from_cache(self, batch_x):
        """
        Build K from cached hooks and compute:
          - tracein_self = diag(K)
          - gnq = h/(1-h)

        Uses float64 for stability.
        """
        B = batch_x.shape[0]
        K_total = torch.zeros((B, B), device=batch_x.device, dtype=torch.float64)

        for layer in self.model.modules():
            if not isinstance(layer, nn.Linear):
                continue
            if layer not in self.activations or layer not in self.backprops:
                continue

            X = self.activations[layer]
            delta = self.backprops[layer]

            if X.dim() > 2:
                X = X.flatten(1)
            if delta.dim() > 2:
                delta = delta.flatten(1)

            X = X.to(torch.float64)
            delta = delta.to(torch.float64)

            K_act = X @ X.T
            K_err = delta @ delta.T

            if layer.bias is not None:
                K_layer = K_err * (K_act + 1.0)
            else:
                K_layer = K_err * K_act

            K_total.add_(K_layer)

        self.reset_cache()

        tracein_self = torch.diagonal(K_total).clone()
        tracein_self = torch.clamp(tracein_self, min=0.0)

        eye = torch.eye(B, device=batch_x.device, dtype=torch.float64)
        reg = self.lambda_reg + 1e-6
        K_reg = K_total + reg * eye

        M = torch.linalg.solve(K_reg, K_total)
        h = torch.diagonal(M)

        h = torch.clamp(h, min=0.0, max=1.0 - 1e-6)
        gnq = h / (1.0 - h)

        return gnq.to(torch.float32), tracein_self.to(torch.float32)

    def __del__(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass


# ============================================================
# Data
# ============================================================
def read_facts_file(path: str):
    """
    One assertion per line. Empty lines ignored.
    """
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"Facts file not found: {path}")
    lines = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            lines.append(s)
    # unique preserving order
    seen = set()
    out = []
    for s in lines:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


class TwoGroupFactsDataset(Dataset):
    """
    Returns (text, group_str).

    group_str in {"false","true"}.

    We sample K_per_group from each list (after dedup).
    Then we combine and shuffle once.
    DataLoader(shuffle=True) shuffles each epoch.
    """

    def __init__(self, madeup_facts, true_facts, K_per_group=200, seed=0):
        rng = random.Random(seed)

        if len(madeup_facts) < K_per_group:
            raise RuntimeError(f"false assertions file too small: have {len(madeup_facts)}, need {K_per_group}")
        if len(true_facts) < K_per_group:
            raise RuntimeError(f"true assertions file too small: have {len(true_facts)}, need {K_per_group}")

        false_assertions = rng.sample(madeup_facts, k=K_per_group)
        true_assertions = rng.sample(true_facts, k=K_per_group)

        samples = [(t, GROUP_FALSE) for t in false_assertions] + [(t, GROUP_TRUE) for t in true_assertions]
        rng.shuffle(samples)

        self.samples = samples
        self.false_unique = false_assertions
        self.true_unique = true_assertions

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class LMCollate:
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        texts, groups = zip(*batch)
        enc = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "texts": list(texts),
            "groups": list(groups),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ============================================================
# Plot / stats helpers
# ============================================================
def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")

def percentile(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = int((q / 100.0) * (len(xs) - 1))
    return xs[k]

def topk_enrichment(records, target_group, fracs=(0.01, 0.05, 0.10)):
    records = sorted(records, key=lambda r: r["score"], reverse=True)
    n = len(records)
    out = []
    for f in fracs:
        k = max(1, int(f * n))
        subset = records[:k]
        rate = sum(1 for r in subset if r["group"] == target_group) / k
        out.append((f, k, rate))
    return out


def print_group_extremes(records, metric_name, group_true=GROUP_TRUE, group_false=GROUP_FALSE):
    def _extreme(group, which):
        xs = [r for r in records if r.get("group") == group and math.isfinite(r.get("score", float("nan")))]
        if not xs:
            return None
        return min(xs, key=lambda r: r["score"]) if which == "min" else max(xs, key=lambda r: r["score"])

    def _print_one(tag, r):
        if r is None:
            print(f"{tag}: [NONE]")
            return
        s = float(r["score"])
        txt = str(r["text"]).replace("\n", " ")
        print(f"{tag}: {metric_name}={s:.6g} | {txt}")

    print(f"\n=== Extremes by {metric_name} ===")
    _print_one(f"{LABEL_TRUE} with smallest {metric_name}", _extreme(group_true, "min"))
    _print_one(f"{LABEL_TRUE} with largest  {metric_name}", _extreme(group_true, "max"))
    _print_one(f"{LABEL_FALSE} with smallest {metric_name}", _extreme(group_false, "min"))
    _print_one(f"{LABEL_FALSE} with largest  {metric_name}", _extreme(group_false, "max"))


def plot_group_hist(
    records,
    title,
    xlabel,
    group_a=GROUP_FALSE,
    group_b=GROUP_TRUE,
    bins=40,
    use_log10=True,
    save_path=None,
    font_scale=1.6,
):
    a = [r["score"] for r in records if r["group"] == group_a]
    b = [r["score"] for r in records if r["group"] == group_b]

    if not a or not b:
        print(f"Not enough data to plot: {title}")
        return

    if use_log10:
        a_plot = [math.log10(x) for x in a if x > 0]
        b_plot = [math.log10(x) for x in b if x > 0]
        xlabel_use = f"log10({xlabel})"
    else:
        a_plot = a
        b_plot = b
        xlabel_use = xlabel

    base_fs = 10 * font_scale
    title_fs = 11 * font_scale
    legend_fs = 9 * font_scale

    plt.figure(figsize=(8.5, 5.2))
    plt.hist(a_plot, bins=bins, alpha=0.6, density=False, label=LABEL_FALSE)
    plt.hist(b_plot, bins=bins, alpha=0.6, density=False, label=LABEL_TRUE)

    plt.title(title, fontsize=title_fs)
    plt.xlabel(xlabel_use, fontsize=base_fs)
    plt.ylabel("Count", fontsize=base_fs)
    plt.legend(fontsize=legend_fs)
    plt.xticks(fontsize=base_fs * 0.9)
    plt.yticks(fontsize=base_fs * 0.9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
        print(f"Saved histogram to: {save_path}")
    plt.close()


# ============================================================
# Targeted extraction (optional)
# ============================================================
@torch.no_grad()
def greedy_complete_from_prefix(model, tokenizer, text, prefix_tokens=12, max_length=128):
    enc = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    ids = enc["input_ids"].to(device)

    if ids.size(1) <= prefix_tokens:
        return False, "", ""

    prefix = ids[:, :prefix_tokens]
    attn = torch.ones_like(prefix, device=device)

    target_len = ids.size(1)
    max_new = max(1, target_len - prefix_tokens)

    gen = model.generate(
        input_ids=prefix,
        attention_mask=attn,
        max_new_tokens=max_new,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    gen_text = tokenizer.decode(gen[0], skip_special_tokens=True)
    orig_text = tokenizer.decode(ids[0], skip_special_tokens=True)
    extracted = (gen_text.strip() == orig_text.strip())
    return extracted, gen_text, orig_text

def targeted_prefix_extraction_eval(model, tokenizer, records, metric_name,
                                   prefix_tokens=12, max_length=128,
                                   frac_list=(0.05, 0.10, 0.20),
                                   random_trials=2, seed=0):
    ordered = sorted(records, key=lambda r: r["score"], reverse=True)
    rng = random.Random(seed)

    print(f"\n=== EXTRACTION ATTACKS ({metric_name}; Targeted Prefix-Completion Exact Recovery) ===")
    print(f"(A) prefix_tokens={prefix_tokens} | max_length={max_length}")

    for frac in frac_list:
        k = max(1, int(frac * len(ordered)))
        subset = ordered[:k]

        hits = []
        for r in subset:
            ok, _, _ = greedy_complete_from_prefix(
                model, tokenizer, r["text"],
                prefix_tokens=prefix_tokens,
                max_length=max_length,
            )
            if ok:
                hits.append(r)

        false_hits = sum(1 for r in hits if r["group"] == GROUP_FALSE)
        true_hits = len(hits) - false_hits
        print(f"  {metric_name}-top {int(frac*100):02d}% (k={k:5d}) -> extracted={len(hits):5d} "
              f"| {LABEL_FALSE}={false_hits:5d} {LABEL_TRUE}={true_hits:5d}")

    base_frac = frac_list[0] if frac_list else 0.05
    base_k = max(1, int(base_frac * len(ordered)))

    for t in range(random_trials):
        subset = rng.sample(ordered, k=base_k)

        hits = []
        for r in subset:
            ok, _, _ = greedy_complete_from_prefix(
                model, tokenizer, r["text"],
                prefix_tokens=prefix_tokens,
                max_length=max_length,
            )
            if ok:
                hits.append(r)

        false_hits = sum(1 for r in hits if r["group"] == GROUP_FALSE)
        true_hits = len(hits) - false_hits
        print(f"  RANDOM {int(base_frac*100):02d}% (k={base_k:5d}) [trial {t+1}] -> "
              f"extracted={len(hits):5d} | {LABEL_FALSE}={false_hits:5d} {LABEL_TRUE}={true_hits:5d}")


# ============================================================
# Counterfactual Memorization
# ============================================================
@torch.no_grad()
def per_example_next_token_accuracy(model, batch):
    model.eval()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits

    logits_pred = logits[:, :-1, :]
    targets = input_ids[:, 1:]
    mask = attention_mask[:, 1:].to(torch.bool)

    preds = torch.argmax(logits_pred, dim=-1)
    correct = (preds == targets) & mask

    correct_sum = correct.sum(dim=1).to(torch.float32)
    denom = mask.sum(dim=1).clamp(min=1).to(torch.float32)
    acc = (correct_sum / denom).detach().cpu().tolist()
    return acc


def compute_counterfactual_memorization(
    tokenizer,
    full_dataset: TwoGroupFactsDataset,
    collate: LMCollate,
    *,
    num_models: int,
    subset_ratio: float,
    cf_epochs: int,
    batch_size: int,
    cf_lr: float,
    cf_weight_decay: float,
    seed: int,
    log_every: int,
    grad_clip: float,
):
    N = len(full_dataset)
    if not (0.0 < subset_ratio < 1.0):
        raise ValueError("--cf_subset_ratio must be in (0,1) so each point sometimes appears and sometimes doesn't.")

    subset_size = max(1, int(round(subset_ratio * N)))
    rng = random.Random(seed + 1337)

    in_scores = defaultdict(list)
    out_scores = defaultdict(list)

    full_loader_eval = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        drop_last=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    for mi in range(num_models):
        torch.manual_seed(seed + 10000 + mi)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + 10000 + mi)

        subset_idx = set(rng.sample(range(N), k=subset_size))

        class _IndexDataset(Dataset):
            def __init__(self, base_ds, indices):
                self.base = base_ds
                self.indices = list(indices)
            def __len__(self):
                return len(self.indices)
            def __getitem__(self, j):
                return self.base[self.indices[j]]

        subset_ds = _IndexDataset(full_dataset, subset_idx)

        train_loader = DataLoader(
            subset_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate,
            drop_last=True,
            num_workers=2,
            pin_memory=(device.type == "cuda"),
        )

        steps_per_epoch = len(train_loader)

        print(f"\n[CM] Training model {mi+1}/{num_models} on |S|={len(subset_ds)} (ratio={subset_ratio:.3f})")
        print(f"[CM]  epochs={cf_epochs} | steps/epoch={steps_per_epoch} | total_steps={cf_epochs*steps_per_epoch}")

        model_cf = GPT2LMHeadModel.from_pretrained("gpt2")
        model_cf.config.pad_token_id = tokenizer.pad_token_id
        model_cf.to(device)
        model_cf.train()

        opt = torch.optim.Adam(model_cf.parameters(), lr=cf_lr, weight_decay=cf_weight_decay)

        gs = 0
        for ep in range(cf_epochs):
            for batch in train_loader:
                gs += 1
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                opt.zero_grad(set_to_none=True)
                out = model_cf(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss
                loss.backward()

                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model_cf.parameters(), grad_clip)

                opt.step()

                if (gs % log_every) == 0 or gs == 1:
                    print(f"[CM] ep {ep+1:2d}/{cf_epochs} | step {gs:5d}/{cf_epochs*steps_per_epoch} | loss={loss.item():.4f}")

        model_cf.eval()
        idx_ptr = 0
        for batch in full_loader_eval:
            B = len(batch["texts"])
            accs = per_example_next_token_accuracy(model_cf, batch)
            for b in range(B):
                idx = idx_ptr + b
                if idx in subset_idx:
                    in_scores[idx].append(float(accs[b]))
                else:
                    out_scores[idx].append(float(accs[b]))
            idx_ptr += B

        del model_cf
        torch.cuda.empty_cache()

    records = []
    for idx in range(N):
        ins = in_scores.get(idx, [])
        outs = out_scores.get(idx, [])
        if not ins or not outs:
            continue
        mem = mean(ins) - mean(outs)
        text, group = full_dataset[idx]
        records.append({"text": text, "group": group, "score": float(mem)})

    print(f"\n[CM] Finished. Records with both in/out estimates: {len(records)}/{N}")
    return records


# ============================================================
# Main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--madeup_facts_file", type=str, required=True,
                   help="Text file: one false assertion per line.")
    p.add_argument("--true_facts_file", type=str, required=True,
                   help="Text file: one true assertion per line.")

    p.add_argument("--K_per_group", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_epochs", type=int, default=5)
    p.add_argument("--max_length", type=int, default=128)

    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--lambda_reg", type=float, default=1e-2)
    p.add_argument("--score_every", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--do_targeted_extraction", action="store_true")
    p.add_argument("--prefix_tokens", type=int, default=12)

    p.add_argument("--log_every", type=int, default=50)

    p.add_argument("--do_counterfactual_mem", action="store_true")
    p.add_argument("--cf_num_models", type=int, default=400)
    p.add_argument("--cf_subset_ratio", type=float, default=0.25)
    p.add_argument("--cf_epochs", type=int, default=60)
    p.add_argument("--cf_batch_size", type=int, default=None)
    p.add_argument("--cf_log_every", type=int, default=100)

    p.add_argument("--cf_lr", type=float, default=1e-4)
    p.add_argument("--cf_weight_decay", type=float, default=1e-5)
    p.add_argument("--cf_grad_clip", type=float, default=0.0)

    return p.parse_args()


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Device: {device}, dtype: {dtype}")
    print("Loading pretrained GPT-2 (no training-from-scratch)...")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.train()

    false_assertions = read_facts_file(args.madeup_facts_file)
    true_assertions = read_facts_file(args.true_facts_file)

    print("\nAssertions file stats:")
    print(f"  {LABEL_FALSE} total unique lines: {len(false_assertions)}")
    print(f"  {LABEL_TRUE}  total unique lines: {len(true_assertions)}")
    print(f"  K_per_group requested            : {args.K_per_group}")

    dataset = TwoGroupFactsDataset(
        madeup_facts=false_assertions,
        true_facts=true_assertions,
        K_per_group=args.K_per_group,
        seed=args.seed,
    )

    print("\nFinal constructed dataset:")
    print(f"  {LABEL_FALSE} unique: {len(dataset.false_unique)}")
    print(f"  {LABEL_TRUE}  unique: {len(dataset.true_unique)}")
    print(f"  TOTAL size         : {len(dataset)} (= 2*K)")

    collate = LMCollate(tokenizer, max_length=args.max_length)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    steps_per_epoch = len(loader)
    total_steps = args.num_epochs * steps_per_epoch
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))

    print("\nTraining schedule:")
    print(f"  Epochs              : {args.num_epochs}")
    print(f"  Steps/epoch         : {steps_per_epoch}")
    print(f"  Total steps         : {total_steps}")
    print(f"  Warmup steps        : {warmup_steps}")
    print(f"  Batch size          : {args.batch_size}")
    print(f"  Max length          : {args.max_length}")
    print(f"  score_every         : {args.score_every}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    ghost = GhostFastScores(model, lambda_reg=args.lambda_reg)

    gnq_by_text = defaultdict(list)
    trace_by_text = defaultdict(list)
    group_by_text = {}
    for t in dataset.false_unique:
        group_by_text[t] = GROUP_FALSE
    for t in dataset.true_unique:
        group_by_text[t] = GROUP_TRUE

    global_step = 0
    for epoch in range(args.num_epochs):
        for batch in loader:
            global_step += 1

            texts = batch["texts"]
            groups = batch["groups"]

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)

            ghost.reset_cache()
            out = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            loss.backward()

            if (global_step % args.score_every) == 0:
                gnq_vals, tr_vals = ghost.compute_scores_from_cache(input_ids)
                gnq_vals = gnq_vals.detach().cpu().tolist()
                tr_vals = tr_vals.detach().cpu().tolist()

                for t, g, s_gnq, s_tr in zip(texts, groups, gnq_vals, tr_vals):
                    group_by_text[t] = g
                    if math.isfinite(s_gnq):
                        gnq_by_text[t].append(float(s_gnq))
                    if math.isfinite(s_tr):
                        trace_by_text[t].append(float(s_tr))

            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            scheduler.step()

            if (global_step % args.log_every) == 0 or global_step == 1:
                lr_now = scheduler.get_last_lr()[0]
                print(f"epoch {epoch+1:2d}/{args.num_epochs} | step {global_step:5d}/{total_steps} | loss={loss.item():.4f} | lr={lr_now:.2e}")

    gnq_max, gnq_mean = [], []
    tr_max, tr_mean = [], []

    for text, vals in gnq_by_text.items():
        if vals:
            gnq_max.append({"text": text, "group": group_by_text.get(text, "unknown"), "score": max(vals)})
            gnq_mean.append({"text": text, "group": group_by_text.get(text, "unknown"), "score": mean(vals)})

    for text, vals in trace_by_text.items():
        if vals:
            tr_max.append({"text": text, "group": group_by_text.get(text, "unknown"), "score": max(vals)})
            tr_mean.append({"text": text, "group": group_by_text.get(text, "unknown"), "score": mean(vals)})

    plot_group_hist(gnq_max, "Training-time GNQ (MAX per assertion)", xlabel="GNQ", save_path="hist_gnq_max.png")
    plot_group_hist(tr_max, "Training-time TraceIn-self (MAX per assertion)", xlabel="TraceIn", save_path="hist_tracein_max.png")

    if gnq_max:
        print_group_extremes(gnq_max, metric_name="GNQ")

    def print_stats(name, records_max):
        mx_false = [r["score"] for r in records_max if r["group"] == GROUP_FALSE]
        mx_true = [r["score"] for r in records_max if r["group"] == GROUP_TRUE]

        print(f"\n=== {name} separation (per assertion) ===")
        print(f"Assertions with logs/scores: {len(records_max)}")
        print(f"{LABEL_FALSE} seen: {len(mx_false)} | {LABEL_TRUE} seen: {len(mx_true)}")

        if mx_false and mx_true:
            print(f"\n[{name} | MAX per assertion]")
            print(f"Mean {name}: {LABEL_FALSE}={mean(mx_false):.3e} | {LABEL_TRUE}={mean(mx_true):.3e}")
            print(f"P50  {name}: {LABEL_FALSE}={percentile(mx_false,50):.3e} | {LABEL_TRUE}={percentile(mx_true,50):.3e}")
            print(f"P90  {name}: {LABEL_FALSE}={percentile(mx_false,90):.3e} | {LABEL_TRUE}={percentile(mx_true,90):.3e}")
            print(f"\nTop-K enrichment of {LABEL_FALSE}:")
            for frac, k, rate in topk_enrichment(records_max, target_group=GROUP_FALSE):
                print(f"  top {int(frac*100):2d}% (k={k:5d}) -> {LABEL_FALSE} rate = {rate:.3f}")

    print_stats("TRAINING-TIME GNQ", gnq_max)
    print_stats("TRAINING-TIME TraceIn", tr_max)

    cf_records = None
    if args.do_counterfactual_mem:
        cf_bs = args.cf_batch_size if args.cf_batch_size is not None else args.batch_size

        cf_records = compute_counterfactual_memorization(
            tokenizer=tokenizer,
            full_dataset=dataset,
            collate=collate,
            num_models=args.cf_num_models,
            subset_ratio=args.cf_subset_ratio,
            cf_epochs=args.cf_epochs,
            batch_size=cf_bs,
            cf_lr=args.cf_lr,
            cf_weight_decay=args.cf_weight_decay,
            seed=args.seed,
            log_every=args.cf_log_every,
            grad_clip=args.cf_grad_clip,
        )

        plot_group_hist(
            cf_records,
            "Counterfactual Memorization (per assertion): mem(x)=E_in[M]-E_out[M]",
            xlabel="mem(x)",
            use_log10=False,
            save_path="hist_counterfactual_mem.png",
        )
        print_stats("COUNTERFACTUAL MEMORIZATION", cf_records)

        if cf_records:
            print_group_extremes(cf_records, metric_name="mem(x)")

    if args.do_targeted_extraction and gnq_max and tr_max:
        model.eval()
        targeted_prefix_extraction_eval(
            model=model, tokenizer=tokenizer, records=gnq_max, metric_name="GNQ",
            prefix_tokens=args.prefix_tokens, max_length=args.max_length, seed=args.seed
        )
        targeted_prefix_extraction_eval(
            model=model, tokenizer=tokenizer, records=tr_max, metric_name="TraceIn",
            prefix_tokens=args.prefix_tokens, max_length=args.max_length, seed=args.seed
        )
        if cf_records is not None and len(cf_records) > 0:
            targeted_prefix_extraction_eval(
                model=model, tokenizer=tokenizer, records=cf_records, metric_name="CFMem",
                prefix_tokens=args.prefix_tokens, max_length=args.max_length, seed=args.seed
            )
        model.train()

    print("\n=== Top 20 assertions by MAX GNQ ===")
    for i, r in enumerate(sorted(gnq_max, key=lambda x: x["score"], reverse=True)[:20]):
        grp = LABEL_TRUE if r["group"] == GROUP_TRUE else LABEL_FALSE if r["group"] == GROUP_FALSE else r["group"]
        print(f"[{i:02d}] GNQ={r['score']:.3e} | {grp} | {r['text'][:140]}...")

    print("\n=== Top 20 assertions by MAX TraceIn-self ===")
    for i, r in enumerate(sorted(tr_max, key=lambda x: x["score"], reverse=True)[:20]):
        grp = LABEL_TRUE if r["group"] == GROUP_TRUE else LABEL_FALSE if r["group"] == GROUP_FALSE else r["group"]
        print(f"[{i:02d}] TraceIn={r['score']:.3e} | {grp} | {r['text'][:140]}...")

    if cf_records is not None and len(cf_records) > 0:
        print("\n=== Top 20 assertions by Counterfactual Memorization ===")
        for i, r in enumerate(sorted(cf_records, key=lambda x: x["score"], reverse=True)[:20]):
            grp = LABEL_TRUE if r["group"] == GROUP_TRUE else LABEL_FALSE if r["group"] == GROUP_FALSE else r["group"]
            print(f"[{i:02d}] CFMem={r['score']:.6f} | {grp} | {r['text'][:140]}...")

    print("\nSaved: hist_gnq_max.png, hist_tracein_max.png")
    if cf_records is not None:
        print("Saved: hist_counterfactual_mem.png")
    print("Done.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
