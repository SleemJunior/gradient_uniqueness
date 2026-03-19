#!/usr/bin/env python3
"""
Run example:

python dyck_gnq_pythia.py   --common_facts_file common_knowledge.txt   --dyck_facts_file dyck_knowledge.txt   --model_name EleutherAI/pythia-14m   --K_per_group 150   --batch_size 16   --num_epochs 40   --score_every 1

Notes:
- Default model is the smallest Pythia model: EleutherAI/pythia-14m
- Later you can swap to 70m / 160m / checkpoints with the same exact data
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

from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
torch.set_default_dtype(dtype)

GROUP_COMMON = "common"
GROUP_DYCK = "dyck"
LABEL_COMMON = "common knowledge"
LABEL_DYCK = "Dyck assertions"


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


def read_facts_file(path: str):
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"Facts file not found: {path}")
    lines = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            lines.append(s)
    seen = set()
    out = []
    for s in lines:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


class TwoGroupFactsDataset(Dataset):
    """
    Returns (text, group_str), where group_str in {"common","dyck"}.
    """

    def __init__(self, common_facts, dyck_facts, K_per_group=200, seed=0):
        rng = random.Random(seed)

        if len(common_facts) < K_per_group:
            raise RuntimeError(f"common file too small: have {len(common_facts)}, need {K_per_group}")
        if len(dyck_facts) < K_per_group:
            raise RuntimeError(f"dyck file too small: have {len(dyck_facts)}, need {K_per_group}")

        common_sample = rng.sample(common_facts, k=K_per_group)
        dyck_sample = rng.sample(dyck_facts, k=K_per_group)

        samples = [(t, GROUP_COMMON) for t in common_sample] + [(t, GROUP_DYCK) for t in dyck_sample]
        rng.shuffle(samples)

        self.samples = samples
        self.common_unique = common_sample
        self.dyck_unique = dyck_sample

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class LMCollate:
    """
    Faithful to your original setup:
    full-sequence LM loss on the entire assertion.
    """

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


def print_group_extremes(records, metric_name):
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
    _print_one(f"{LABEL_COMMON} with smallest {metric_name}", _extreme(GROUP_COMMON, "min"))
    _print_one(f"{LABEL_COMMON} with largest  {metric_name}", _extreme(GROUP_COMMON, "max"))
    _print_one(f"{LABEL_DYCK} with smallest {metric_name}", _extreme(GROUP_DYCK, "min"))
    _print_one(f"{LABEL_DYCK} with largest  {metric_name}", _extreme(GROUP_DYCK, "max"))


def plot_group_hist(
    records,
    title,
    xlabel,
    group_a=GROUP_DYCK,
    group_b=GROUP_COMMON,
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
    plt.hist(a_plot, bins=bins, alpha=0.6, density=False, label=LABEL_DYCK)
    plt.hist(b_plot, bins=bins, alpha=0.6, density=False, label=LABEL_COMMON)

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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--common_facts_file", type=str, required=True)
    p.add_argument("--dyck_facts_file", type=str, required=True)

    p.add_argument("--model_name", type=str, default="EleutherAI/pythia-14m")
    p.add_argument("--revision", type=str, default=None)

    p.add_argument("--K_per_group", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_epochs", type=int, default=40)
    p.add_argument("--max_length", type=int, default=128)

    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--lambda_reg", type=float, default=1e-2)
    p.add_argument("--score_every", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=50)

    return p.parse_args()


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Device: {device}, dtype: {dtype}")
    print(f"Loading pretrained model: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, revision=args.revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, revision=args.revision)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.train()

    common_assertions = read_facts_file(args.common_facts_file)
    dyck_assertions = read_facts_file(args.dyck_facts_file)

    print("\nAssertions file stats:")
    print(f"  {LABEL_COMMON} total unique lines: {len(common_assertions)}")
    print(f"  {LABEL_DYCK} total unique lines: {len(dyck_assertions)}")
    print(f"  K_per_group requested          : {args.K_per_group}")

    dataset = TwoGroupFactsDataset(
        common_facts=common_assertions,
        dyck_facts=dyck_assertions,
        K_per_group=args.K_per_group,
        seed=args.seed,
    )

    print("\nFinal constructed dataset:")
    print(f"  {LABEL_COMMON} unique: {len(dataset.common_unique)}")
    print(f"  {LABEL_DYCK} unique: {len(dataset.dyck_unique)}")
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
    for t in dataset.common_unique:
        group_by_text[t] = GROUP_COMMON
    for t in dataset.dyck_unique:
        group_by_text[t] = GROUP_DYCK

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

    plot_group_hist(
        gnq_max,
        "Training-time GNQ (MAX per assertion)",
        xlabel="GNQ",
        save_path="hist_gnq_max.png"
    )
    plot_group_hist(
        tr_max,
        "Training-time TraceIn-self (MAX per assertion)",
        xlabel="TraceIn",
        save_path="hist_tracein_max.png"
    )

    if gnq_max:
        print_group_extremes(gnq_max, metric_name="GNQ")

    def print_stats(name, records_max):
        mx_dyck = [r["score"] for r in records_max if r["group"] == GROUP_DYCK]
        mx_common = [r["score"] for r in records_max if r["group"] == GROUP_COMMON]

        print(f"\n=== {name} separation (per assertion) ===")
        print(f"Assertions with logs/scores: {len(records_max)}")
        print(f"{LABEL_DYCK} seen: {len(mx_dyck)} | {LABEL_COMMON} seen: {len(mx_common)}")

        if mx_dyck and mx_common:
            print(f"\n[{name} | MAX per assertion]")
            print(f"Mean {name}: {LABEL_DYCK}={mean(mx_dyck):.3e} | {LABEL_COMMON}={mean(mx_common):.3e}")
            print(f"P50  {name}: {LABEL_DYCK}={percentile(mx_dyck,50):.3e} | {LABEL_COMMON}={percentile(mx_common,50):.3e}")
            print(f"P90  {name}: {LABEL_DYCK}={percentile(mx_dyck,90):.3e} | {LABEL_COMMON}={percentile(mx_common,90):.3e}")
            print(f"\nTop-K enrichment of {LABEL_DYCK}:")
            for frac, k, rate in topk_enrichment(records_max, target_group=GROUP_DYCK):
                print(f"  top {int(frac*100):2d}% (k={k:5d}) -> {LABEL_DYCK} rate = {rate:.3f}")

    print_stats("TRAINING-TIME GNQ", gnq_max)
    print_stats("TRAINING-TIME TraceIn", tr_max)

    print("\n=== Top 20 assertions by MAX GNQ ===")
    for i, r in enumerate(sorted(gnq_max, key=lambda x: x["score"], reverse=True)[:20]):
        grp = LABEL_COMMON if r["group"] == GROUP_COMMON else LABEL_DYCK
        print(f"[{i:02d}] GNQ={r['score']:.3e} | {grp} | {r['text'][:140]}...")

    print("\n=== Top 20 assertions by MAX TraceIn-self ===")
    for i, r in enumerate(sorted(tr_max, key=lambda x: x["score"], reverse=True)[:20]):
        grp = LABEL_COMMON if r["group"] == GROUP_COMMON else LABEL_DYCK
        print(f"[{i:02d}] TraceIn={r['score']:.3e} | {grp} | {r['text'][:140]}...")

    print("\nSaved: hist_gnq_max.png, hist_tracein_max.png")
    print("Done.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
