#!/usr/bin/env python3
"""
HOW TO RUN:
python mem_gnq_ckpt.py \
  --model_name cglez/gpt2-ag_news \
  --revisions epoch-1,epoch-10,epoch-20,epoch-30,epoch-40,epoch-50,epoch-60,epoch-70,epoch-80,epoch-90,epoch-100 \
  --num_points 500 \
  --batch_size 8 \
  --max_length 128 \
  --do_attack \
  --attack_revision epoch-100 \
  --prefix_tokens 45 \
  --match_threshold 0.98 \
  --out_dir out_gnq_ckpt_all
"""

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import GPT2Tokenizer, GPT2LMHeadModel

try:
    from datasets import load_dataset
except Exception as e:
    raise RuntimeError(
        "This script requires the `datasets` package. Install with:\n"
        "  pip install datasets\n"
        f"Original error: {e}"
    )

# -----------------------
# Device / dtype
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
torch.set_default_dtype(dtype)

# -----------------------
# Plot font scaling (paper-like)
# -----------------------
PLOT_FONT_SCALE = 2 
BASE_FS = int(10 * PLOT_FONT_SCALE)
TITLE_FS = int(11 * PLOT_FONT_SCALE)
LEGEND_FS = int(8 * PLOT_FONT_SCALE)
TICK_FS = int(9 * PLOT_FONT_SCALE)


def _print_saved(msg: str):
    # Keep your "Saved ..." prints.
    if msg.startswith("Saved plot:") or msg.startswith("Saved CSV:") or msg.startswith("Saved attack CSV:"):
        print(msg)


# ============================================================
# EXACT GNQ IMPLEMENTATION
# ============================================================
class GhostFastScores:
    """
    Computes:
      - GNQ = h/(1-h), where h = diag((K+λI)^-1 K)
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

    def compute_scores_from_cache(self, batch_x, attention_mask=None):
        """
        IMPORTANT: padding fix
          - if attention_mask is provided (B,T), we zero-out padded token rows
            in both X and delta BEFORE flattening. This prevents pad tokens from
            contributing to K.
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

            # remove padding tokens
            if attention_mask is not None and X.dim() >= 3:
                m = attention_mask.to(X.dtype).unsqueeze(-1)  # [B, T, 1]
                X = X * m
                if delta.dim() >= 3:
                    delta = delta * m

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
# EXTRACTION ATTACK: prefix -> greedy continuation; score by token match fraction
# ============================================================
@torch.no_grad()
def greedy_complete_from_prefix_with_match(
    model,
    tokenizer,
    text: str,
    *,
    prefix_tokens: int = 12,
    max_length: int = 128,
) -> Tuple[float, str, str, str]:
    """
    Returns:
      match_frac
      prefix_text: decoded prefix (first prefix_tokens)
      gen_text: decoded generated text (aligned/truncated)
      orig_text: decoded original text (aligned/truncated)
    """
    enc = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    ids = enc["input_ids"].to(device)

    if ids.size(1) <= prefix_tokens:
        return 0.0, "", "", ""

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

    gen_ids = gen[0]
    orig_ids = ids[0]

    # Align to same length to compare fairly
    L = min(gen_ids.numel(), orig_ids.numel())
    if L <= prefix_tokens:
        return 0.0, "", "", ""

    gen_ids = gen_ids[:L]
    orig_ids = orig_ids[:L]

    gen_cont = gen_ids[prefix_tokens:]
    orig_cont = orig_ids[prefix_tokens:]
    match_frac = float((gen_cont == orig_cont).float().mean().item())

    prefix_text = tokenizer.decode(prefix[0], skip_special_tokens=True)
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    orig_text = tokenizer.decode(orig_ids, skip_special_tokens=True)
    return match_frac, prefix_text, gen_text, orig_text


# ============================================================
# Data: fixed 500 training points from AG-News train
# ============================================================
def get_agnews_text(ex) -> str:
    if isinstance(ex, dict):
        if "text" in ex and isinstance(ex["text"], str):
            return ex["text"].strip()
        title = ex.get("title", "")
        desc = ex.get("description", "")
        combo = (str(title) + " " + str(desc)).strip()
        if combo:
            return combo
    return str(ex).strip()


@dataclass(frozen=True)
class Point:
    ag_idx: int
    text: str


class FixedPointsDataset(Dataset):
    def __init__(self, points: List[Point]):
        self.points = points

    def __len__(self):
        return len(self.points)

    def __getitem__(self, i):
        p = self.points[i]
        return p.text, p.ag_idx, i  # (text, original idx, local idx)


class LMCollate:
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        texts, ag_idxs, local_idxs = zip(*batch)
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
            "ag_idxs": list(ag_idxs),
            "local_idxs": list(local_idxs),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ============================================================
# Utilities / plotting
# ============================================================
def ensure_out_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_epoch(rev: str) -> int:
    try:
        if "epoch-" in rev:
            return int(rev.split("epoch-")[-1].split()[0].split("/")[0])
    except Exception:
        pass
    return -1


def plot_attack_success_and_avg_gnq_by_quantile(
    success: List[int],
    gnq_total: List[float],
    out_path: Path,
    title_suffix: str,
    num_bins: int = 10,
    use_median_gnq: bool = False,  # False => average GNQ, True => median GNQ
):
    """
    Advisor-requested figure:
      - bars: attack success rate per quantile bin
      - line (right y-axis): average/median GNQ in the bin

    NOTE: Higher bin index = higher GNQ.
    """
    assert num_bins >= 2

    idx = [i for i, x in enumerate(gnq_total) if math.isfinite(x)]
    scores = [float(gnq_total[i]) for i in idx]
    succ = [int(success[i]) for i in idx]
    n = len(scores)
    if n == 0:
        return

    order = sorted(range(n), key=lambda i: scores[i])  # ascending GNQ
    bin_id = [0] * n
    for rank, i in enumerate(order):
        d = min(num_bins - 1, int(num_bins * rank / max(1, n)))
        bin_id[i] = d

    rates = []
    gnq_stat = []
    counts = []
    for d in range(num_bins):
        ids = [i for i in range(n) if bin_id[i] == d]
        counts.append(len(ids))
        if ids:
            rates.append(sum(succ[i] for i in ids) / len(ids))
            vals = [scores[i] for i in ids]
            if use_median_gnq:
                vals_sorted = sorted(vals)
                gnq_stat.append(vals_sorted[len(vals_sorted) // 2])
            else:
                gnq_stat.append(sum(vals) / len(vals))
        else:
            rates.append(float("nan"))
            gnq_stat.append(float("nan"))

    xs = list(range(1, num_bins + 1))
    labels = [f"B{i}" for i in xs]  # B1 lowest ... B{num_bins} highest

    fig = plt.figure(figsize=(9.2, 5.4))
    ax1 = plt.gca()

    ax1.bar(xs, rates, alpha=0.85)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xlabel(f"GNQ$_{{\\mathrm{{total}}}}$ quantile bin (B{num_bins} = highest GNQ)", fontsize=BASE_FS)
    ax1.set_ylabel("Attack success rate", fontsize=BASE_FS)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels, fontsize=TICK_FS)
    ax1.tick_params(axis="y", labelsize=TICK_FS)
    ax1.grid(True, axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(xs, gnq_stat, marker="o")
    ax2.set_ylabel("Median GNQ" if use_median_gnq else "Average GNQ", fontsize=BASE_FS)
    ax2.tick_params(axis="y", labelsize=TICK_FS)

    # plt.title(f"Attack success rate and GNQ by GNQ$_{{\\mathrm{{total}}}}$ quantile{title_suffix}", fontsize=TITLE_FS)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    _print_saved(f"Saved plot: {out_path}")


# ============================================================
# GNQ per checkpoint
# ============================================================
def compute_gnq_for_revision(
    model_name: str,
    revision: str,
    loader: DataLoader,
    *,
    lambda_reg: float,
) -> List[float]:
    print(f"\n[GNQ] Loading {model_name} @ {revision}")
    model = GPT2LMHeadModel.from_pretrained(model_name, revision=revision)
    model.config.pad_token_id = loader.collate_fn.tokenizer.pad_token_id
    model.to(device)
    model.train()

    ghost = GhostFastScores(model, lambda_reg=lambda_reg)

    N = len(loader.dataset)
    gnq_scores = [float("nan")] * N

    for bi, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        local_idxs = batch["local_idxs"]

        model.zero_grad(set_to_none=True)
        ghost.reset_cache()

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss
        loss.backward()

        with torch.no_grad():
            gnq_vals, _ = ghost.compute_scores_from_cache(input_ids, attention_mask=attention_mask)
            gnq_vals = gnq_vals.detach().cpu().tolist()

        for li, s in zip(local_idxs, gnq_vals):
            gnq_scores[int(li)] = float(s)

        if bi == 0 or (bi % 50) == 0:
            print(f"[GNQ] batch {bi:4d}/{len(loader)} | loss={loss.item():.4f}")

    del ghost
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    missing = sum(1 for x in gnq_scores if not math.isfinite(x))
    if missing:
        print(f"[GNQ][WARN] Missing scores for {missing}/{N} points.")
    return gnq_scores


# ============================================================
# Plot trajectories
# ============================================================
def plot_trajectories_and_print_key_texts(
    epochs: List[int],
    trajs: Dict[int, List[float]],
    points: List[Point],
    out_path: Path,
    title: str,
):
    """
    """
    items = []
    for local_idx, ys in trajs.items():
        ys_clean = [float(y) for y in ys if math.isfinite(y)]
        if not ys_clean:
            continue
        final = float(ys[-1]) if math.isfinite(ys[-1]) else ys_clean[-1]
        avg = sum(ys_clean) / len(ys_clean)
        items.append((local_idx, final, avg))

    if items:
        items_sorted_final = sorted(items, key=lambda t: t[1], reverse=True)
        top2 = items_sorted_final[:2]
        rest = [t for t in items_sorted_final[2:]]
        rest_sorted_low_avg = sorted(rest, key=lambda t: t[2])
        bottom2 = rest_sorted_low_avg[:2] if len(rest_sorted_low_avg) >= 2 else rest_sorted_low_avg

        def _print_text(tag: str, li: int, final: float, avg: float):
            txt = points[li].text.replace("\n", " ")
            print(f"\n[{tag}] local_idx={li} | AGidx={points[li].ag_idx} | final_GNQ={final:.6g} | avg_GNQ={avg:.6g}")
            print(f"Text: {txt}")

        for k, (li, final, avg) in enumerate(top2, 1):
            _print_text(f"Top increasing line #{k}", li, final, avg)

        for k, (li, final, avg) in enumerate(bottom2, 1):
            _print_text(f"Lowest line among rest #{k}", li, final, avg)

    plt.figure(figsize=(10, 6))  

    for local_idx, ys in trajs.items():
        lbl = f"pt{local_idx} (AGidx={points[local_idx].ag_idx})"
        plt.plot(epochs, ys, marker="o", linewidth=1.6, label=lbl)

    plt.xlabel("Checkpoint epoch", fontsize=BASE_FS)
    plt.ylabel("GNQ", fontsize=BASE_FS)
    plt.title(title, fontsize=TITLE_FS)
    plt.grid(True, alpha=0.25)
    plt.xticks(fontsize=TICK_FS)
    plt.yticks(fontsize=TICK_FS)

    plt.legend(
        fontsize=LEGEND_FS,
        ncol=4,                 
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    _print_saved(f"Saved plot: {out_path}")


# ============================================================
# Attack
# ============================================================
@torch.no_grad()
def run_attack_on_points(
    model_name: str,
    revision: str,
    tokenizer: GPT2Tokenizer,
    points: List[Point],
    *,
    prefix_tokens: int,
    max_length: int,
    match_threshold: float,
) -> Tuple[List[int], List[float], List[str], List[str]]:
    """
    Returns:
      success: 0/1
      match_scores
      prefix_texts
      gen_texts
    (orig text is points[i].text)
    """
    print(f"\n[ATTACK] Loading {model_name} @ {revision}")
    model = GPT2LMHeadModel.from_pretrained(model_name, revision=revision)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()

    success: List[int] = []
    match_scores: List[float] = []
    prefix_texts: List[str] = []
    gen_texts: List[str] = []

    for i, p in enumerate(points):
        match, prefix_txt, gen_txt, _orig_txt = greedy_complete_from_prefix_with_match(
            model, tokenizer, p.text,
            prefix_tokens=prefix_tokens,
            max_length=max_length,
        )
        match_scores.append(float(match))
        success.append(1 if match >= match_threshold else 0)
        prefix_texts.append(prefix_txt)
        gen_texts.append(gen_txt)

        if i == 0 or (i % 50) == 0:
            print(f"[ATTACK] {i:4d}/{len(points)} done...")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return success, match_scores, prefix_texts, gen_texts


def print_four_attack_examples(
    tokenizer: GPT2Tokenizer,
    points: List[Point],
    gnq_total: List[float],
    success: List[int],
    prefix_texts: List[str],
    gen_texts: List[str],
):
    """
    Advisor-requested examples:
      (1) high GNQ success
      (2) low  GNQ success
      (3) high GNQ failure
      (4) low  GNQ failure
    For each: show prefix and model output.
    """
    idx_all = [i for i, g in enumerate(gnq_total) if math.isfinite(g)]
    if not idx_all:
        return

    succ_idx = [i for i in idx_all if success[i] == 1]
    fail_idx = [i for i in idx_all if success[i] == 0]

    def pick_high_low(cands: List[int]) -> Tuple[Optional[int], Optional[int]]:
        if not cands:
            return None, None
        hi = max(cands, key=lambda i: gnq_total[i])
        lo = min(cands, key=lambda i: gnq_total[i])
        return hi, lo

    hi_succ, lo_succ = pick_high_low(succ_idx)
    hi_fail, lo_fail = pick_high_low(fail_idx)

    def show(tag: str, i: Optional[int]):
        if i is None:
            print(f"\n[{tag}] NONE (no examples in this category)")
            return
        gnq = float(gnq_total[i])
        pref = prefix_texts[i].replace("\n", " ")
        gen = gen_texts[i].replace("\n", " ")
        orig = points[i].text.replace("\n", " ")
        print(f"\n[{tag}] local_idx={i} | GNQ_total={gnq:.6g} | AGidx={points[i].ag_idx}")
        print(f"Prefix ({len(tokenizer(pref, add_special_tokens=False)['input_ids'])} toks decoded): {pref}")
        print(f"Model output (full decoded): {gen}")
        print(f"Reference text (full decoded): {orig}")

    show("1) High-GNQ that IS memorized (success)", hi_succ)
    show("2) Low-GNQ that IS memorized (success)", lo_succ)
    show("3) High-GNQ that is NOT memorized (failure)", hi_fail)
    show("4) Low-GNQ that is NOT memorized (failure)", lo_fail)


# ============================================================
# Main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="cglez/gpt2-ag_news")
    p.add_argument(
        "--revisions",
        type=str,
        default="epoch-1,epoch-10,epoch-20,epoch-30,epoch-40,epoch-50,epoch-60,epoch-70,epoch-80,epoch-90,epoch-100",
    )
    p.add_argument("--num_points", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--lambda_reg", type=float, default=1e-2)

    p.add_argument("--num_plot_points", type=int, default=20)
    p.add_argument("--out_dir", type=str, default="out_gnq_ckpt")

    p.add_argument("--do_attack", action="store_true")
    p.add_argument("--attack_revision", type=str, default=None)
    p.add_argument("--prefix_tokens", type=int, default=12)
    p.add_argument("--match_threshold", type=float, default=0.90)

    p.add_argument("--attack_bins", type=int, default=5, help="Number of quantile bins for the attack plot.")
    p.add_argument("--use_median_gnq", action="store_true", help="If set, plot median GNQ (else mean) on RHS.")

    return p.parse_args()


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = ensure_out_dir(args.out_dir)

    revisions = [r.strip() for r in args.revisions.split(",") if r.strip()]
    if not revisions:
        raise ValueError("Empty --revisions.")
    attack_revision = args.attack_revision or revisions[-1]

    epochs = [parse_epoch(r) for r in revisions]

    print(f"Device: {device}, dtype: {dtype}")
    print(f"Model: {args.model_name}")
    print(f"Revisions: {revisions}")
    print(f"Sampling {args.num_points} points from AG-News train split...")

    ds = load_dataset("ag_news", split="train")
    N_train = len(ds)
    if args.num_points > N_train:
        raise ValueError(f"--num_points {args.num_points} > train size {N_train}")

    sample_idxs = sorted(random.sample(range(N_train), k=args.num_points))
    points: List[Point] = [Point(ag_idx=int(idx), text=get_agnews_text(ds[int(idx)])) for idx in sample_idxs]

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    fixed_ds = FixedPointsDataset(points)
    collate = LMCollate(tokenizer, max_length=args.max_length)
    collate.tokenizer = tokenizer

    loader = DataLoader(
        fixed_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        drop_last=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # GNQ per revision
    gnq_by_rev: Dict[str, List[float]] = {}
    for rev in revisions:
        gnq_by_rev[rev] = compute_gnq_for_revision(
            model_name=args.model_name,
            revision=rev,
            loader=loader,
            lambda_reg=args.lambda_reg,
        )

    # trajectories plot + print requested texts
    k_plot = min(args.num_plot_points, args.num_points)
    plot_local = random.sample(range(args.num_points), k=k_plot)
    trajs: Dict[int, List[float]] = {li: [gnq_by_rev[rev][li] for rev in revisions] for li in plot_local}
    plot_trajectories_and_print_key_texts(
        epochs=epochs,
        trajs=trajs,
        points=points,
        out_path=out_dir / "gnq_trajectories.png",
        title=f"GNQ trajectories across checkpoints ({args.model_name})",
    )

    # Total GNQ per point
    gnq_total: List[float] = []
    for li in range(args.num_points):
        s = 0.0
        ok = True
        for rev in revisions:
            v = gnq_by_rev[rev][li]
            if not math.isfinite(v):
                ok = False
                break
            s += float(v)
        gnq_total.append(s if ok else float("nan"))

    # Save GNQ CSV
    csv_path = out_dir / "gnq_scores_checkpoints.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["local_idx", "ag_train_idx", "gnq_total"] + [f"gnq@{rev}" for rev in revisions] + ["text_snip"])
        for li, p in enumerate(points):
            w.writerow(
                [li, p.ag_idx, gnq_total[li]]
                + [gnq_by_rev[rev][li] for rev in revisions]
                + [p.text[:140].replace("\n", " ")]
            )
    _print_saved(f"Saved CSV: {csv_path}")

    # Attack
    if args.do_attack:
        success, match_scores, prefix_texts, gen_texts = run_attack_on_points(
            model_name=args.model_name,
            revision=attack_revision,
            tokenizer=tokenizer,
            points=points,
            prefix_tokens=args.prefix_tokens,
            max_length=args.max_length,
            match_threshold=args.match_threshold,
        )

        # advisor examples (prints)
        print_four_attack_examples(
            tokenizer=tokenizer,
            points=points,
            gnq_total=gnq_total,
            success=success,
            prefix_texts=prefix_texts,
            gen_texts=gen_texts,
        )

        # combined plot: bars (success rate) + line (avg/median GNQ)
        title_suffix = f" @match≥{args.match_threshold:.2f}"
        plot_attack_success_and_avg_gnq_by_quantile(
            success=success,
            gnq_total=gnq_total,
            out_path=out_dir / "success_rate_by_gnq_decile.png",
            title_suffix=title_suffix,
            num_bins=args.attack_bins,
            use_median_gnq=args.use_median_gnq,
        )

        # Save attack CSV (kept)
        attack_csv = out_dir / "attack_vs_gnq_total.csv"
        with attack_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "local_idx", "ag_train_idx", "attack_success", "match_frac", "gnq_total",
                "prefix_text", "gen_text_snip", "text_snip"
            ])
            for li, p in enumerate(points):
                w.writerow([
                    li,
                    p.ag_idx,
                    success[li],
                    match_scores[li],
                    gnq_total[li],
                    prefix_texts[li][:180].replace("\n", " "),
                    gen_texts[li][:180].replace("\n", " "),
                    p.text[:180].replace("\n", " "),
                ])
        _print_saved(f"Saved attack CSV: {attack_csv}")

    print("\nDone.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
