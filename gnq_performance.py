import time
import copy

import torch
import torch.nn as nn
from torch.func import functional_call, vmap, grad
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


try:
    from datasets import load_dataset
    HAVE_DATASETS = True
except ImportError:
    HAVE_DATASETS = False

# ============================================================
# 0. Device / dtype setup
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64  
torch.set_default_dtype(dtype)

print(f"Using device: {device}, dtype: {dtype}")

NAIVE_P_MAX = 20_000  


# ============================================================
# Helpers: GPU memory measurement
# ============================================================
def start_mem_measure():
    """Reset and record baseline allocated memory (GPU only)."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        return torch.cuda.memory_allocated(device)
    return 0


def end_mem_measure(start_alloc):
    """Return peak memory (MB) used since start_mem_measure."""
    if device.type == "cuda":
        max_alloc = torch.cuda.max_memory_allocated(device)
        used = max_alloc - start_alloc
        return used / (1024 ** 2)  # MB
    return 0.0


# ============================================================
# 1. GhostFastGNQ 
# ============================================================
class GhostFastGNQ:
    """
    Computes GNQ using the 'Ghost Kernel' trick.
    Avoids building full gradient matrix G (B x P).
    Works directly with activations X and backprop errors delta.

    Typical usage in a real training loop (GPT-2 or MLP):
      - Attach hooks once.
      - During the usual training step, run forward + backward.
      - After backward (no extra backward), call compute_gnq_from_cache(x)
        to build K and solve for GNQ using cached activations/backprops.
    """
    def __init__(self, model, lambda_reg=1e-4):
        self.model = model  
        self.lambda_reg = lambda_reg

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
        """Clear cached activations/backprops before a new training step."""
        self.activations.clear()
        self.backprops.clear()

    def _run_fwd_bwd(self, batch_x, batch_y, loss_closure):
        """
        Shared forward+backward for GhostFastGNQ when used *standalone*
        (extra backward). Kept for completeness; we avoid this in timing
        when we want 'no extra backward'.
        """
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        self.reset_cache()

        if batch_x.is_floating_point():
            old_req = batch_x.requires_grad
            batch_x.requires_grad_(True)
        else:
            old_req = None

        self.model.zero_grad(set_to_none=True)
        loss = loss_closure(self.model, batch_x, batch_y)
        loss.backward()

        if old_req is not None:
            batch_x.requires_grad_(old_req)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        return t0, t1

    def _kernel_and_solve(self, batch_x, t0, t1):
        """
        Build K from hooks and solve GNQ.
        """
        B = batch_x.shape[0]
        param_dtype = next(self.model.parameters()).dtype

        K_total = torch.zeros((B, B),
                              device=batch_x.device,
                              dtype=param_dtype)

        for layer in self.model.modules():
            if isinstance(layer, nn.Linear):
                if layer not in self.activations or layer not in self.backprops:
                    continue

                X = self.activations[layer]     
                delta = self.backprops[layer]    

                if X.dim() > 2:
                    X = X.flatten(1)
                if delta.dim() > 2:
                    delta = delta.flatten(1)

                K_act = X @ X.T         
                K_err = delta @ delta.T  

                if layer.bias is not None:
                    K_layer = K_err * (K_act + 1.0)
                else:
                    K_layer = K_err * K_act

                K_total.add_(K_layer)

        self.reset_cache()

        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        eye_B = torch.eye(B, device=batch_x.device, dtype=K_total.dtype)
        K_reg = K_total + self.lambda_reg * eye_B
        M = torch.linalg.solve(K_reg, K_total)
        h = torch.diagonal(M)
        h = torch.clamp(h, max=1.0 - 1e-9)
        gnq_values = h / (1.0 - h)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t3 = time.perf_counter()

        timings = {
            "extract": t1 - t0,
            "kernel": t2 - t1,
            "solve": t3 - t2,
            "total": t3 - t0,
        }
        return gnq_values, timings

    def compute_gnq(self, params, buffers, batch_x, batch_y):
        def cls_loss(model, bx, by):
            logits = model(bx)
            return nn.CrossEntropyLoss(reduction="sum")(logits, by)

        t0, t1 = self._run_fwd_bwd(batch_x, batch_y, cls_loss)
        gnq_values, timings = self._kernel_and_solve(batch_x, t0, t1)
        return gnq_values, None, timings

    def compute_gnq_with_loss(self, batch_x, batch_y, loss_closure):
        t0, t1 = self._run_fwd_bwd(batch_x, batch_y, loss_closure)
        gnq_values, timings = self._kernel_and_solve(batch_x, t0, t1)
        return gnq_values, None, timings

    def compute_gnq_from_cache(self, batch_x):
        """
        Use cached activations/backprops from a *previous* forward+backward
        (e.g., the real training iteration). No extra backward.

        Call sequence:
          - ghost_calc.reset_cache()
          - forward (model(...))
          - loss.backward()   # hooks fire, filling activations/backprops
          - gnq_vals, timings = ghost_calc.compute_gnq_from_cache(batch_x)
        """
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        # Pass t0 as both t0 and t1 so 'extract' = 0, and total = kernel+solve
        gnq_values, timings = self._kernel_and_solve(batch_x, t0, t0)
        return gnq_values, timings

    def __del__(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass


# ============================================================
# 2. FastGNQ (vmap + functional_call; explicit G). NOT MENTIONED IN THE PAPER.
# ============================================================
class FastGNQ:
    """
    Fast GNQ using functorch vmap to compute per-sample gradients
    and explicit kernel K = G G^T.
    """
    def __init__(self, model, lambda_reg=1e-4):
        self.model = model
        self.lambda_reg = lambda_reg

    def compute_gnq(self, params, buffers, batch_x, batch_y):

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        def compute_loss_stateless(params, buffers, x, y):
            pred = functional_call(self.model, (params, buffers), (x.unsqueeze(0),))
            return nn.CrossEntropyLoss()(pred, y.unsqueeze(0))

        compute_grad = grad(compute_loss_stateless)
        compute_sample_grads = vmap(compute_grad, in_dims=(None, None, 0, 0))
        G_dict = compute_sample_grads(params, buffers, batch_x, batch_y)

        G = torch.cat(
            [g.view(batch_x.shape[0], -1) for g in G_dict.values()],
            dim=1
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        K = G @ G.T

        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        B = K.shape[0]
        eye_B = torch.eye(B, device=K.device, dtype=K.dtype)
        K_reg = K + self.lambda_reg * eye_B
        M = torch.linalg.solve(K_reg, K)
        h = torch.diagonal(M)
        h = torch.clamp(h, max=1.0 - 1e-9)
        gnq_values = h / (1.0 - h)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t3 = time.perf_counter()

        timings = {
            "extract": t1 - t0,
            "kernel": t2 - t1,
            "solve": t3 - t2,
            "total": t3 - t0,
        }
        return gnq_values, G, timings


# ============================================================
# 3. Naive GNQ
# ============================================================
def naive_gnq_full_pipeline(model, batch_x, batch_y, lambda_reg=1e-4):
    """
    Naive GNQ: exactly what you'd *conceptually* do in theory:
    - Compute per-sample gradients by looping over examples.
    - For each j, build S_j = sum_{k!=j} g_k g_k^T + lambda I
    - Invert S_j and compute g_j^T S_j^{-1} g_j.

    This is purely for reference / validation on small models.
    """
    # model is assumed already on device/dtype
    model.train()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    B = batch_x.shape[0]
    grads_list = []
    criterion = nn.CrossEntropyLoss(reduction="sum")

    # Phase 1: per-sample gradients
    for i in range(B):
        model.zero_grad(set_to_none=True)
        out_i = model(batch_x[i].unsqueeze(0))
        loss_i = criterion(out_i, batch_y[i].unsqueeze(0))
        loss_i.backward()

        g_i = []
        for p in model.parameters():
            if p.grad is not None:
                g_i.append(p.grad.view(-1))
        grads_list.append(torch.cat(g_i))

    G = torch.stack(grads_list)  # [B, P]

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    time_extract = t1 - t0

    # Phase 2: GNQ calc with explicit S_j inversion
    N_p = G.shape[1]
    gnq_values = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    calc_start = time.perf_counter()

    eye_P = torch.eye(N_p, device=G.device, dtype=G.dtype)

    for j in range(B):
        g_j = G[j].unsqueeze(1)  # [P, 1]

        S = torch.zeros((N_p, N_p), device=G.device, dtype=G.dtype)
        for k in range(B):
            if k == j:
                continue
            g_k = G[k].unsqueeze(1)
            S += g_k @ g_k.T
        S += lambda_reg * eye_P

        S_inv = torch.linalg.inv(S)
        gnq_j = (g_j.T @ S_inv @ g_j).item()
        gnq_values.append(gnq_j)

    if device.type == "cuda":
        torch.cuda.synchronize()
    time_calc = time.perf_counter() - calc_start

    timings = {
        "extract": time_extract,
        "calc": time_calc,
        "total": time_extract + time_calc,
    }
    return torch.tensor(gnq_values, device=G.device, dtype=G.dtype), G, timings


# ============================================================
# 3.1 Diagonal GNQ approximation (NOT MENTIONED IN THE PAPER.)
# ============================================================
def diag_gnq_full_pipeline(model, batch_x, batch_y, epsilon=1e-2):
    """
    Diagonal approximation of GNQ (your D_ij) on a *single batch*:

      1) Loop over samples, compute per-sample gradients g_i.
      2) Accumulate diag_S_total = sum_i g_i^2.
      3) For each j, approximate S_j by its diagonal:
            diag_S_loo_j = diag_S_total - g_j^2 + epsilon
         and compute
            D_j = sum_p g_{j,p}^2 / diag_S_loo_j[p].

    This uses a loop over samples just like Naive (no reuse of G_fast),
    so the extract time reflects the cost of per-sample backprops.
    """
    model.train()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    B = batch_x.shape[0]
    grads_list = []
    criterion = nn.CrossEntropyLoss(reduction="sum")

    diag_S_total = None
    grad_dim = None

    # Phase 1: per-sample gradients + diagonal accumulation
    for i in range(B):
        model.zero_grad(set_to_none=True)
        out_i = model(batch_x[i].unsqueeze(0))
        loss_i = criterion(out_i, batch_y[i].unsqueeze(0))
        loss_i.backward()

        g_i = []
        for p in model.parameters():
            if p.grad is not None:
                g_i.append(p.grad.view(-1))
        g_i = torch.cat(g_i)  # [P]

        if grad_dim is None:
            grad_dim = g_i.numel()
            diag_S_total = torch.zeros(grad_dim, device=g_i.device, dtype=g_i.dtype)

        grads_list.append(g_i)
        diag_S_total += g_i ** 2

    G = torch.stack(grads_list)  # [B, P]

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    time_extract = t1 - t0

    # Phase 2: approximate D_j using diagonal of S_j
    if device.type == "cuda":
        torch.cuda.synchronize()
    calc_start = time.perf_counter()

    D_ij_list = []
    for j in range(B):
        g_j = G[j]  # [P]
        diag_S_loo = diag_S_total - g_j ** 2
        diag_S_loo += epsilon  # add ε once after exclusion

        D_j = torch.sum((g_j ** 2) / diag_S_loo).item()
        D_ij_list.append(D_j)

    if device.type == "cuda":
        torch.cuda.synchronize()
    time_calc = time.perf_counter() - calc_start

    timings = {
        "extract": time_extract,
        "calc": time_calc,
        "total": time_extract + time_calc,
    }
    D_ij_tensor = torch.tensor(D_ij_list, device=batch_x.device, dtype=batch_x.dtype)
    return D_ij_tensor, G, timings


# ============================================================
# 4. Helpers: build MLP and MNIST loader
# ============================================================
def build_mlp(input_dim, hidden_dims, output_dim):
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    model = nn.Sequential(*layers)
    return model.to(device=device, dtype=dtype)


def make_mnist_loader(batch_size, resize_to=None):
    """
    Returns a real MNIST train DataLoader, as in normal training:
    - train=True
    - shuffled
    - normalized
    """
    tfms = [transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]
    if resize_to is not None:
        tfms.insert(0, transforms.Resize(resize_to))

    transform = transforms.Compose(tfms)

    dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, drop_last=True)
    return loader


def warmup_mlp_and_get_batch(model, loader, warmup_steps=10, lr=0.1):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction="sum")

    data_iter = iter(loader)

    def next_batch():
        nonlocal data_iter
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)
        x = x.view(x.shape[0], -1).to(device=device, dtype=dtype)
        y = y.to(device=device)
        return x, y

    for step in range(warmup_steps):
        x, y = next_batch()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    x_gnq, y_gnq = next_batch()
    return x_gnq, y_gnq


# ============================================================
# 5. Run experiments for MLP regimes (no extra backward for Ghost)
# ============================================================
def run_regime(name, input_dim, hidden_dims, output_dim,
               batch_size, resize_to,
               lambda_reg=1e-2,
               warmup_steps_mlp=10,
               diag_epsilon=1e-2):
    print("\n" + "#" * 80)
    print(f"REGIME: {name}")
    print("#" * 80)

    torch.manual_seed(42)


    loader = make_mnist_loader(batch_size=batch_size, resize_to=resize_to)

    base_model = build_mlp(input_dim, hidden_dims, output_dim)

    x, y = warmup_mlp_and_get_batch(
        base_model, loader, warmup_steps=warmup_steps_mlp, lr=0.1
    )

    total_params = sum(p.numel() for p in base_model.parameters())

    print(f"Batch Size: {batch_size}")
    print(f"Model: MLP({input_dim} -> {hidden_dims} -> {output_dim})")
    print(f"Total Parameters (N_p): {total_params}")
    print(f"Naive GNQ allowed up to N_p <= {NAIVE_P_MAX}")
    print(f"Warmup steps (MLP): {warmup_steps_mlp}")
    print("-" * 90)

    # Clone the *trained* base_model, so all methods see the same weights
    model_naive = copy.deepcopy(base_model).train()
    model_fast = copy.deepcopy(base_model).train()
    model_ghost = copy.deepcopy(base_model).train()
    model_diag = copy.deepcopy(base_model).train()

    params_fast = dict(model_fast.named_parameters())
    buffers_fast = dict(model_fast.named_buffers())

    results = {}

    # ---------- GhostFastGNQ (no extra backward counted) ----------
    print("Running GhostFastGNQ (hooks, no extra backward in timing)...")
    ghost_calc = GhostFastGNQ(model_ghost, lambda_reg=lambda_reg)

    ghost_calc.reset_cache()
    criterion_ghost = nn.CrossEntropyLoss(reduction="sum")
    model_ghost.zero_grad(set_to_none=True)
    logits_ghost = model_ghost(x)
    loss_ghost = criterion_ghost(logits_ghost, y)
    loss_ghost.backward()  # hooks fill activations/backprops

    mem_start = start_mem_measure()
    try:
        ghost_gnq, t_ghost = ghost_calc.compute_gnq_from_cache(x)
        mem_mb = end_mem_measure(mem_start)
        results["ghost"] = {
            "status": "ok",
            "gnq": ghost_gnq,
            "timings": t_ghost,
            "mem_mb": mem_mb,
        }
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        results["ghost"] = {"status": "OOM", "gnq": None, "timings": None, "mem_mb": None}
        print("  [GhostFastGNQ] OOM on this regime.")

    # ---------- FastGNQ (vmap) ----------
    print("Warming up FastGNQ (vmap)...")
    fast_calc = FastGNQ(model_fast, lambda_reg=lambda_reg)
    try:
        with torch.no_grad():
            fast_calc.compute_gnq(params_fast, buffers_fast, x, y)

        print("Running FastGNQ (vmap)...")
        mem_start = start_mem_measure()
        fast_gnq, G_fast, t_fast = fast_calc.compute_gnq(
            params_fast, buffers_fast, x, y
        )
        mem_mb = end_mem_measure(mem_start)
        results["fast"] = {
            "status": "ok",
            "gnq": fast_gnq,
            "timings": t_fast,
            "mem_mb": mem_mb,
        }
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        results["fast"] = {"status": "OOM", "gnq": None, "timings": None, "mem_mb": None}
        print("  [FastGNQ] OOM on this regime.")

    # ---------- Naive GNQ ----------
    if total_params > NAIVE_P_MAX:
        print("Skipping Naive GNQ: model too large for P^2 algorithm.")
        results["naive"] = {
            "status": f"skipped (N_p={total_params} > {NAIVE_P_MAX})",
            "gnq": None,
            "timings": None,
            "mem_mb": None,
        }
    else:
        print("Running Naive GNQ (sequential P^2)...")
        try:
            mem_start = start_mem_measure()
            naive_gnq, G_naive, t_naive = naive_gnq_full_pipeline(
                model_naive, x, y, lambda_reg=lambda_reg
            )
            mem_mb = end_mem_measure(mem_start)
            results["naive"] = {
                "status": "ok",
                "gnq": naive_gnq,
                "timings": t_naive,
                "mem_mb": mem_mb,
            }
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            results["naive"] = {"status": "OOM", "gnq": None, "timings": None, "mem_mb": None}
            print("  [Naive GNQ] OOM on this regime.")

    # ---------- Diagonal GNQ ----------
    print("Running Diag GNQ (diagonal S approximation)...")
    try:
        mem_start = start_mem_measure()
        diag_gnq, G_diag, t_diag = diag_gnq_full_pipeline(
            model_diag, x, y, epsilon=diag_epsilon
        )
        mem_mb = end_mem_measure(mem_start)
        results["diag"] = {
            "status": "ok",
            "gnq": diag_gnq,
            "timings": t_diag,
            "mem_mb": mem_mb,
        }
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        results["diag"] = {"status": "OOM", "gnq": None, "timings": None, "mem_mb": None}
        print("  [Diag GNQ] OOM on this regime.")

    # ---------- Summary table ----------
    print("\nSummary for regime:", name)
    print("=" * 110)
    print(f"{'Algo':<10} | {'Status':<30} | {'Extract (s)':<12} | "
          f"{'GNQ (s)':<12} | {'Total (s)':<12} | {'Peak Mem (MB)':<14}")
    print("-" * 110)

    def get_times(key):
        if key not in results or results[key]["timings"] is None:
            return ("-", "-", "-")
        t = results[key]["timings"]
        if key in ["naive", "diag"]:
            return (f"{t['extract']:.6f}", f"{t['calc']:.6f}", f"{t['total']:.6f}")
        else:
            gnq_math = t["kernel"] + t["solve"]
            return (f"{t['extract']:.6f}", f"{gnq_math:.6f}", f"{t['total']:.6f}")

    def get_mem(key):
        mem = results.get(key, {}).get("mem_mb", None)
        if mem is None:
            return "-"
        return f"{mem:>.1f}"

    for key in ["naive", "fast", "ghost", "diag"]:
        status = results.get(key, {}).get("status", "not run")
        ex, gnq_t, tot = get_times(key)
        mem_str = get_mem(key)
        print(f"{key:<10} | {status:<30} | {ex:<12} | {gnq_t:<12} | {tot:<12} | {mem_str:<14}")
    print("=" * 110)

    # ---------- Agreement where possible (vs Naive) ----------
    if results.get("naive", {}).get("status") == "ok":
        naive_gnq = results["naive"]["gnq"]
        for key in ["fast", "ghost", "diag"]:
            if results.get(key, {}).get("status") == "ok":
                gnq = results[key]["gnq"]
                max_diff = (gnq - naive_gnq).abs().max().item()
                ok = torch.allclose(gnq, naive_gnq, atol=1e-5)
                print(f"{key.capitalize()} vs Naive: "
                      f"{'✅' if ok else '❌'} (max |Δ| = {max_diff:.3e})")
    else:
        print("Naive not available; cannot check numeric agreement in this regime.")


# ============================================================
# 6. GPT-2 helpers: sample WikiText-2 batches
# ============================================================
def sample_wikitext_batch(ds, tokenizer, batch_size, max_length):
    texts = []
    n = len(ds)
    while len(texts) < batch_size:
        idx = torch.randint(0, n, (1,)).item()
        line = ds[idx]["text"].strip()
        if len(line) == 0:
            continue
        texts.append(line)

    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    labels = input_ids.clone()
    pad_id = tokenizer.pad_token_id
    labels[labels == pad_id] = -100

    return input_ids, attention_mask, labels


def warmup_gpt2(model, ds_train, tokenizer,
                batch_size=4, max_length=64,
                warmup_steps=10, lr=1e-4):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in range(warmup_steps):
        input_ids, attention_mask, labels = sample_wikitext_batch(
            ds_train, tokenizer, batch_size, max_length
        )
        optimizer.zero_grad(set_to_none=True)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss * input_ids.size(0)
        loss.backward()
        optimizer.step()

    input_ids, attention_mask, labels = sample_wikitext_batch(
        ds_train, tokenizer, batch_size, max_length
    )
    return input_ids, attention_mask, labels


# ============================================================
# 6.1. GPT-2 helpers: token frequencies (rarity) + perplexity
# ============================================================
def build_token_log_probs(ds_train, tokenizer,
                          max_docs=5000,
                          max_length=128):
    vocab_size = tokenizer.vocab_size
    counts = torch.zeros(vocab_size, dtype=torch.long)

    n = len(ds_train)
    use_docs = min(max_docs, n)

    for i in range(use_docs):
        text = ds_train[i]["text"]
        if not text or len(text.strip()) == 0:
            continue

        ids = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )["input_ids"]

        if len(ids) == 0:
            continue

        ids_tensor = torch.tensor(ids, dtype=torch.long)
        counts.index_add_(0, ids_tensor, torch.ones_like(ids_tensor, dtype=torch.long))

    counts = counts.float()
    counts[counts == 0] = 0.5
    probs = counts / counts.sum()
    log_probs = torch.log(probs)
    return log_probs 


@torch.no_grad()
def compute_seq_perplexities(model, input_ids, attention_mask, labels):
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits 


    shift_logits = logits[:, :-1, :]          
    shift_labels = labels[:, 1:]              
    shift_mask = (shift_labels != -100)      


    log_probs = torch.log_softmax(shift_logits, dim=-1) 


    valid_labels = shift_labels.clone()
    valid_labels[~shift_mask] = 0  

    target_log_probs = log_probs.gather(
        -1, valid_labels.unsqueeze(-1)
    ).squeeze(-1)  

    target_log_probs = target_log_probs * shift_mask  

    token_counts = shift_mask.sum(dim=1)  
    token_counts = token_counts.clamp_min(1)

    seq_nll = -target_log_probs.sum(dim=1) / token_counts 
    seq_ppl = torch.exp(seq_nll)

    return seq_nll.cpu(), seq_ppl.cpu()


# ============================================================
# 7. GPT-2 regime -> GNQ reuses the training backward (no extra backward)
# ============================================================
def run_gpt2_regime(batch_size=4, max_length=64,
                    lambda_reg=1e-2,
                    warmup_steps_gpt2=10):
    print("\n" + "#" * 80)
    print("REGIME: gpt2-small-lm (GhostFastGNQ only, training-like)")
    print("#" * 80)

    try:
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
    except ImportError:
        print("Transformers is not installed. Skipping GPT-2 regime.")
        return

    if not HAVE_DATASETS:
        print("datasets library not available, cannot use WikiText-2. Skipping GPT-2 regime.")
        return

    torch.manual_seed(42)


    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.train()  

    total_params = sum(p.numel() for p in model.parameters())


    ds_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    # ---- Build unigram token distribution for rarity ----
    print("Estimating token frequencies on WikiText-2 train...")
    token_log_probs = build_token_log_probs(
        ds_train, tokenizer,
        max_docs=5000,
        max_length=max_length,
    ) 

    input_ids, attention_mask, labels = warmup_gpt2(
        model, ds_train, tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        warmup_steps=warmup_steps_gpt2,
        lr=1e-4,
    )

    B, T = input_ids.shape

    print(f"Batch Size: {B}, Seq Len: {T}")
    print("Model: GPT2LMHeadModel('gpt2')")
    print(f"Total Parameters (N_p): {total_params}")
    print(f"Naive GNQ allowed up to N_p <= {NAIVE_P_MAX}")
    print(f"Warmup steps (GPT-2): {warmup_steps_gpt2}")
    print("-" * 90)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # =====================================================
    # Baseline training iteration (no GNQ at all)
    # =====================================================
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss * input_ids.size(0)
    loss.backward()
    optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    baseline_iter_time = t1 - t0
    baseline_throughput = B / baseline_iter_time

    print("\n[GPT-2] Baseline training iteration (no GNQ):")
    print(f"  time_per_iter = {baseline_iter_time:.6f} s")
    print(f"  throughput    = {baseline_throughput:.2f} sequences/s")
    print("-" * 90)

    # =====================================================
    # Training iteration WITH GNQ, reusing the same backward
    # =====================================================
    ghost_calc = GhostFastGNQ(model, lambda_reg=lambda_reg)

    results = {}

    print("[GPT-2] Training iteration with GhostFastGNQ (no extra backward)...")
    mem_start = start_mem_measure()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    ghost_calc.reset_cache()

    optimizer.zero_grad(set_to_none=True)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss * input_ids.size(0)
    loss.backward() 

    ghost_gnq, gnq_timings = ghost_calc.compute_gnq_from_cache(input_ids)

    optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    iter_with_gnq = t1 - t0

    mem_mb = end_mem_measure(mem_start)

    results["ghost"] = {
        "status": "ok",
        "gnq": ghost_gnq.detach().cpu(),
        "timings": gnq_timings,
        "mem_mb": mem_mb,
    }

    # Fast & Naive are deliberately skipped here for GPT-2 (impractical G[B,P])
    results["fast"] = {
        "status": "skipped (G[B,P] too large / impractical for GPT-2)",
        "gnq": None,
        "timings": None,
        "mem_mb": None,
    }
    results["naive"] = {
        "status": f"skipped (N_p={total_params} >> {NAIVE_P_MAX})",
        "gnq": None,
        "timings": None,
        "mem_mb": None,
    }
    results["diag"] = {
        "status": "skipped (diag approximation not implemented for GPT-2)",
        "gnq": None,
        "timings": None,
        "mem_mb": None,
    }

    # =====================================================
    # Summary for regime (GNQ timings = kernel+solve only)
    # =====================================================
    print("\nSummary for regime: gpt2-small-lm")
    print("=" * 110)
    print(f"{'Algo':<10} | {'Status':<60} | {'Extract (s)':<12} | "
          f"{'GNQ (s)':<12} | {'Total (s)':<12} | {'Peak Mem (MB)':<14}")
    print("-" * 110)

    def get_times_gpt(key):
        if key not in results or results[key]["timings"] is None:
            return ("-", "-", "-")
        t = results[key]["timings"]
        # ghost only for this regime; treat like fast/ghost
        gnq_math = t["kernel"] + t["solve"]
        return (f"{t['extract']:.6f}", f"{gnq_math:.6f}", f"{t['total']:.6f}")

    def get_mem_gpt(key):
        mem = results.get(key, {}).get("mem_mb", None)
        if mem is None:
            return "-"
        return f"{mem:>.1f}"

    for key in ["naive", "fast", "ghost", "diag"]:
        status = results.get(key, {}).get("status", "not run")
        ex, gnq_t, tot = get_times_gpt(key)
        mem_str = get_mem_gpt(key)
        print(f"{key:<10} | {status:<60} | {ex:<12} | {gnq_t:<12} | {tot:<12} | {mem_str:<14}")
    print("=" * 110)

    # =====================================================
    # Iteration timing with GNQ overhead (no extra backward)
    # =====================================================
    print("\n[GPT-2] Training iteration timing with GNQ overhead (no extra backward)")
    print("=" * 110)
    print(f"{'Algo':<10} | {'Status':<30} | {'Iter w/o GNQ (s)':<18} | "
          f"{'Iter w/ GNQ (s)':<18} | {'Overhead x':<12} | "
          f"{'TP w/o GNQ':<14} | {'TP w/ GNQ':<14}")
    print("-" * 110)

    ghost_status = results["ghost"]["status"]
    if ghost_status == "ok":
        tp_no_gnq = baseline_throughput
        tp_with_gnq = B / iter_with_gnq
        overhead_factor = iter_with_gnq / baseline_iter_time

        print(f"{'ghost':<10} | {ghost_status:<30} | "
              f"{baseline_iter_time:>18.6f} | "
              f"{iter_with_gnq:>18.6f} | "
              f"{overhead_factor:>12.3f} | "
              f"{tp_no_gnq:>14.2f} | "
              f"{tp_with_gnq:>14.2f}")
    else:
        print(f"{'ghost':<10} | {ghost_status:<30} | "
              f"{baseline_iter_time:>18.6f} | "
              f"{'-':>18} | "
              f"{'-':>12} | "
              f"{baseline_throughput:>14.2f} | "
              f"{'-':>14}")
    print("=" * 110)

    if results["ghost"]["status"] != "ok":
        print("Ghost GNQ not available for GPT-2 (OOM or error).")
        return

    # --------------------------------------------------------
    # Per-sequence GNQ, rarity, perplexity
    # --------------------------------------------------------
    gnq_vals = results["ghost"]["gnq"]  # [B] on CPU
    print(f"\nGhost GNQ computed for {len(gnq_vals)} sequences (no extra backward).")

    # 1) Per-sequence perplexity
    seq_ce, seq_ppl = compute_seq_perplexities(
        model, input_ids, attention_mask, labels
    )  

    # 2) Per-sequence token rarity: avg -log p(tok) over non-pad tokens
    with torch.no_grad():
        token_log_probs_local = token_log_probs  
        seq_rarity = []
        for i in range(B):
            ids = input_ids[i].detach().cpu()     
            mask = attention_mask[i].detach().cpu().bool() 

            valid_ids = ids[mask]
            if valid_ids.numel() == 0:
                seq_rarity.append(0.0)
                continue

            # Clamp to vocab range just in case
            valid_ids = valid_ids.clamp(min=0, max=token_log_probs_local.shape[0] - 1)
            log_p = token_log_probs_local[valid_ids]   
            rarity_i = (-log_p).mean().item()          
            seq_rarity.append(rarity_i)

        seq_rarity = torch.tensor(seq_rarity)

    # --------------------------------------------------------
    # Print per-sequence table (GNQ, rarity, perplexity, snippet)
    # --------------------------------------------------------
    print("\nPer-sequence stats (GPT-2, WikiText-2):")
    for i in range(B):
        txt = tokenizer.decode(
            input_ids[i].detach().cpu(),
            skip_special_tokens=True
        )
        snippet = txt[:200].replace("\n", " ")
        print(
            f"[{i}] GNQ={gnq_vals[i].item():.6f} | "
            f"rarity(avg -log p)={seq_rarity[i].item():.4f} | "
            f"ppl={seq_ppl[i].item():.4f} | text={snippet}..."
        )

    # --------------------------------------------------------
    # Correlations + scatter plots
    # --------------------------------------------------------
    try:
        import numpy as np
        import matplotlib.pyplot as plt

        gnq_np = gnq_vals.numpy()
        rarity_np = seq_rarity.numpy()
        ppl_np = seq_ppl.numpy()

        # Pearson correlations
        def pearson_corr(x, y):
            x = x - x.mean()
            y = y - y.mean()
            denom = (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)
            return float((x * y).sum() / denom)

        r_gnq_rarity = pearson_corr(rarity_np, gnq_np)
        r_gnq_ppl = pearson_corr(ppl_np, gnq_np)

        print("\nCorrelations:")
        print(f"  corr(GNQ, rarity)     = {r_gnq_rarity:.4f}")
        print(f"  corr(GNQ, perplexity) = {r_gnq_ppl:.4f}")

        # Scatter: GNQ vs rarity
        plt.figure()
        plt.scatter(rarity_np, gnq_np, alpha=0.7)
        plt.xlabel("Average token rarity (mean -log p(token))")
        plt.ylabel("GNQ")
        plt.title("GNQ vs token rarity (GPT-2, WikiText-2)")
        plt.tight_layout()
        plt.savefig("gpt2_gnq_vs_rarity.png")

        # Scatter: GNQ vs perplexity
        plt.figure()
        plt.scatter(ppl_np, gnq_np, alpha=0.7)
        plt.xlabel("Per-sequence perplexity")
        plt.ylabel("GNQ")
        plt.xscale("log")
        plt.title("GNQ vs perplexity (GPT-2, WikiText-2)")
        plt.tight_layout()
        plt.savefig("gpt2_gnq_vs_perplexity.png")

        print("\nSaved plots:")
        print("  gpt2_gnq_vs_rarity.png")
        print("  gpt2_gnq_vs_perplexity.png")

    except ImportError:
        print("\nmatplotlib or numpy not installed; skipping plots.")


# ============================================================
# 8. Training-overhead table for the largest MLP regime only
# ============================================================
def run_large_mlp_training_overhead(
    name="xxl-mlp-784-4096-4096-10",
    input_dim=784,
    hidden_dims=[4096, 4096],
    output_dim=10,
    batch_size=32,
    resize_to=(28, 28),
    lambda_reg=1e-2,
    warmup_steps_mlp=10,
    diag_epsilon=1e-2,
):
    print("\n" + "#" * 80)
    print(f"TRAINING OVERHEAD REGIME (LARGEST MLP): {name}")
    print("#" * 80)

    torch.manual_seed(42)

    loader = make_mnist_loader(batch_size=batch_size, resize_to=resize_to)
    base_model = build_mlp(input_dim, hidden_dims, output_dim)

    x, y = warmup_mlp_and_get_batch(
        base_model, loader, warmup_steps=warmup_steps_mlp, lr=0.1
    )
    B = x.shape[0]

    total_params = sum(p.numel() for p in base_model.parameters())
    print(f"Batch Size: {batch_size}")
    print(f"Model: MLP({input_dim} -> {hidden_dims} -> {output_dim})")
    print(f"Total Parameters (N_p): {total_params}")
    print(f"Warmup steps (MLP): {warmup_steps_mlp}")
    print("-" * 90)

    # Clone the warmed-up model so each algo starts from identical weights
    model_baseline = copy.deepcopy(base_model).train()
    model_naive    = copy.deepcopy(base_model).train()
    model_fast     = copy.deepcopy(base_model).train()
    model_ghost    = copy.deepcopy(base_model).train()
    model_diag     = copy.deepcopy(base_model).train()

    criterion = nn.CrossEntropyLoss(reduction="sum")

    def train_step(model):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        return t1 - t0

    # --------------------------------------------------------
    # Baseline iter (no GNQ)
    # --------------------------------------------------------
    baseline_iter_time = train_step(model_baseline)
    baseline_throughput = B / baseline_iter_time

    print("\n[XXL-MLP] Baseline training iteration (no GNQ):")
    print(f"  time_per_iter = {baseline_iter_time:.6f} s")
    print(f"  throughput    = {baseline_throughput:.2f} samples/s")
    print("-" * 90)

    # --------------------------------------------------------
    # Overhead for each GNQ algorithm
    #   - All use the same (x, y)
    #   - Ghost reuses its training backward (no extra backward)
    #   - Naive / Fast / Diag do extra work after training step
    # --------------------------------------------------------
    overhead_results = {}

    # ---------- Naive ----------
    if total_params > NAIVE_P_MAX:
        overhead_results["naive"] = {
            "status": f"skipped (N_p={total_params} > {NAIVE_P_MAX})",
            "iter_time": None,
            "throughput": None,
            "overhead": None,
        }
    else:
        print("[XXL-MLP] Training iteration + Naive GNQ...")
        optimizer_naive = torch.optim.SGD(model_naive.parameters(), lr=0.1)
        try:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            # 1) normal training step
            optimizer_naive.zero_grad(set_to_none=True)
            logits = model_naive(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer_naive.step()

            # 2) Naive GNQ on the same model and batch (extra work)
            naive_gnq, _, _ = naive_gnq_full_pipeline(
                model_naive, x, y, lambda_reg=lambda_reg
            )

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            iter_time = t1 - t0
            throughput = B / iter_time
            overhead = iter_time / baseline_iter_time

            overhead_results["naive"] = {
                "status": "ok",
                "iter_time": iter_time,
                "throughput": throughput,
                "overhead": overhead,
            }
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            overhead_results["naive"] = {
                "status": "OOM",
                "iter_time": None,
                "throughput": None,
                "overhead": None,
            }

    # ---------- Fast ----------
    print("[XXL-MLP] Training iteration + FastGNQ...")
    fast_calc = FastGNQ(model_fast, lambda_reg=lambda_reg)
    params_fast = dict(model_fast.named_parameters())
    buffers_fast = dict(model_fast.named_buffers())

    optimizer_fast = torch.optim.SGD(model_fast.parameters(), lr=0.1)
    try:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # 1) normal training step
        optimizer_fast.zero_grad(set_to_none=True)
        logits = model_fast(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer_fast.step()

        # 2) Fast GNQ (extra work)
        fast_gnq, _, _ = fast_calc.compute_gnq(params_fast, buffers_fast, x, y)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        iter_time = t1 - t0
        throughput = B / iter_time
        overhead = iter_time / baseline_iter_time

        overhead_results["fast"] = {
            "status": "ok",
            "iter_time": iter_time,
            "throughput": throughput,
            "overhead": overhead,
        }
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        overhead_results["fast"] = {
            "status": "OOM",
            "iter_time": None,
            "throughput": None,
            "overhead": None,
        }

    # ---------- Ghost (NO extra backward) ----------
    print("[XXL-MLP] Training iteration + GhostFastGNQ (no extra backward)...")
    ghost_calc = GhostFastGNQ(model_ghost, lambda_reg=lambda_reg)
    optimizer_ghost = torch.optim.SGD(model_ghost.parameters(), lr=0.1)

    try:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        ghost_calc.reset_cache()
        optimizer_ghost.zero_grad(set_to_none=True)

        # 1) normal training forward+backward (hooks populate cache)
        logits = model_ghost(x)
        loss = criterion(logits, y)
        loss.backward()

        # 2) GNQ from cache (kernel + solve only)
        ghost_gnq, _ = ghost_calc.compute_gnq_from_cache(x)

        optimizer_ghost.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        iter_time = t1 - t0
        throughput = B / iter_time
        overhead = iter_time / baseline_iter_time

        overhead_results["ghost"] = {
            "status": "ok",
            "iter_time": iter_time,
            "throughput": throughput,
            "overhead": overhead,
        }
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        overhead_results["ghost"] = {
            "status": "OOM",
            "iter_time": None,
            "throughput": None,
            "overhead": None,
        }

    # ---------- Diagonal approximation ----------
    print("[XXL-MLP] Training iteration + Diag GNQ (D_ij)...")
    optimizer_diag = torch.optim.SGD(model_diag.parameters(), lr=0.1)
    try:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # 1) normal training step
        optimizer_diag.zero_grad(set_to_none=True)
        logits = model_diag(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer_diag.step()

        # 2) Diagonal GNQ (extra work)
        diag_gnq, _, _ = diag_gnq_full_pipeline(
            model_diag, x, y, epsilon=diag_epsilon
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        iter_time = t1 - t0
        throughput = B / iter_time
        overhead = iter_time / baseline_iter_time

        overhead_results["diag"] = {
            "status": "ok",
            "iter_time": iter_time,
            "throughput": throughput,
            "overhead": overhead,
        }
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        overhead_results["diag"] = {
            "status": "OOM",
            "iter_time": None,
            "throughput": None,
            "overhead": None,
        }

    print("\n[XXL-MLP] Training iteration timing with GNQ overhead")
    print("=" * 110)
    print(f"{'Algo':<10} | {'Status':<30} | {'Iter w/o GNQ (s)':<18} | "
          f"{'Iter w/ GNQ (s)':<18} | {'Overhead x':<12} | "
          f"{'TP w/o GNQ':<14} | {'TP w/ GNQ':<14}")
    print("-" * 110)

    for key in ["naive", "fast", "ghost", "diag"]:
        res = overhead_results.get(key, None)
        if res is None:
            status = "not run"
            iter_w = "-"
            overhead = "-"
            tp_w = "-"
        else:
            status = res["status"]
            if status == "ok":
                iter_w = f"{res['iter_time']:.6f}"
                overhead = f"{res['overhead']:.3f}"
                tp_w = f"{res['throughput']:.2f}"
            else:
                iter_w = "-"
                overhead = "-"
                tp_w = "-"

        print(f"{key:<10} | {status:<30} | "
              f"{baseline_iter_time:>18.6f} | "
              f"{iter_w:>18} | "
              f"{overhead:>12} | "
              f"{baseline_throughput:>14.2f} | "
              f"{tp_w:>14}")
    print("=" * 110)

# ============================================================
# 9. Main: MLP regimes + GPT-2 regime
# ============================================================
def main():
    # MLP regimes: GNQ-only timings (Naive / Fast / Ghost / Diag)
    regimes = [
        dict(
            name="small-mlp-100-40-10",
            input_dim=100,           
            hidden_dims=[40],
            output_dim=10,
            batch_size=32,
            resize_to=(10, 10),
        ),
        dict(
            name="medium-mlp-784-256-10",
            input_dim=784,           
            hidden_dims=[256],
            output_dim=10,
            batch_size=32,
            resize_to=(28, 28),
        ),
        dict(
            name="large-mlp-784-1024-1024-10",
            input_dim=784,
            hidden_dims=[1024, 1024],
            output_dim=10,
            batch_size=32,
            resize_to=(28, 28),
        ),
        dict(
            name="xl-mlp-784-2048-2048-10",
            input_dim=784,
            hidden_dims=[2048, 2048],
            output_dim=10,
            batch_size=32,
            resize_to=(28, 28),
        ),
        dict(
            name="xxl-mlp-784-4096-4096-10",
            input_dim=784,
            hidden_dims=[4096, 4096],
            output_dim=10,
            batch_size=32,
            resize_to=(28, 28),
        ),
    ]

    for cfg in regimes:
        run_regime(**cfg, warmup_steps_mlp=10, lambda_reg=1e-2, diag_epsilon=1e-2)

    # Training-overhead table ONLY for the largest MLP
    run_large_mlp_training_overhead(
        name="xxl-mlp-784-4096-4096-10",
        input_dim=784,
        hidden_dims=[4096, 4096],
        output_dim=10,
        batch_size=32,
        resize_to=(28, 28),
        lambda_reg=1e-2,
        warmup_steps_mlp=10,
        diag_epsilon=1e-2,
    )

    # GPT-2 ghost-only experiment on a realistic LM batch
    run_gpt2_regime(
        batch_size=16,
        max_length=128,
        lambda_reg=1e-2,
        warmup_steps_gpt2=100,
    )

if __name__ == "__main__":
    main()
