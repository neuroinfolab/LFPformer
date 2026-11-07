"""Step-based training loop for the GCP full-recording run (CCFDistc / ManipDistc).

Used by train_ssl.py when config["step_mode"] is true. The per-step math (forward,
loss, clip, optimizer, Gumbel decay) is the same as ssl_train_epoch_detailed;
what changes is everything around it, so a 257M-sample epoch on spot VMs can be
measured and resumed (see gcp_neuroframe/GCP_PRETRAINING.md §2e):

  * W&B every `log_every_steps`: all train loss components all-reduced over
    ranks, grad norm (pre-clip), lr, throughput, data-wait fraction, GPU memory,
    cost so far and projected. One W&B run id across restarts.
  * validation every `val_every_steps` on a FIXED val subset with a fixed RNG
    seed (comparable over time), all-reduced, plus per-region loss.
  * linear probe every `probe_every_steps` on the val subset (rank 0).
  * checkpoint every `ckpt_every_steps` with exact mid-epoch resume state;
    best-val and end-of-epoch snapshots.
  * SIGTERM (spot preemption) -> checkpoint at the next step, then exit 0.
  * budget guard: stop cleanly once the estimated cost reaches BUDGET_USD_HARD.
  * manual graceful stop: create <results_dir>/STOP.
"""
import json
import math
import os
import signal
import time
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


# --------------------------------------------------------------------------- #
def _worker_ignore_term(_worker_id):
    """DataLoader workers must not die on SIGTERM (torchrun forwards it to the whole
    group on preemption); the main process saves a checkpoint and then shuts them down."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)


class ResumableShardSampler(Sampler):
    """DistributedSampler-equivalent random sampler that can start mid-epoch.

    Permutation = torch.randperm(N, seed + epoch), padded to a multiple of
    world_size (like DistributedSampler, drop_last=False), rank r takes
    perm[r::world]. `start` skips the first `start` indices of this rank's
    list, so resuming after k steps continues with exactly the unseen samples.
    Memory: one int64 permutation (8 B/sample), not a Python list.
    """

    def __init__(self, n: int, world: int, rank: int, seed: int = 0):
        self.n, self.world, self.rank, self.seed = int(n), int(world), int(rank), int(seed)
        self.per_rank = math.ceil(self.n / self.world)
        self.epoch, self.start = 0, 0

    def set_epoch(self, epoch: int, start: int = 0):
        self.epoch, self.start = int(epoch), int(start)

    def __len__(self):
        return max(self.per_rank - self.start, 0)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        perm = torch.randperm(self.n, generator=g)
        total = self.per_rank * self.world
        if total > self.n:
            perm = torch.cat([perm, perm[: total - self.n]])
        mine = perm[self.rank::self.world][self.start:]
        for chunk in torch.split(mine, 1 << 20):
            yield from chunk.tolist()


def _lr_factor(step, warmup_steps, start_factor, total_steps, base_lr, eta_min):
    """Same shape as the original SequentialLR(LinearLR warmup, CosineAnnealingLR)."""
    if warmup_steps > 0 and step < warmup_steps:
        return start_factor + (1.0 - start_factor) * step / warmup_steps
    decay = max(total_steps - warmup_steps, 1)
    t = min(max(step - warmup_steps, 0), decay)
    lr = eta_min + (base_lr - eta_min) * 0.5 * (1.0 + math.cos(math.pi * t / decay))
    return lr / base_lr


# --------------------------------------------------------------------------- #
def run_step_mode(ctx: Dict[str, Any]) -> None:
    c = ctx["config"]
    log = ctx["log"]
    backbone = ctx["backbone"]
    inner = ctx["backbone_module"]
    optimizer = ctx["optimizer"]
    device = ctx["device"]
    use_amp = ctx["use_amp"]
    scaler = ctx["scaler"]
    is_main, is_ddp = ctx["is_main"], ctx["is_ddp"]
    world, rank = ctx["world_size"], ctx["global_rank"]
    grad_clip_norm = float(c.get("grad_clip_norm", 1.0))
    preprocess_batch = ctx["preprocess_batch"]
    quick_probe = ctx["quick_linear_probe"]
    backbone_name = c["backbone"]

    log_every = int(c.get("log_every_steps", 50))
    val_every = int(c.get("val_every_steps", 5000))
    probe_every = int(c.get("probe_every_steps", 25000))
    ckpt_every = int(c.get("ckpt_every_steps", 2000))
    probe_epochs = int(c.get("probe_epochs", 20))
    ssl_epochs = int(c.get("ssl_epochs", 4))
    results_dir = c.get("results_dir") or os.environ.get("RESULTS_DIR", "./results")
    ckpt_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    price_h = float(os.environ.get("GPU_PRICE_USD_PER_HOUR", c.get("gpu_price_usd_per_hour", 0.0)))
    budget_hard = float(os.environ.get("BUDGET_USD_HARD", c.get("budget_usd_hard", 0.0)))
    sampler_seed = int(c.get("sampler_seed", 1234))
    val_seed = int(c.get("val_seed", 4321))

    # ---- data ----------------------------------------------------------------
    train_dataset = ctx["train_dataset"]
    per_gpu_batch = ctx["per_gpu_batch"]
    grad_accum = max(int(os.environ.get("GRAD_ACCUM_STEPS", "1") or 1), 1)
    num_workers = int(c.get("num_workers", 8))
    prefetch = int(c.get("prefetch_factor", 8))
    sampler = ResumableShardSampler(len(train_dataset), world, rank, seed=sampler_seed)
    # global_step is an optimizer step.  With accumulation, each step consumes
    # grad_accum physical batches on every rank.
    steps_per_epoch = math.ceil(sampler.per_rank / (per_gpu_batch * grad_accum))
    total_steps = ssl_epochs * steps_per_epoch
    physical_global_batch = per_gpu_batch * world
    global_batch = physical_global_batch * grad_accum

    from blind_localization.data.lazyloader_dataset import CompactDataset
    vs_kwargs = dict(ctx["compact_kwargs_labeled"])
    val_subset = CompactDataset(compact_dir=c["val_subset_dir"], **vs_kwargs)
    vs_sampler = (torch.utils.data.distributed.DistributedSampler(
        val_subset, num_replicas=world, rank=rank, shuffle=False, drop_last=False) if is_ddp else None)
    val_loader = DataLoader(val_subset, batch_size=per_gpu_batch, sampler=vs_sampler, shuffle=False,
                            num_workers=num_workers, pin_memory=True, worker_init_fn=_worker_ignore_term)
    probe_loader = DataLoader(val_subset, batch_size=per_gpu_batch, shuffle=False,
                              num_workers=num_workers, pin_memory=True, worker_init_fn=_worker_ignore_term)
    n_classes = getattr(val_subset, "num_effective_classes", None)
    id_to_name = {}
    if getattr(val_subset, "effective_label_map", None):
        id_to_name = {v: k for k, v in val_subset.effective_label_map.items()}

    # ---- LR schedule (closed form: no fast-forward loop on resume) ------------
    base_lr = float(c.get("lr_ssl", 1e-4))
    warmup = int(c.get("warmup_steps", 0))
    start_factor = float(c.get("warmup_start_factor", 0.001))
    eta_min = float(c.get("eta_min", 0.0))
    for g in optimizer.param_groups:
        g.setdefault("initial_lr", base_lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: _lr_factor(s, warmup, start_factor, total_steps, base_lr, eta_min))

    # ---- state / resume ------------------------------------------------------
    st = dict(global_step=0, epoch=0, step_in_epoch=0, best_val_loss=float("inf"),
              best_probe_acc=0.0, wall_s_prev=0.0, n_nonfinite=0, restarts=0,
              lr_scale=1.0, vals_since_best=0, cost_alerts_sent=[])
    resume = c.get("resume_from")
    if resume and os.path.isfile(resume):
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        if "step_state" not in ck:
            raise RuntimeError(f"{resume} is not a step-mode checkpoint")
        inner.load_state_dict(ck["backbone_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        if use_amp and "scaler_state_dict" in ck:
            scaler.load_state_dict(ck["scaler_state_dict"])
        st.update(ck["step_state"])
        st["restarts"] = int(st.get("restarts", 0)) + 1
        if hasattr(inner, "model") and ck.get("gumbel_temperature") is not None:
            inner.model.quantizer.temperature = float(ck["gumbel_temperature"])
        assert ck["step_state"].get("steps_per_epoch", steps_per_epoch) == steps_per_epoch, \
            "world size / batch changed: steps_per_epoch differs from the checkpoint"
        log(f"[step] resumed {resume}: global_step={st['global_step']} epoch={st['epoch']} "
            f"step_in_epoch={st['step_in_epoch']} restarts={st['restarts']}")
        del ck
    # ---- LR-adjust restart option: LR_SCALE multiplies the whole schedule from now on,
    # ramped linearly from the previous scale over LR_SCALE_RAMP_STEPS (default 500).
    scale_from = float(st.get("lr_scale", 1.0))
    scale_to = float(os.environ.get("LR_SCALE", "") or scale_from)
    ramp = max(int(os.environ.get("LR_SCALE_RAMP_STEPS", "500") or 500), 1)
    gs0 = st["global_step"]

    def lr_scale_at(step):
        if scale_to == scale_from:
            return scale_to
        return scale_from + (scale_to - scale_from) * min(1.0, max(step - gs0, 0) / ramp)

    st["lr_scale"] = scale_to
    if scale_to != scale_from:
        log(f"[step] LR scale change {scale_from} -> {scale_to} (ramp {ramp} steps from gs={gs0})")
    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s_: lr_scale_at(s_) * _lr_factor(s_, warmup, start_factor, total_steps, base_lr, eta_min))
    sched.last_epoch = st["global_step"]
    for g in optimizer.param_groups:
        g["lr"] = base_lr * lr_scale_at(st["global_step"]) * _lr_factor(st["global_step"], warmup, start_factor, total_steps, base_lr, eta_min)

    # ---- W&B -----------------------------------------------------------------
    use_wb = is_main and wandb is not None and wandb.run is not None
    if use_wb:
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")
        wandb.config.update(dict(
            step_mode=True, world_size=world, per_gpu_batch=per_gpu_batch,
            grad_accum_steps=grad_accum, physical_global_batch=physical_global_batch,
            global_batch=global_batch,
            n_train=len(train_dataset), n_val_subset=len(val_subset), steps_per_epoch=steps_per_epoch,
            total_steps=total_steps, gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            gpu_price_usd_per_hour=price_h, budget_usd_hard=budget_hard,
            code_hash=os.environ.get("CODE_HASH", ""), dataset_tag=os.environ.get("DATASET_TAG", ""),
        ), allow_val_change=True)
        wandb.log({"global_step": st["global_step"], "run/restarts": st["restarts"]})

    run_name = os.environ.get("WANDB_RUN_NAME") or os.environ.get("WANDB_RUN_ID") or "run"
    val_patience = int(c.get("alert_val_patience", 6))
    gn_hist = []

    def alert(title, text, level="WARN", wait_s=3600):
        if not use_wb:
            return
        try:
            lv = getattr(wandb.AlertLevel, level, wandb.AlertLevel.WARN)
            wandb.alert(title=f"[{run_name}] {title}", text=text, level=lv, wait_duration=wait_s)
        except Exception as e:  # noqa: BLE001
            log(f"[WARN] wandb.alert failed: {e}")
        log(f"[ALERT {level}] {title}: {text}")

    alert("started" if st["restarts"] == 0 else f"resumed (restart #{st['restarts']})",
          f"gs={st['global_step']:,} epoch={st['epoch']} lr_scale={st['lr_scale']}", "INFO", 0)

    log(f"[step] world={world} per_gpu_batch={per_gpu_batch} grad_accum={grad_accum} "
        f"global_batch={global_batch} "
        f"n_train={len(train_dataset):,} steps/epoch={steps_per_epoch:,} total_steps={total_steps:,} "
        f"val_subset={len(val_subset):,} log/val/probe/ckpt every {log_every}/{val_every}/{probe_every}/{ckpt_every}")

    # ---- preemption / stop signals -------------------------------------------
    stop_flag = {"sig": False}

    def _on_term(signum, frame):  # noqa: ARG001
        stop_flag["sig"] = True
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGUSR1, _on_term)

    def _collective_flag(local: bool) -> bool:
        if not is_ddp:
            return local
        t = torch.tensor([1.0 if local else 0.0], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t.item() > 0)

    t_job0 = time.time()

    def wall_total():
        return st["wall_s_prev"] + (time.time() - t_job0)

    def cost_now():
        return price_h * wall_total() / 3600.0

    # ---- checkpoint ----------------------------------------------------------
    def save_ckpt(path, tag):
        if not is_main:
            return
        s = dict(st)
        s["wall_s_prev"] = wall_total()
        s["steps_per_epoch"] = steps_per_epoch
        payload = {
            "backbone_state_dict": inner.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "config": c,
            "epoch": st["epoch"],
            "step_state": s,
            "gumbel_temperature": float(inner.model.quantizer.temperature) if hasattr(inner, "model") else None,
            "wandb_run_id": wandb.run.id if use_wb else os.environ.get("WANDB_RUN_ID"),
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tag": tag,
        }
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)
        log(f"[ckpt] {tag} -> {path} (gs={st['global_step']})")

    latest = os.path.join(ckpt_dir, f"ssl_{backbone_name}_step_latest.pt")

    # kept milestone snapshots (model only, never overwritten) every N samples
    snap_every_samples = int(float(c.get("snapshot_every_samples", 25_000_000)))
    snap_every_steps = max(snap_every_samples // global_batch, 1) if snap_every_samples > 0 else 0
    snap_dir = os.path.join(ckpt_dir, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)

    def save_snapshot():
        if not is_main:
            return
        gs_ = st["global_step"]
        path = os.path.join(snap_dir, f"ssl_{backbone_name}_gs{gs_:08d}.pt")
        tmp = path + ".tmp"
        torch.save({
            "backbone_state_dict": inner.state_dict(),
            "config": c,
            "epoch": st["epoch"],
            "step_state": dict(st, steps_per_epoch=steps_per_epoch, wall_s_prev=wall_total()),
            "samples_seen": gs_ * global_batch,
            "gumbel_temperature": float(inner.model.quantizer.temperature) if hasattr(inner, "model") else None,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tag": "snapshot",
        }, tmp)
        os.replace(tmp, path)
        log(f"[snapshot] gs={gs_:,} samples={gs_ * global_batch:,} -> {path}")
        if use_wb:
            wandb.log({"global_step": gs_, "ckpt/snapshot_samples_seen": gs_ * global_batch})

    # ---- validation (fixed subset, fixed RNG, all-reduced) --------------------
    def validate():
        net = inner
        net.eval()
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(device) if torch.cuda.is_available() else None
        torch.manual_seed(val_seed + rank)
        keys = ["loss", "ssl_loss", "xyz_loss", "dist_loss", "distpred_loss", "var_loss",
                "contrastive_loss", "diversity_loss", "codevector_perplexity"]
        sums = torch.zeros(len(keys) + 1, device=device, dtype=torch.float64)
        ncls = int(n_classes or 0)
        reg_sum = torch.zeros(max(ncls, 1), device=device, dtype=torch.float64)
        reg_cnt = torch.zeros(max(ncls, 1), device=device, dtype=torch.float64)
        t0 = time.time()
        with torch.no_grad():
            for batch in val_loader:
                x = preprocess_batch(batch[0].float().to(device, non_blocking=True),
                                     net.needs_resampling, net.needs_normalization)
                y = batch[1].to(device)
                coords = batch[2].float().to(device, non_blocking=True) if len(batch) > 2 else None
                kw = {"coords": coords} if coords is not None and (
                    getattr(net, "dist_lambda", 0) or getattr(net, "xyz_lambda", 0)) else {}
                r = net(x, mode="ssl_forward_detailed", **kw)
                vals = [float(r[k].detach()) if torch.is_tensor(r.get(k)) else float(r.get(k) or 0.0) for k in keys]
                if not all(np.isfinite(vals)):
                    continue
                sums += torch.tensor(vals + [1.0], device=device, dtype=torch.float64)
                if ncls:
                    lab = y[(y >= 0) & (y < ncls)]
                    if lab.numel():
                        # batch-level loss attributed to each labeled sample's region
                        reg_sum.index_add_(0, lab, torch.full_like(lab, 0, dtype=torch.float64) + vals[0])
                        reg_cnt.index_add_(0, lab, torch.ones_like(lab, dtype=torch.float64))
        if is_ddp:
            dist.all_reduce(sums)
            dist.all_reduce(reg_sum)
            dist.all_reduce(reg_cnt)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
        net.train()
        n = max(sums[-1].item(), 1.0)
        out = {f"val/{k}": sums[i].item() / n for i, k in enumerate(keys)}
        out["val/n_batches"] = sums[-1].item()
        out["val/time_s"] = time.time() - t0
        if ncls:
            m = reg_cnt > 0
            rl = (reg_sum[m] / reg_cnt[m]).cpu().numpy()
            if len(rl):
                out["val/region_loss_cv"] = float(rl.std() / rl.mean()) if rl.mean() > 0 else float("nan")
                out["val/region_loss_max_min"] = float(rl.max() / rl.min()) if rl.min() > 0 else float("nan")
        return out

    # Diagnostic mode used when changing DDP world size or hardware.  It
    # validates the resumed checkpoint without taking an optimizer step, which
    # separates validation-layout effects from training effects.
    if os.environ.get("VALIDATE_ONLY", "0") == "1":
        v = validate()
        if use_wb:
            v["global_step"] = st["global_step"]
            wandb.log(v)
        if is_main:
            log(f"[val-only gs={st['global_step']:,}] loss={v['val/loss']:.4f} "
                f"dist={v['val/dist_loss']:.4f} ({v['val/time_s']:.0f}s)")
            open(os.path.join(results_dir, "MAX_STEP_DONE"), "w").write(str(st["global_step"]))
        return

    # ---- train loop ----------------------------------------------------------
    keys = ["loss", "ssl_loss", "xyz_loss", "dist_loss", "distpred_loss", "var_loss",
            "contrastive_loss", "diversity_loss", "codevector_perplexity"]
    acc = torch.zeros(len(keys) + 4, device=device, dtype=torch.float64)  # + n, data_wait, nonfinite, gn
    t_win = time.time()
    backbone.train()
    stopping = None
    # codebook-collapse guard (10-12: wav2vec2-large at LR >= 3.5e-4 collapsed: perplexity -> ~G*V or a crash to ~3,
    # pre-clip grad norm -> ~1-5 vs >= 3,000 in every healthy trial, dist loss frozen). Off unless collapse_perplexity > 0.
    # Trips when perplexity > collapse_perplexity, perplexity < collapse_perplexity_low or grad norm < collapse_grad_norm
    # for collapse_windows consecutive log windows.
    collapse_ppl = float(c.get("collapse_perplexity", 0) or 0)
    collapse_ppl_lo = float(c.get("collapse_perplexity_low", 0) or 0)
    collapse_gn = float(c.get("collapse_grad_norm", 0) or 0)
    collapse_need = int(c.get("collapse_windows", 3))
    collapse_min_gs = int(c.get("collapse_min_step", 200))  # skip start-up transients (base on test_ds starts at ppl ~7)
    collapse_run, collapse_hit = 0, False

    for epoch in range(st["epoch"], ssl_epochs):
        sampler.set_epoch(epoch, start=st["step_in_epoch"] * per_gpu_batch * grad_accum
                          if epoch == st["epoch"] else 0)
        st["epoch"] = epoch
        loader = DataLoader(train_dataset, batch_size=per_gpu_batch, sampler=sampler, shuffle=False,
                            num_workers=num_workers, prefetch_factor=prefetch if num_workers > 0 else None,
                            pin_memory=True, persistent_workers=False,
                            worker_init_fn=_worker_ignore_term)
        it = iter(loader)
        while True:
            optimizer.zero_grad(set_to_none=True)
            data_wait = 0.0
            vals_sum = np.zeros(len(keys), dtype=np.float64)
            local_bad = False
            accum_this = 0
            for _micro in range(grad_accum):
                tw = time.perf_counter()
                try:
                    batch = next(it)
                except StopIteration:
                    break
                data_wait += time.perf_counter() - tw
                accum_this += 1

                x = preprocess_batch(batch[0].float().to(device, non_blocking=True),
                                     inner.needs_resampling, inner.needs_normalization)
                coords = batch[2].float().to(device, non_blocking=True) if len(batch) > 2 else None
                needs_coords = coords is not None and (hasattr(inner, "xyz_lambda") or hasattr(inner, "dist_lambda")
                                                       or hasattr(inner, "dist_pred_head")
                                                       or getattr(inner, "distpred_lambda", 0) > 0)
                kw = {"coords": coords} if needs_coords else {}
                if use_amp:
                    with torch.cuda.amp.autocast():
                        r = backbone(x, mode="ssl_forward_detailed", **kw)
                else:
                    r = backbone(x, mode="ssl_forward_detailed", **kw)
                loss = r["loss"] / grad_accum
                # Backward on every rank for every microbatch, preserving DDP's
                # collective order.  Dividing each loss makes the optimizer
                # update an average over all accumulated microbatches.
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                local_bad = local_bad or not bool(torch.isfinite(loss).item())
                vals_sum += np.asarray([
                    float(r[k].detach()) if torch.is_tensor(r.get(k)) else float(r.get(k) or 0.0)
                    for k in keys
                ])
            if accum_this == 0:
                break
            if use_amp:
                scaler.unscale_(optimizer)
            bad = _collective_flag(local_bad)
            if bad:
                optimizer.zero_grad(set_to_none=True)
                st["n_nonfinite"] += 1
                if use_amp:
                    scaler.update()
                acc[len(keys) + 2] += 1
            else:
                gn = torch.nn.utils.clip_grad_norm_(backbone.parameters(), grad_clip_norm) \
                    if grad_clip_norm > 0 else torch.tensor(0.0)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                acc[len(keys) + 3] += float(gn)
                vals = vals_sum / accum_this
                acc[: len(keys)] += torch.tensor(vals, device=device, dtype=torch.float64)
                acc[len(keys)] += 1
            sched.step()
            if hasattr(inner, "step_gumbel_temperature"):
                inner.step_gumbel_temperature()
            acc[len(keys) + 1] += data_wait
            st["global_step"] += 1
            st["step_in_epoch"] += 1
            gs = st["global_step"]

            # ---- log -------------------------------------------------------------
            if gs % log_every == 0:
                buf = acc.clone()
                if is_ddp:
                    dist.all_reduce(buf)
                dt = time.time() - t_win
                n_ok = max(buf[len(keys)].item(), 1.0)
                if collapse_ppl > 0:  # all ranks see the same all-reduced buf
                    ppl_w = buf[keys.index("codevector_perplexity")].item() / n_ok
                    gn_w = buf[len(keys) + 3].item() / n_ok
                    bad = ppl_w > collapse_ppl or ppl_w < collapse_ppl_lo or gn_w < collapse_gn
                    collapse_run = collapse_run + 1 if bad and gs >= collapse_min_gs else 0
                    if collapse_run >= collapse_need and not collapse_hit:
                        collapse_hit = True
                        alert("codebook collapse", f"perplexity {ppl_w:.1f} (ok {collapse_ppl_lo:.0f}-{collapse_ppl:.0f}), pre-clip grad "
                              f"norm {gn_w:.3g} (min {collapse_gn:.3g}) for {collapse_run} log windows at gs={gs:,}; stopping the run",
                              "ERROR", 0)
                if use_wb:
                    d = {f"train/{k}": buf[i].item() / n_ok for i, k in enumerate(keys)}
                    samples_seen = gs * global_batch
                    frac = gs / total_steps
                    usd = cost_now()
                    d.update({
                        "global_step": gs,
                        "train/epoch_float": epoch + st["step_in_epoch"] / steps_per_epoch,
                        "train/samples_seen": samples_seen,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/grad_norm_preclip": buf[len(keys) + 3].item() / n_ok,
                        "train/nonfinite_steps_window": buf[len(keys) + 2].item() / max(world, 1),
                        "train/nonfinite_steps_total": st["n_nonfinite"],
                        "train/gumbel_temperature": float(inner.model.quantizer.temperature) if hasattr(inner, "model") else 0.0,
                        "train/lr_scale": lr_scale_at(gs),
                        "perf/samples_per_s": log_every * global_batch / max(dt, 1e-6),
                        "perf/step_time_s": dt / log_every,
                        "perf/data_wait_frac": buf[len(keys) + 1].item() / max(world, 1) / max(dt, 1e-6),
                        "perf/gpu_mem_max_gb": torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0.0,
                        "cost/wall_hours": wall_total() / 3600.0,
                        "cost/gpu_hours": wall_total() / 3600.0 * world,
                        "cost/usd_est": usd,
                        "cost/usd_projected_total": usd / max(frac, 1e-9) if frac > 0.002 else float("nan"),
                        "cost/eta_hours": (wall_total() / 3600.0) * (1 - frac) / max(frac, 1e-9) if frac > 0.002 else float("nan"),
                    })
                    wandb.log(d)
                    # ---- alerts on the logging window ----
                    nf = buf[len(keys) + 2].item() / max(world, 1)
                    if nf > 0:
                        alert("non-finite loss", f"{nf:.0f} non-finite step(s) in the last {log_every} steps at gs={gs:,} "
                              f"(total {st['n_nonfinite']})", "ERROR", 1800)
                    gnw = buf[len(keys) + 3].item() / n_ok
                    if len(gn_hist) >= 20 and gnw > 20 * float(np.median(gn_hist)):
                        alert("grad-norm spike", f"pre-clip grad norm {gnw:.3g} vs recent median {np.median(gn_hist):.3g} at gs={gs:,}", "ERROR", 1800)
                    gn_hist.append(gnw); del gn_hist[:-200]
                    if budget_hard > 0:
                        for fr in (0.5, 0.75, 0.9):
                            if usd >= fr * budget_hard and fr not in st["cost_alerts_sent"]:
                                st["cost_alerts_sent"].append(fr)
                                alert(f"cost {int(fr*100)}% of cap", f"est ${usd:,.0f} of ${budget_hard:,.0f} at gs={gs:,}; "
                                      f"projected total ${d['cost/usd_projected_total']:,.0f}", "WARN", 0)
                if is_main and gs % (log_every * 20) == 0:
                    log(f"[step {gs:,}/{total_steps:,} ep {epoch}] loss={buf[0].item()/n_ok:.3f} "
                        f"dist={buf[3].item()/n_ok:.4f} lr={optimizer.param_groups[0]['lr']:.2e} "
                        f"{log_every*global_batch/max(dt,1e-6):.0f} samp/s ${cost_now():,.0f}")
                acc.zero_()
                t_win = time.time()

            # ---- validate --------------------------------------------------------
            if gs % val_every == 0:
                v = validate()
                if use_wb:
                    v["global_step"] = gs
                    wandb.log(v)
                if is_main:
                    log(f"[val gs={gs:,}] loss={v['val/loss']:.4f} dist={v['val/dist_loss']:.4f} ({v['val/time_s']:.0f}s)")
                    if np.isfinite(st["best_val_loss"]) and v["val/loss"] > 1.5 * st["best_val_loss"]:
                        alert("val loss spike", f"val/loss {v['val/loss']:.4f} > 1.5x best {st['best_val_loss']:.4f} at gs={gs:,}", "ERROR", 1800)
                    if v["val/loss"] < st["best_val_loss"]:
                        st["best_val_loss"] = v["val/loss"]
                        st["vals_since_best"] = 0
                        save_ckpt(os.path.join(ckpt_dir, f"ssl_{backbone_name}_best_valloss.pt"), "best_val")
                    else:
                        st["vals_since_best"] += 1
                        if st["vals_since_best"] == val_patience:
                            alert("val loss stalled", f"no val/loss improvement for {val_patience} validations "
                                  f"(best {st['best_val_loss']:.4f}, now {v['val/loss']:.4f}, gs={gs:,}). "
                                  f"Consider pause + LR-adjust restart.", "WARN", 0)
                    if use_wb:
                        wandb.log({"global_step": gs, "val/vals_since_best": st["vals_since_best"]})
                t_win = time.time()

            # ---- probe -----------------------------------------------------------
            if probe_every > 0 and probe_epochs > 0 and gs % probe_every == 0:
                if is_main:
                    try:
                        res = quick_probe(backbone, probe_loader, None, device, epoch_count=probe_epochs,
                                          num_classes=n_classes, max_probe_samples=len(val_subset))
                        if res is not None:
                            acc_p, bal_p = res[0], res[1]
                            if use_wb:
                                wandb.log({"global_step": gs, "probe/acc": acc_p, "probe/bal_acc": bal_p})
                            log(f"[probe gs={gs:,}] acc={acc_p:.4f} bal_acc={bal_p:.4f}")
                            if acc_p > st["best_probe_acc"]:
                                st["best_probe_acc"] = acc_p
                                save_ckpt(os.path.join(ckpt_dir, f"ssl_{backbone_name}_best_probe.pt"), "best_probe")
                    except Exception as e:  # noqa: BLE001
                        log(f"[WARN] probe failed at gs={gs}: {e}")
                    backbone.train()
                if is_ddp:
                    dist.barrier()
                t_win = time.time()

            # ---- checkpoint / stop conditions -------------------------------------
            max_gs = int(os.environ.get("MAX_GLOBAL_STEP", c.get("max_global_step", 0)) or 0)
            want_stop_local = collapse_hit or stop_flag["sig"] or os.path.exists(os.path.join(results_dir, "STOP")) or (
                budget_hard > 0 and cost_now() >= budget_hard) or (max_gs > 0 and gs >= max_gs)
            if snap_every_steps and gs % snap_every_steps == 0:
                save_snapshot()
            want_stop = _collective_flag(want_stop_local)
            if gs % ckpt_every == 0 or want_stop:
                save_ckpt(latest, "latest")
                if is_ddp:
                    dist.barrier()
            if want_stop:
                stopping = ("collapse" if collapse_hit else
                            "budget" if budget_hard > 0 and cost_now() >= budget_hard else
                            "signal" if stop_flag["sig"] else
                            "max_global_step" if (max_gs > 0 and gs >= max_gs) else "STOP file")
                break
        if stopping:
            break
        # end of epoch: snapshot + reset position
        st["step_in_epoch"] = 0
        st["epoch"] = epoch + 1
        save_ckpt(os.path.join(ckpt_dir, f"ssl_{backbone_name}_epoch{epoch:02d}.pt"), f"epoch{epoch}")
        save_ckpt(latest, "latest")
        if use_wb:
            wandb.log({"global_step": st["global_step"], "train/epoch_done": epoch})

    if stopping:
        log(f"[step] stopping ({stopping}) at gs={st['global_step']}, cost ${cost_now():,.0f}")
        alert(f"stopped: {stopping}", f"gs={st['global_step']:,} epoch={st['epoch']} cost ${cost_now():,.0f} "
              f"best_val={st['best_val_loss']:.4f}", "WARN" if stopping in ("budget", "signal", "collapse") else "INFO", 0)
        if use_wb:
            wandb.log({"global_step": st["global_step"], "run/stop_reason_budget": float(stopping == "budget")})
        if is_main and stopping == "budget":
            open(os.path.join(results_dir, "BUDGET_STOP"), "w").write(f"{cost_now():.2f}\n")
        if is_main and stopping == "collapse":
            open(os.path.join(results_dir, "COLLAPSE_STOP"), "w").write(str(st["global_step"]))
        if stopping == "max_global_step":
            v = validate()  # all ranks (collective); final val at the step limit (A/B endpoint)
            if use_wb:
                v["global_step"] = st["global_step"]
                wandb.log(v)
            if is_main:
                open(os.path.join(results_dir, "MAX_STEP_DONE"), "w").write(str(st["global_step"]))
        return
    # finished all epochs
    v = validate()
    if use_wb:
        v["global_step"] = st["global_step"]
        wandb.log(v)
    if is_main:
        open(os.path.join(results_dir, "TRAINING_DONE"), "w").write(json.dumps(
            dict(global_step=st["global_step"], cost_usd=cost_now(), best_val_loss=st["best_val_loss"])))
    log(f"[step] DONE gs={st['global_step']:,} cost ${cost_now():,.0f}")
    alert("TRAINING DONE", f"gs={st['global_step']:,} cost ${cost_now():,.0f} best_val={st['best_val_loss']:.4f}", "INFO", 0)
