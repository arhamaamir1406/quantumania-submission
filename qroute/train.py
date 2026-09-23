"""Train the routing value network.

Three losses, in descending order of how much they matter to the final score:

1. **Ranking** (`--w-rank`). Over sibling groups -- successors of the same
   state -- match the softmax over `f = cost + V` to the softmax over the true
   final cost. Search only ever uses `V` to *order* candidates, so this is the
   loss that is literally the objective. Regression can have excellent MAE and
   still order siblings wrongly, because sibling values differ by a fraction of
   a SWAP while the value itself is tens.

2. **Value regression** (`--w-value`). Pinball loss over the value quantiles
   against the true cost-to-go `swaps_to_go + 0.5 * depth_to_go`. Anchors the
   scale, which the ranking loss alone leaves free, and makes `f` comparable
   across beam rounds (beam search compares successors of *different* parents).

3. **Decomposition** (`--w-decomp`). Pinball on the swaps and depth heads
   separately. The value is structurally `swaps + 0.5 * depth`, so this is
   supervision on the two terms the score is made of -- it tells the net
   *why* a state is expensive, not just that it is.

Validation is split by program, never by state, so no trajectory straddles the
split. The headline validation number is not MAE: it is **decision accuracy**
and **regret** on held-out sibling groups -- how often ranking by `f` picks the
successor that actually leads to the best final score, and how much score is
lost when it does not.

Usage::

    python -m qroute.train --preset base --target-states 300000 --workers 12
    python -m qroute.train --preset base --dagger 2 --resume models/value_base.pt
    python -m qroute.train --preset small --distill-from models/value_base.pt
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from starter_kit.hardware import build_hardware_graph

from .collect import Shard, collect
from .features import GraphContext
from .gnn import PRESETS, RoutingNet, build_ctx, load_net
from .mdp import Problem


# ---------------------------------------------------------------------------
# grouping
# ---------------------------------------------------------------------------
def build_groups(group_ids: np.ndarray, min_size: int = 2, ignore: int = -1):
    """-> (padded index array (G, Smax), mask) for groups of at least min_size.

    Rows tagged `ignore` are excluded -- that is how the caller masks out the
    other side of the train/val split without them coalescing into one huge
    bogus group.
    """
    keep = np.flatnonzero(group_ids != ignore)
    if not len(keep):
        return np.zeros((0, min_size), np.int64), np.zeros((0, min_size), bool)
    order = keep[np.argsort(group_ids[keep], kind="stable")]
    gid = group_ids[order]
    bounds = np.flatnonzero(np.diff(gid)) + 1
    chunks = np.split(order, bounds)
    chunks = [c for c in chunks if len(c) >= min_size]
    if not chunks:
        return np.zeros((0, min_size), np.int64), np.zeros((0, min_size), bool)
    smax = max(len(c) for c in chunks)
    idx = np.zeros((len(chunks), smax), np.int64)
    mask = np.zeros((len(chunks), smax), bool)
    for i, c in enumerate(chunks):
        idx[i, :len(c)] = c
        mask[i, :len(c)] = True
    return idx, mask


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------
def pinball(pred: torch.Tensor, target: torch.Tensor, taus: torch.Tensor):
    """Quantile regression loss. pred (B, Q), target (B,), taus (Q,)."""
    d = target.unsqueeze(-1) - pred
    return torch.maximum(taus * d, (taus - 1.0) * d).mean()


def listwise_rank(f: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor,
                  temp: float = 0.5):
    """Soft cross-entropy between softmax(-f/T) and softmax(-truth/T).

    Padded slots are given a large finite penalty rather than -inf: with -inf a
    padded slot contributes `0 * -inf = nan`, and the masked terms are dropped
    explicitly afterwards for the same reason.
    """
    neg = -1e4
    fm = (-f / temp).masked_fill(~mask, neg)
    tm = (-truth / temp).masked_fill(~mask, neg)
    logp = F.log_softmax(fm, dim=-1)
    q = F.softmax(tm, dim=-1)
    term = torch.where(mask, q * logp, torch.zeros_like(logp))
    return -term.sum(-1).mean()


def pairwise_margin(f: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor):
    """Hinge on every ordered pair whose true costs differ."""
    diff_f = f.unsqueeze(-1) - f.unsqueeze(-2)
    diff_t = truth.unsqueeze(-1) - truth.unsqueeze(-2)
    pair = mask.unsqueeze(-1) & mask.unsqueeze(-2) & (diff_t.abs() > 1e-6)
    if not pair.any():
        return f.sum() * 0.0
    target = torch.sign(diff_t)
    margin = diff_t.abs().clamp(max=2.0)
    loss = F.relu(margin - target * diff_f)
    loss = torch.where(pair, loss, torch.zeros_like(loss))
    return loss.sum() / pair.sum().clamp(min=1)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------
class EMA:
    def __init__(self, net, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in net.state_dict().items() if v.is_floating_point()}

    @torch.no_grad()
    def update(self, net):
        for k, v in net.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)

    def apply_to(self, net):
        out = copy.deepcopy(net)
        sd = out.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v.to(sd[k].dtype))
        return out


# ---------------------------------------------------------------------------
# data on device
# ---------------------------------------------------------------------------
class DeviceData:
    """The whole dataset resident on the device.

    Features are kept in fp16 (they are all bounded, normalised quantities) and
    cast per batch -- that halves the footprint, so a few hundred thousand
    states fit alongside the model on one card.
    """

    def __init__(self, shard: Shard, device, store=torch.float16,
                 compute=torch.float32):
        def t(a, dtype=None):
            x = torch.from_numpy(np.ascontiguousarray(a))
            return x.to(device=device, dtype=dtype) if dtype else x.to(device)

        self.compute = compute
        self.node = t(shard.node, store)
        self.edge = t(shard.edge, store)
        self.gate = t(shard.gate, store)
        self.glob = t(shard.glob, store)
        self.gate_pa = t(shard.gate_pa.astype(np.int64))
        self.gate_pb = t(shard.gate_pb.astype(np.int64))
        self.base_cost = t(shard.base_cost, torch.float32)
        self.y_swaps = t(shard.y_swaps, torch.float32)
        self.y_depth = t(shard.y_depth, torch.float32)
        self.y = self.y_swaps + 0.5 * self.y_depth

    def batch(self, idx):
        c = self.compute
        return {
            "node": self.node[idx].to(c), "edge": self.edge[idx].to(c),
            "gate": self.gate[idx].to(c), "glob": self.glob[idx].to(c),
            "gate_pa": self.gate_pa[idx], "gate_pb": self.gate_pb[idx],
        }


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(net, data: DeviceData, rows: np.ndarray, gidx: np.ndarray,
             gmask: np.ndarray, ctx, device, batch=2048, q=None):
    net.eval()
    qi = net.n_quantiles // 2 if q is None else q
    preds = torch.zeros(len(data.y), device=device)
    for i in range(0, len(rows), batch):
        b = torch.from_numpy(rows[i:i + batch]).to(device)
        preds[b] = net(data.batch(b), ctx)["value"][:, qi].float()
    mae = (preds[torch.from_numpy(rows).to(device)] -
           data.y[torch.from_numpy(rows).to(device)]).abs().mean().item()

    acc = regret = float("nan")
    if len(gidx):
        gi = torch.from_numpy(gidx).to(device)
        gm = torch.from_numpy(gmask).to(device)
        f = preds[gi] + data.base_cost[gi]
        truth = data.y[gi] + data.base_cost[gi]
        big = torch.finfo(f.dtype).max
        f = f.masked_fill(~gm, big)
        truth_m = truth.masked_fill(~gm, big)
        pick = f.argmin(-1)
        chosen = truth_m.gather(-1, pick[:, None]).squeeze(-1)
        best = truth_m.min(-1).values
        acc = (chosen <= best + 1e-6).float().mean().item()
        regret = (chosen - best).mean().item()
    return mae, acc, regret


def benchmark(net, device, budget=10.0, quantile=None, beam=None):
    """End-to-end: run the real portfolio with this net on the six benchmarks."""
    from starter_kit.benchmarks import BENCHMARKS
    from starter_kit.scorer import score_summary

    from .portfolio import solve

    hw = build_hardware_graph()
    total, rows = 0.0, []
    for name, prog in BENCHMARKS.items():
        pl, rt = solve(prog, hw, budget=budget, net=net, net_device=device,
                       net_quantile=quantile, net_beam=beam)
        s = score_summary(prog, hw, pl, rt)
        assert s["valid"], f"{name}: {s['message']}"
        total += s["score"]
        rows.append((name, s["score"]))
    return total, rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # data
    ap.add_argument("--target-states", type=int, default=300_000,
                    help="dataset size in labelled states (the primary knob)")
    ap.add_argument("--programs", type=int, default=10 ** 9,
                    help="safety cap on programs; --target-states normally binds first")
    ap.add_argument("--workers", type=int, default=0, help="0 = cpu_count-1")
    ap.add_argument("--teacher-width", type=int, default=400)
    ap.add_argument("--label-width", type=int, default=60)
    ap.add_argument("--max-paths", type=int, default=12)
    ap.add_argument("--groups-per-program", type=int, default=6)
    ap.add_argument("--siblings", type=int, default=8)
    ap.add_argument("--max-qubits", type=int, default=20)
    ap.add_argument("--lookahead", type=int, default=0,
                    help="sibling groups over the wide move set (pre-positioning "
                         "SWAPs for the next N gates); trains a pruning policy")
    ap.add_argument("--collect-cap", type=float, default=14400.0,
                    help="per-worker wall-clock cap on collection, seconds")
    ap.add_argument("--data", default=None, help="reuse a saved .npz dataset")
    ap.add_argument("--save-data", default=None)
    ap.add_argument("--drive", default=None,
                    help="net that drives round-0 collection (teacher + labels)")
    ap.add_argument("--wide-labels", action="store_true",
                    help="with --lookahead: label siblings with the driving net's "
                         "pruned wide search, not the narrow beam")
    ap.add_argument("--collect-device", default="cpu",
                    help="device for the driving net inside collection workers")
    ap.add_argument("--extra-data", default=None,
                    help="an earlier .npz dataset to train on alongside the new one")
    ap.add_argument("--dagger", type=int, default=0,
                    help="extra collect+train rounds driven by the current net")
    # model
    ap.add_argument("--preset", default="base", choices=list(PRESETS))
    ap.add_argument("--quantiles", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--distill-from", default=None)
    ap.add_argument("--w-distill", type=float, default=1.0)
    # optimisation
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--groups-per-batch", type=int, default=48)
    ap.add_argument("--singles-per-batch", type=int, default=256)
    ap.add_argument("--steps-per-epoch", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--w-rank", type=float, default=1.0)
    ap.add_argument("--w-pair", type=float, default=0.5)
    ap.add_argument("--w-value", type=float, default=1.0)
    ap.add_argument("--w-decomp", type=float, default=0.5)
    ap.add_argument("--rank-temp", type=float, default=0.5)
    ap.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "off"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    ap.add_argument("--benchmark", action="store_true",
                    help="score the six public benchmarks with the trained net")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    out_path = Path(args.out or f"models/value_{args.preset}.pt")
    if args.workers == 0:
        import os
        args.workers = max(1, (os.cpu_count() or 2) - 1)

    print(f"device {dev}   preset {args.preset}   out {out_path}")

    # ---- model ----------------------------------------------------------
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        net = RoutingNet.from_checkpoint(ck).to(dev)
        print(f"resumed {args.resume}")
    else:
        net = RoutingNet.from_preset(args.preset, n_quantiles=args.quantiles,
                                     dropout=args.dropout).to(dev)
    print(f"parameters: {net.n_params:,}")

    teacher = None
    if args.distill_from:
        teacher = load_net(args.distill_from, str(dev))
        if teacher is None:
            raise SystemExit(f"could not load teacher {args.distill_from}")
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"distilling from {args.distill_from} ({teacher.n_params:,} params)")

    ctx_graph = GraphContext(Problem([("2Q", 0, 1)], build_hardware_graph()),
                             window=net.window)
    ctx = build_ctx(None, ctx_graph, dev, torch.float32)

    rounds = 1 + max(0, args.dagger)
    best_overall = None
    for rnd in range(rounds):
        # ---- data --------------------------------------------------------
        if args.data and rnd == 0:
            print(f"loading dataset {args.data}")
            shard = Shard.load(args.data)
            print(f"  {len(shard)} states")
        else:
            drive = str(out_path) if rnd > 0 and out_path.exists() else args.drive
            print(f"[round {rnd}] collecting {args.target_states:,} states on "
                  f"{args.workers} workers" + (f", driven by {drive}" if drive else ""))
            shard = collect(args.programs, target_states=args.target_states,
                            seed=args.seed + 31 * rnd,
                            max_qubits=args.max_qubits,
                            teacher_width=args.teacher_width,
                            label_width=args.label_width, max_paths=args.max_paths,
                            groups_per_program=args.groups_per_program,
                            siblings=args.siblings, time_cap=args.collect_cap,
                            workers=args.workers, net_path=drive,
                            device=args.collect_device,
                            lookahead=args.lookahead, wide_labels=args.wide_labels)
            if args.save_data and rnd == 0:
                Path(args.save_data).parent.mkdir(parents=True, exist_ok=True)
                shard.save(args.save_data)
                print(f"  saved {args.save_data}")

        if args.extra_data and rnd == 0:
            shard = Shard.concat([shard, Shard.load(args.extra_data)])
            print(f"  + {args.extra_data}: {len(shard)} states total")
        data = DeviceData(shard, dev)
        n = len(shard)
        y = shard.y_swaps + 0.5 * shard.y_depth
        print(f"  cost-to-go: mean {y.mean():.2f}  p90 {np.percentile(y, 90):.2f}  "
              f"max {y.max():.2f}")

        # split by program
        progs = np.unique(shard.prog_id)
        rs = np.random.default_rng(args.seed)
        rs.shuffle(progs)
        val_progs = set(progs[:max(1, int(0.1 * len(progs)))].tolist())
        is_val = np.isin(shard.prog_id, list(val_progs))
        tr_rows = np.flatnonzero(~is_val)
        va_rows = np.flatnonzero(is_val)

        tr_g, tr_gm = build_groups(np.where(is_val, -1, shard.group))
        va_g, va_gm = build_groups(np.where(is_val, shard.group, -1))
        singles = tr_rows
        print(f"  train {len(tr_rows)} rows / {len(tr_g)} ranking groups   "
              f"val {len(va_rows)} rows / {len(va_g)} groups")

        # ---- optimiser ---------------------------------------------------
        decay, no_decay = [], []
        for nm, p in net.named_parameters():
            (no_decay if p.ndim <= 1 or "bias" in nm else decay).append(p)
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": args.wd},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=args.lr, betas=(0.9, 0.95))
        total_steps = args.epochs * args.steps_per_epoch
        warm = max(1, int(args.warmup * total_steps))

        def lr_at(step):
            if step < warm:
                return step / warm
            t = (step - warm) / max(1, total_steps - warm)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
        amp = args.amp
        if amp == "auto":
            amp = "bf16" if dev.type == "cuda" and torch.cuda.is_bf16_supported() else "off"
        amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)
        scaler = torch.amp.GradScaler(dev.type, enabled=(amp == "fp16"))

        def autocast():
            if amp_dtype is None:
                return contextlib.nullcontext()
            return torch.autocast(dev.type, dtype=amp_dtype)

        ema = EMA(net, args.ema)
        taus = torch.linspace(0.5 / net.n_quantiles, 1 - 0.5 / net.n_quantiles,
                              net.n_quantiles, device=dev)
        ctx_train = dict(ctx); ctx_train["pe_flip"] = True

        rng = np.random.default_rng(args.seed + rnd)
        best = (float("inf"), None)
        step = 0
        for ep in range(1, args.epochs + 1):
            net.train()
            t0 = time.monotonic()
            agg = np.zeros(5)
            for _ in range(args.steps_per_epoch):
                gsel = rng.integers(0, max(1, len(tr_g)), args.groups_per_batch) \
                    if len(tr_g) else np.zeros(0, np.int64)
                gi = tr_g[gsel] if len(tr_g) else np.zeros((0, 2), np.int64)
                gm = tr_gm[gsel] if len(tr_g) else np.zeros((0, 2), bool)
                ssel = singles[rng.integers(0, len(singles), args.singles_per_batch)]
                flat = np.concatenate([gi.reshape(-1), ssel]) if len(gi) else ssel
                uniq, inv = np.unique(flat, return_inverse=True)
                rows = torch.from_numpy(uniq).to(dev)

                with autocast():
                    pred = net(data.batch(rows), ctx_train)
                v = pred["value"].float()
                qs, qd = pred["swaps"].float(), pred["depth"].float()
                yv, ys, yd = data.y[rows], data.y_swaps[rows], data.y_depth[rows]

                l_val = pinball(v, yv, taus)
                l_dec = pinball(qs, ys, taus) + pinball(qd, yd, taus)

                l_rank = torch.zeros((), device=dev)
                l_pair = torch.zeros((), device=dev)
                if len(gi):
                    back = torch.from_numpy(
                        inv[:gi.size].reshape(gi.shape)).to(dev)
                    mk = torch.from_numpy(gm).to(dev)
                    vmed = v[:, net.n_quantiles // 2]
                    f = (vmed[back] + data.base_cost[torch.from_numpy(gi).to(dev)])
                    truth = (data.y[torch.from_numpy(gi).to(dev)] +
                             data.base_cost[torch.from_numpy(gi).to(dev)])
                    l_rank = listwise_rank(f, truth, mk, args.rank_temp)
                    l_pair = pairwise_margin(f, truth, mk)

                l_dist = torch.zeros((), device=dev)
                if teacher is not None:
                    with torch.no_grad(), autocast():
                        tp = teacher(data.batch(rows), ctx)
                    l_dist = (F.smooth_l1_loss(v, tp["value"].float()) +
                              0.5 * F.smooth_l1_loss(qs, tp["swaps"].float()) +
                              0.5 * F.smooth_l1_loss(qd, tp["depth"].float()))

                loss = (args.w_value * l_val + args.w_decomp * l_dec +
                        args.w_rank * l_rank + args.w_pair * l_pair +
                        args.w_distill * l_dist)

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
                scaler.step(opt)
                scaler.update()
                sched.step()
                ema.update(net)
                step += 1
                agg += [loss.item(), l_val.item(), l_dec.item(),
                        l_rank.detach().item(), l_pair.detach().item()]

            agg /= args.steps_per_epoch
            eval_net = ema.apply_to(net)
            mae, acc, reg = evaluate(eval_net, data, va_rows, va_g, va_gm, ctx, dev)
            score = reg if np.isfinite(reg) else mae
            if score < best[0]:
                best = (score, {k: v.detach().cpu().clone()
                                for k, v in eval_net.state_dict().items()})
            if ep % 5 == 0 or ep == 1 or ep == args.epochs:
                print(f"  ep {ep:3d} loss {agg[0]:.4f} (val {agg[1]:.3f} dec "
                      f"{agg[2]:.3f} rank {agg[3]:.3f} pair {agg[4]:.3f})  "
                      f"| val MAE {mae:.3f}  decision-acc {acc:.3f}  "
                      f"regret {reg:.4f}  [{time.monotonic()-t0:.0f}s]")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best[1], "cfg": net.cfg,
                    "preset": args.preset, "val_regret": best[0],
                    "n_states": int(n)}, out_path)
        print(f"[round {rnd}] saved {out_path}  (val regret {best[0]:.4f})")
        net.load_state_dict({k: v.to(dev) for k, v in best[1].items()})
        best_overall = best[0]

    if args.benchmark:
        print("\nend-to-end benchmark with the trained net:")
        total, rows = benchmark(net, str(dev))
        for name, sc in rows:
            print(f"  {name:15s} {sc:6.1f}")
        print(f"  {'TOTAL':15s} {total:6.1f}   (hand-heuristic portfolio: 91.5)")
    return best_overall


if __name__ == "__main__":
    main()
