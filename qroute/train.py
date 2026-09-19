"""Train the GraphSAGE value function on beam-search trajectories.

Data generation: solve randomly generated programs with beam search under the
hand-written heuristic, then label every state on the winning trajectory with
its true cost-to-go (final_cost - state_cost). The net is then a drop-in
replacement for the heuristic, and search under it can only be as good as the
teacher plus whatever the net generalises -- which is why we also keep the
hand heuristic in the portfolio.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from starter_kit.hardware import build_hardware_graph

from .generate import random_program
from .gnn import Encoder, ValueNet
from .mdp import Problem
from .placement import constructive_placement, embed_placement
from .search import beam_search, path_states


def collect(n_programs: int, width: int, seed: int, max_qubits: int,
            time_cap: float, verbose: bool = True):
    hw = build_hardware_graph()
    rng = random.Random(seed)
    XN, XG, Y = [], [], []
    kinds: dict[str, int] = {}
    t0 = time.monotonic()
    made = 0
    while made < n_programs:
        if time.monotonic() - t0 > time_cap:
            print(f"  [collect] time cap hit after {made} programs")
            break
        kind, prog = random_program(rng, max_qubits)
        if not prog:
            continue
        problem = Problem(prog, hw)
        emb = embed_placement(prog, hw)
        placement = emb if emb is not None else constructive_placement(prog, hw)
        s0 = problem.initial(placement)
        final = beam_search(problem, s0, width=width)
        if final is None:
            if problem.is_terminal(s0):
                final = s0
            else:
                continue
        states = path_states(problem, placement, final.ops())
        enc = Encoder(problem)
        xn, xg, _ = enc.encode(states)
        XN.append(xn.numpy())
        XG.append(xg.numpy())
        Y.append(np.array([final.cost - s.cost for s in states], dtype=np.float32))
        kinds[kind] = kinds.get(kind, 0) + 1
        made += 1
        if verbose and made % 25 == 0:
            print(f"  [collect] {made}/{n_programs} programs, "
                  f"{sum(len(y) for y in Y)} states, {time.monotonic()-t0:.0f}s")
    return (np.concatenate(XN), np.concatenate(XG), np.concatenate(Y), kinds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--programs", type=int, default=250)
    ap.add_argument("--width", type=int, default=250)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-qubits", type=int, default=20)
    ap.add_argument("--collect-cap", type=float, default=600.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="models/value_sage.pt")
    args = ap.parse_args()

    print(f"device: {args.device}")
    print(f"collecting trajectories from {args.programs} random programs "
          f"(beam width {args.width})")
    xn, xg, y, kinds = collect(args.programs, args.width, args.seed,
                               args.max_qubits, args.collect_cap)
    print(f"  {len(y)} labelled states; shapes {xn.shape} {xg.shape}")
    print(f"  program mix: {dict(sorted(kinds.items()))}")
    print(f"  cost-to-go: mean {y.mean():.2f}  max {y.max():.2f}")

    n = len(y)
    idx = np.random.default_rng(args.seed).permutation(n)
    split = int(0.9 * n)
    tr, va = idx[:split], idx[split:]

    dev = torch.device(args.device)
    a_norm = Encoder(Problem([("2Q", 0, 1)], build_hardware_graph())).a_norm.to(dev)
    XN = torch.from_numpy(xn).to(dev)
    XG = torch.from_numpy(xg).to(dev)
    Y = torch.from_numpy(y).to(dev)

    net = ValueNet(hidden=args.hidden, layers=args.layers).to(dev)
    print(f"  params: {sum(p.numel() for p in net.parameters()):,}")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.SmoothL1Loss()

    # baseline: predicting the training mean
    base_mae = (Y[va] - Y[tr].mean()).abs().mean().item()
    best_mae, best_state = float("inf"), None

    for ep in range(1, args.epochs + 1):
        net.train()
        perm = tr[np.random.permutation(len(tr))]
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            b = torch.from_numpy(perm[i:i + args.batch]).to(dev)
            pred = net(XN[b], XG[b], a_norm)
            loss = loss_fn(pred, Y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        net.eval()
        with torch.no_grad():
            vb = torch.from_numpy(va).to(dev)
            pv = net(XN[vb], XG[vb], a_norm)
            mae = (pv - Y[vb]).abs().mean().item()
        if mae < best_mae:
            best_mae = mae
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            print(f"  epoch {ep:3d}  train {tot/len(perm):.4f}  val MAE {mae:.3f}"
                  f"  (mean-predictor {base_mae:.3f})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "hidden": args.hidden,
                "layers": args.layers, "val_mae": best_mae,
                "mean_predictor_mae": base_mae, "n_states": int(n)}, out)
    print(f"\nsaved {out}  val MAE {best_mae:.3f} vs mean-predictor {base_mae:.3f}"
          f"  ({100*(1-best_mae/base_mae):.1f}% better)")


if __name__ == "__main__":
    main()
