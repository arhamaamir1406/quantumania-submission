"""Training-data generation for the routing value network.

Two kinds of labelled example, both scored by the same expert:

* **trajectory states** -- every state on a strong beam search's winning path,
  labelled with the expert's realised cost-to-go. Cheap, and it teaches the
  overall scale of the value function.

* **sibling groups** -- at sampled decision points we expand the *whole*
  move-set successor list and run an independent beam from each successor, so
  every sibling carries its own cost-to-go estimate. These are what the ranking
  loss trains on, and they are the only examples that teach the thing search
  actually needs: which of several near-identical-looking successors is
  genuinely better. Regression alone cannot learn that, because the value
  differences between siblings are far smaller than the value's own scale.

States are recorded at *gate-round boundaries* -- exactly the states
`beam_search` scores -- so the training distribution matches the inference one.

`--dagger` rounds re-collect from the current network's own rollouts, which
fixes the usual failure of pure imitation: the expert never visits the states
the student's mistakes lead to.
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass

import numpy as np

from starter_kit.hardware import build_hardware_graph

from .features import Encoder, GraphContext
from .generate import random_program
from .mdp import Problem, State
from .placement import (constructive_placement, embed_placement,
                        identity_placement, random_placement)
from .search import beam_search, gate_moves, greedy_rollout, wide_moves

FEATURE_KEYS = ("node", "edge", "gate", "glob")
INDEX_KEYS = ("gate_pa", "gate_pb")


def round_states(problem: Problem, placement: dict[int, int],
                 ops: list[tuple]) -> list[State]:
    """Replay a solution, returning the state after each executed gate round.

    These are the states `beam_search` ranks, so this is the distribution the
    value net must be accurate on.
    """
    s = problem.initial(placement)
    out = [s]
    for op in ops:
        if op[0] == "SWAP":
            s = problem.apply_swap_raw(s, op[1], op[2])
        else:
            t = problem.advance(s)
            if t.k > s.k:
                s = t
                out.append(s)
    return out


def _final_stats(problem: Problem, s: State) -> tuple[float, float]:
    return float(s.swaps), float(s.depth)


def label_cost_to_go(problem: Problem, s: State, width: int,
                     max_paths: int, wide: dict | None = None) -> tuple[float, float] | None:
    """(swaps_to_go, depth_to_go) from an independent beam started at `s`.

    With `wide` (beam_search kwargs: lookahead, prune, prune_fn, value_fn) the
    label comes from the policy-pruned wide search, so a pre-positioning move
    is valued by what further pre-positioning can make of it.
    """
    if problem.is_terminal(s):
        return 0.0, 0.0
    final = beam_search(problem, s, width=width, max_paths=max_paths, **(wide or {}))
    if final is None:
        return None
    return float(final.swaps - s.swaps), float(final.depth - s.depth)


def _placements(prog, hw, rng):
    out = []
    emb = embed_placement(prog, hw)
    if emb is not None:
        out.append(emb)
    out.append(constructive_placement(prog, hw))
    out.append(constructive_placement(prog, hw, rng, jitter=1.5))
    out.append(identity_placement(prog, hw))
    out.append(random_placement(prog, hw, rng))
    return out


@dataclass
class Shard:
    node: np.ndarray
    edge: np.ndarray
    gate: np.ndarray
    glob: np.ndarray
    gate_pa: np.ndarray
    gate_pb: np.ndarray
    base_cost: np.ndarray
    y_swaps: np.ndarray
    y_depth: np.ndarray
    group: np.ndarray
    prog_id: np.ndarray

    def save(self, path):
        np.savez_compressed(path, **{k: getattr(self, k) for k in self.__dataclass_fields__})

    @staticmethod
    def load(path) -> "Shard":
        z = np.load(path)
        return Shard(**{k: z[k] for k in Shard.__dataclass_fields__})

    @staticmethod
    def concat(shards: list["Shard"]) -> "Shard":
        shards = [s for s in shards if len(s.base_cost)]
        if not shards:
            raise ValueError("no data collected")
        out, gbase, pbase = {}, 0, 0
        parts = {k: [] for k in Shard.__dataclass_fields__}
        for s in shards:
            for k in parts:
                v = getattr(s, k)
                if k == "group":
                    v = v + gbase
                elif k == "prog_id":
                    v = v + pbase
                parts[k].append(v)
            gbase = int(parts["group"][-1].max()) + 1
            pbase = int(parts["prog_id"][-1].max()) + 1
        for k, v in parts.items():
            out[k] = np.concatenate(v)
        return Shard(**out)

    def __len__(self):
        return len(self.base_cost)


class _Accumulator:
    def __init__(self):
        self.rows: dict[str, list] = {k: [] for k in Shard.__dataclass_fields__}
        self.group = 0
        self.n_pre = 0          # pre-positioning siblings attempted (for reporting)

    def add(self, enc: Encoder, states: list[State], labels: list[tuple],
            prog_id: int, same_group: bool):
        if not states:
            return
        feats = enc.encode_numpy(states)
        gid = self.group if same_group else -1
        for i, (ys, yd) in enumerate(labels):
            for k in FEATURE_KEYS:
                self.rows[k].append(feats[k][i].astype(np.float16))
            for k in INDEX_KEYS:
                self.rows[k].append(feats[k][i].astype(np.int16))
            self.rows["base_cost"].append(np.float32(feats["cost"][i]))
            self.rows["y_swaps"].append(np.float32(ys))
            self.rows["y_depth"].append(np.float32(yd))
            self.rows["group"].append(np.int32(self.group if same_group else self.group + i))
            self.rows["prog_id"].append(np.int32(prog_id))
        self.group += 1 if same_group else len(labels)

    def shard(self) -> Shard:
        if not self.rows["base_cost"]:
            empty = lambda dt: np.zeros((0,), dtype=dt)  # noqa: E731
            return Shard(np.zeros((0, 1, 1), np.float16), np.zeros((0, 1, 1), np.float16),
                         np.zeros((0, 1, 1), np.float16), np.zeros((0, 1), np.float16),
                         np.zeros((0, 1), np.int16), np.zeros((0, 1), np.int16),
                         empty(np.float32), empty(np.float32), empty(np.float32),
                         empty(np.int32), empty(np.int32))
        return Shard(**{k: np.stack(v) if v and np.ndim(v[0]) else np.array(v)
                        for k, v in self.rows.items()})


def collect_one(prog, hw, rng, acc: _Accumulator, prog_id: int, *,
                teacher_width: int, label_width: int, max_paths: int,
                groups_per_program: int, siblings: int,
                value_fn_factory=None, lookahead: int = 0,
                wide_labels: bool = False) -> int:
    """Collect from one program. Returns the number of examples added."""
    problem = Problem(prog, hw)
    if problem.n_ops == 0:
        return 0
    enc = Encoder(problem, GraphContext(problem))
    before = len(acc.rows["base_cost"])

    vf = value_fn_factory(problem) if value_fn_factory is not None else None
    wide = (dict(lookahead=lookahead, prune=4, prune_fn=vf, value_fn=vf)
            if wide_labels and vf is not None and lookahead > 0 else None)
    placement = rng.choice(_placements(prog, hw, rng))
    s0 = problem.initial(placement)

    # -- expert trajectory ------------------------------------------------
    if problem.is_terminal(s0):
        final = s0
    elif wide is not None:
        final = beam_search(problem, s0, width=teacher_width, max_paths=max_paths, **wide)
    else:
        final = beam_search(problem, s0, width=teacher_width, max_paths=max_paths,
                            value_fn=vf)
        if final is None:
            return 0
    traj = round_states(problem, placement, final.ops())
    fs, fd = _final_stats(problem, final)
    acc.add(enc, traj, [(fs - s.swaps, fd - s.depth) for s in traj],
            prog_id, same_group=False)

    # -- sibling groups at sampled decision points -------------------------
    interior = [s for s in traj if not problem.is_terminal(s)]
    if interior and groups_per_program:
        picks = rng.sample(interior, min(groups_per_program, len(interior)))
        for s in picks:
            if lookahead > 0:
                # Wide groups for the pruning policy: half ordinary moves, half
                # pre-positioning ones, so both are always represented.
                narrow = gate_moves(problem, s, max_paths)
                keys = {(t.k, t.pos) for t in narrow}
                pre = [t for t in wide_moves(problem, s, max_paths, lookahead)
                       if (t.k, t.pos) not in keys]
                n_pre = min(len(pre), max(siblings // 2, siblings - len(narrow)))
                moves = rng.sample(pre, n_pre)
                moves += rng.sample(narrow, min(len(narrow), siblings - n_pre))
                acc.n_pre += n_pre
            else:
                moves = gate_moves(problem, s, max_paths)
            if len(moves) < 2:
                continue
            if len(moves) > siblings:
                moves = rng.sample(moves, siblings)
            labelled, kept = [], []
            for t in moves:
                lab = label_cost_to_go(problem, t, label_width, max_paths, wide)
                if lab is None:
                    continue
                labelled.append(lab)
                kept.append(t)
            if len(kept) >= 2:
                acc.add(enc, kept, labelled, prog_id, same_group=True)

    return len(acc.rows["base_cost"]) - before


def _worker(job: dict):
    """One process: collect until it hits its share of the state target."""
    hw = build_hardware_graph()
    rng = random.Random(job["seed"])
    acc = _Accumulator()
    wid = job["wid"]
    target = job["target_states"]
    cap = job["n_programs"]

    factory = None
    net_path = job["net_path"]
    if net_path and os.path.exists(net_path):
        try:
            import torch
            torch.set_num_threads(1)        # one core per worker; no oversubscription
            from .gnn import load_net, make_value_fn
            net = load_net(net_path, job["device"])
            if net is not None:
                def factory(problem, _net=net):
                    return make_value_fn(_net, problem, device=job["device"],
                                         quantile=job["quantile"], max_batch=256)
        except Exception:
            factory = None

    t0 = time.monotonic()
    last = t0
    made = 0
    while made < cap:
        now = time.monotonic()
        if now - t0 > job["time_cap"]:
            break
        if target and len(acc.rows["base_cost"]) >= target:
            break
        kind, prog = random_program(rng, job["max_qubits"])
        if not prog:
            continue
        try:
            collect_one(prog, hw, rng, acc, made,
                        teacher_width=job["teacher_width"],
                        label_width=job["label_width"],
                        max_paths=job["max_paths"],
                        groups_per_program=job["groups_per_program"],
                        siblings=job["siblings"], value_fn_factory=factory,
                        lookahead=job.get("lookahead", 0),
                        wide_labels=job.get("wide_labels", False))
        except Exception:
            pass
        made += 1
        if job["progress"] and now - last > job["progress"]:
            last = now
            got = len(acc.rows["base_cost"])
            frac = f"/{target}" if target else ""
            rate = got / max(1e-6, now - t0)
            print(f"    [w{wid}] {got}{frac} states ({acc.n_pre} pre-moves), {made} programs, "
                  f"{now - t0:.0f}s ({rate:.0f} states/s)", flush=True)
    return acc.shard()


def _spawn_safe() -> bool:
    """Can `multiprocessing` with the spawn start method re-import __main__?

    On Windows every worker re-imports the parent's __main__ by path. If the
    session was started from stdin, a REPL, or a notebook there is no such
    path, and each worker fails *and respawns*. Falling back to one process is
    far better than that.
    """
    import sys
    main = sys.modules.get("__main__")
    path = getattr(main, "__file__", None)
    return bool(path) and os.path.exists(path)


def collect(n_programs: int = 10 ** 9, *, target_states: int | None = None,
            seed: int = 7, max_qubits: int = 20,
            teacher_width: int = 400, label_width: int = 60, max_paths: int = 12,
            groups_per_program: int = 4, siblings: int = 8,
            time_cap: float = 3600.0, workers: int = 0,
            net_path: str | None = None, device: str = "cpu",
            quantile: int | None = None, verbose: bool = True,
            progress: float = 60.0, lookahead: int = 0,
            wide_labels: bool = False) -> Shard:
    """Generate a dataset across `workers` processes.

    Prefer `target_states` to `n_programs`: yield per program swings by an
    order of magnitude across the program families (a 5-gate chain and an
    80-gate dense instance are both drawn uniformly), so a program count is a
    poor way to ask for a dataset size. With a target, each worker collects its
    share and stops; `n_programs` and `time_cap` remain as safety caps.
    """
    workers = workers or 1
    if workers > 1 and not _spawn_safe():
        if verbose:
            print("  [collect] __main__ is not importable by spawned workers "
                  "(stdin/REPL/notebook); falling back to a single process")
        workers = 1

    share = (target_states + workers - 1) // workers if target_states else None
    base = dict(max_qubits=max_qubits, teacher_width=teacher_width,
                label_width=label_width, max_paths=max_paths,
                groups_per_program=groups_per_program, siblings=siblings,
                time_cap=time_cap, net_path=net_path, device=device,
                quantile=quantile, target_states=share, lookahead=lookahead,
                wide_labels=wide_labels,
                n_programs=max(1, n_programs // workers),
                progress=progress if verbose else 0.0)
    jobs = [dict(base, seed=seed + 1000 * w, wid=w) for w in range(workers)]

    t0 = time.monotonic()
    if workers == 1:
        shards = [_worker(jobs[0])]
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            shards = pool.map(_worker, jobs)
    data = Shard.concat(shards)
    if verbose:
        n_groups = len(np.unique(data.group))
        sized = np.bincount(data.group - data.group.min())
        print(f"  collected {len(data)} states from {len(np.unique(data.prog_id))} "
              f"programs in {time.monotonic()-t0:.0f}s "
              f"({n_groups} groups, {int((sized > 1).sum())} ranking groups)")
    return data
