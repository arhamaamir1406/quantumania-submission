"""Random program generator -- our own held-out test set, and GNN training data.

Covers the shape distribution the public benchmarks sample from (stars, chains,
ladders, brickwork, sparse and dense random) plus a few they do not, so neither
the tuning nor the value net overfits to six instances.
"""
from __future__ import annotations

import random


def star(rng, n):
    leaves = list(range(1, n))
    rng.shuffle(leaves)
    return [("2Q", 0, l) for l in leaves]


def chain(rng, n):
    return [("2Q", i, i + 1) for i in range(n - 1)]


def ladder(rng, cols):
    top, bot = list(range(cols)), list(range(cols, 2 * cols))
    p = [("2Q", r[i], r[i + 1]) for r in (top, bot) for i in range(cols - 1)]
    p += [("2Q", top[i], bot[i]) for i in range(cols)]
    return p


def brickwork(rng, width, repeats):
    even = [("2Q", i, i + 1) for i in range(0, width - 1, 2)]
    odd = [("2Q", i, i + 1) for i in range(1, width - 1, 2)]
    out = []
    for _ in range(repeats):
        out += even + odd
    return out


def random_pairs(rng, n, m):
    out = []
    for _ in range(m):
        a, b = sorted(rng.sample(range(n), 2))
        out.append(("2Q", a, b))
    return out


def tree(rng, n):
    out = [("2Q", rng.randrange(v), v) for v in range(1, n)]
    rng.shuffle(out)
    return out


def cycle(rng, n):
    return [("2Q", i, (i + 1) % n) for i in range(n)]


KINDS = ["star", "chain", "ladder", "brickwork", "sparse", "dense", "tree", "cycle"]


def random_program(rng: random.Random, max_qubits: int = 20):
    kind = rng.choice(KINDS)
    n = rng.randint(6, max_qubits)
    if kind == "star":
        return kind, star(rng, n)
    if kind == "chain":
        return kind, chain(rng, n)
    if kind == "ladder":
        return kind, ladder(rng, max(2, min(n // 2, 10)))
    if kind == "brickwork":
        return kind, brickwork(rng, n, rng.randint(1, 4))
    if kind == "sparse":
        return kind, random_pairs(rng, n, rng.randint(n, 2 * n))
    if kind == "dense":
        return kind, random_pairs(rng, n, rng.randint(2 * n, 4 * n))
    if kind == "tree":
        return kind, tree(rng, n)
    return kind, cycle(rng, n)


def test_set(seed: int = 1234, count: int = 60, max_qubits: int = 20):
    rng = random.Random(seed)
    return [random_program(rng, max_qubits) for _ in range(count)]
