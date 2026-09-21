# Quantumania — QSITE 2026 Computational Track

Placement, routing and scheduling for the Quantum Coalition challenge at
Q-SITE 2026.

**Current result: 91.5 across the six public benchmarks, down from the
provided baseline's 283.5 (−67.7%). All six validate.**

| benchmark | 2Q gates | baseline | ours | swaps | depth | swaps ≥ | floor | gap |
|---|---|---|---|---|---|---|---|---|
| `ghz_star` | 7 | 14.0 | 7.5 | 4 | 7 | 2 | 5.5 | 2.0 |
| `chain_trotter` | 9 | 15.0 | **4.5** | 0 | 9 | 0 | 4.5 | **0.0** |
| `ladder_trotter` | 16 | 35.5 | 11.0 | 6 | 10 | 1 | 4.0 | 7.0 |
| `qaoa_random` | 18 | 39.0 | 18.5 | 12 | 13 | 1 | 5.0 | 13.5 |
| `dense_random` | 40 | 122.0 | 47.0 | 34 | 26 | 4 | 10.0 | 37.0 |
| `vqe_layers` | 45 | 58.0 | **3.0** | 0 | 6 | 0 | 3.0 | **0.0** |
| **total** | 135 | **283.5** | **91.5** | 56 | | **8** | 32.0 | 59.5 |

`floor` is a provable lower bound, computed by `qroute/bounds.py` — see
*Lower bounds* below. `chain_trotter` and `vqe_layers` sit exactly on it, so
those two are **provably optimal**.

## Approach

### The objective is an MDP return

Routing is modelled as a Markov decision process whose state carries the
*scheduler's clock* — `tau`, the per-physical-qubit last-used layer — alongside
the mapping and the gate pointer. Step costs are

```
SWAP  ->  1 + 0.5 * max(0, new_depth - old_depth)
GATE  ->      0.5 * max(0, new_depth - old_depth)
```

which sum over an episode to exactly `swaps + 0.5 * depth`, the competition
score. No proxy objective, no reward shaping.

This is the point of the design. Distance-based heuristics like SABRE cannot
see the depth term at all, because their state has no notion of which layer
anything landed in. Putting `tau` in the state makes depth a first-class
quantity the search can optimise directly.

### Branching on move sets, not single SWAPs

With the front layer pinned to a single gate (see *Correctness*, below), a
one-SWAP-at-a-time greedy has no progress guarantee — it wanders between
states that each look locally reasonable and never executes the active gate.
We measured this: it hit the step cap on most instances.

Instead we branch on **move sets**: for the active gate, commit to a whole
shortest path *and* a meeting point along it. Every successor executes the
gate, so the search terminates in exactly one round per gate. It also exposes
the meet-in-the-middle choice directly — splitting the walk across both
endpoints puts the SWAPs on disjoint qubits, so the ASAP scheduler packs them
into shared layers and the depth cost of a distance-`d` gate drops from `d−1`
to about `⌈(d−1)/2⌉`.

### Portfolio

Every strategy runs, self-scores, and the best valid candidate wins:

1. **Zero-SWAP embedding** — VF2 subgraph monomorphism of the interaction
   graph into the hardware graph. Two O(1) rejections first (more edges, or
   higher max degree, than the hardware) so dense programs fail instantly.
   This alone solves `chain_trotter` and `vqe_layers` optimally.
2. **Constructive placement** — grow outward from the busiest logical qubit,
   placing each next qubit where it minimises weighted distance to its
   already-placed neighbours.
3. **Greedy rollouts** from a spread of placements (constructive, jittered
   constructive, random), with noise for diversity.
4. **Beam search** over move sets from the most promising placements, with a
   `(gate, mapping)` transposition table.
5. **Learned value function** (`RoutingNet`) driving both beam search and a
   greedy rollout, when a checkpoint is present — see below.
6. **The provided baseline**, kept as a safety net — so the result can never
   be worse than it, and never invalid.

### Learned value function

`qroute/gnn.py` implements `RoutingNet`, a multi-stream graph transformer that
predicts cost-to-go from a routing state. It replaces the hand-written
lookahead in rollouts and beam search: `f(t) = t.cost + V(t)`.

**The objective is structural, not a loss weighting.** The network has two
heads, and the value is their fixed combination:

```
swaps_to_go ~ head_s(z) >= 0
depth_to_go ~ head_d(z) >= 0
V           = swaps_to_go + 0.5 * depth_to_go
```

There is no path by which the net can emit a value that is not
`swaps + 0.5 * depth` of something. Adding capacity therefore cannot drift the
model off the competition metric; it can only make the two terms sharper.

Both heads emit five **monotone quantiles** (cumulative softplus) rather than a
point estimate, so beam search can be run optimistically — ranking on a low
quantile keeps candidates whose downside is good, which matters once the top of
the beam is crowded with states of near-identical mean value.

**Three token streams** (`qroute/features.py`), fused by repeated attention:

| stream | tokens | carries |
|---|---|---|
| node | one per physical qubit (20) | occupancy, the scheduler's clock `tau` and its slack, active-gate geometry, what the resident logical wants next |
| edge | one per hardware edge (23) | what applying *this* SWAP does to the active gate and to the gate window — candidate SWAPs are first-class tokens |
| gate | one per upcoming program op (16) | the program's future: separation, mutual contention, depth slack |

Each block runs gated message passing along hardware edges (locality as a hard
structural prior), then full self-attention **biased by shortest-path distance
buckets**, Graphormer-style, so a qubit can attend directly to one four hops
away without four rounds of propagation. Gate tokens self-attend over the
window and cross-attend to the nodes, bound to geometry by gathering node
embeddings at the two physical qubits their logicals currently occupy. Globals
condition every block through FiLM. Laplacian eigenvector positional encodings
give the net hardware structure without node identities; their signs are
flipped at random during training so it cannot memorise them.

| preset | d_model | blocks | parameters |
|---|---|---|---|
| `tiny` | 64 | 3 | 0.38M |
| `small` | 128 | 4 | 2.2M |
| `base` | 256 | 8 | **18.0M** |
| `large` | 384 | 12 | 59.4M |

The default is `base`, which trains comfortably on a single 5090. The previous
GraphSAGE net was 28.5k parameters; this is ~630x larger, and the capacity is
spent on the two things a routing value function actually needs and the old one
had no way to represent — candidate SWAPs as tokens, and the future gate
sequence as a sequence.

### Training

`qroute/collect.py` generates data, `qroute/train.py` fits. Three losses, in
descending order of how much they matter to the score:

1. **Ranking** — over *sibling groups* (successors of the same state), match
   `softmax(-f)` to `softmax(-true final cost)`. Search only ever uses `V` to
   **order** candidates, so this is the loss that is literally the objective. A
   net can have excellent MAE and still order siblings wrongly, because sibling
   values differ by a fraction of a SWAP while the value itself is tens.
2. **Value regression** — pinball loss on the value quantiles. Anchors the
   scale that ranking alone leaves free, and keeps `f` comparable across beam
   rounds, where successors of *different* parents are compared.
3. **Decomposition** — pinball on the swaps and depth heads separately. It
   tells the net *why* a state is expensive, not just that it is.

Sibling groups are the expensive part and the reason this works: at sampled
decision points the whole move-set successor list is expanded and an
independent beam is run from *each* successor, so every sibling carries its own
cost-to-go. `--dagger N` then re-collects from the current net's own rollouts,
which fixes the standard failure of pure imitation — the expert never visits
the states the student's mistakes lead to.

States are recorded at **gate-round boundaries**, exactly the states
`beam_search` scores, so the training distribution matches the inference one.
Validation is split by program, never by state. The headline validation metric
is not MAE but **decision accuracy** and **regret** on held-out sibling groups:
how often ranking by `f` picks the successor that really leads to the best
final score, and how much score is lost when it does not.

Mechanics: AdamW with cosine decay and warmup, bf16 autocast, gradient
clipping, EMA weights, the dataset resident on-device in fp16. Collection runs
across processes (`--workers`). `--distill-from` trains a small student on a
large teacher, which is how a CPU-affordable checkpoint is produced.

> **Status: implemented and smoke-tested end to end, not yet trained.** All
> three paths (fresh, DAgger, distillation) run; no checkpoint is shipped yet.
> Until one exists under `models/`, the portfolio skips this strategy — and
> with an untrained checkpoint present the benchmark total is unchanged at
> 91.5, because a portfolio member only wins an instance if it self-scores
> better. A regression in the net can cost search time; it can never cost score.

### Inference cost

An 18M-parameter forward pass is ~3 ms/state on CPU and ~30 µs on a GPU, so
the affordable beam width differs by two orders of magnitude. The portfolio
detects the device, picks the widest checkpoint that fits
(`value_large` > `value_base` > `value_small` on GPU, reversed on CPU), scales
the beam width accordingly, deduplicates and memoises states within and across
calls, and keeps every call inside the wall-clock budget.

## Lower bounds

`qroute/bounds.py` proves a floor for each instance. With `D` the hardware max
degree (3) and the interaction graph holding one edge per distinct logical pair
the program makes interact:

- **Degree bound.** Logical `v` must at some point be adjacent to each of its
  `d(v)` distinct partners; it is adjacent to at most `D` at a time, and one
  SWAP adds at most `D-1` to its neighbourhood. So
  `swaps >= ceil((d(v) - D) / (D - 1))`.
- **Edge-count bound.** A SWAP moves two qubits, each acquiring at most `D-1`
  new partners, so it realises at most `2(D-1)` new pairs; the placement
  realises at most `|E_hw|`. So
  `swaps >= ceil((|E_int| - min(|E_int|, |E_hw|)) / (2(D-1)))`.
- **Embeddability.** If the interaction graph is not subgraph-monomorphic to
  the hardware graph, at least one SWAP is required.
- **Depth.** Physical depth is at least the logical program's ASAP depth.

Run `python -m qroute.bounds` for the table.

### On "zero SWAPs"

A total of **0 SWAPs across these six benchmarks is not achievable**, and the
bounds above say so without reference to any algorithm:

- `ghz_star` — logical qubit 0 interacts with 7 distinct partners, but no
  physical qubit has more than 3 neighbours. At least 2 SWAPs.
- `qaoa_random` — max logical degree 5 > 3. At least 1 SWAP.
- `dense_random` — max logical degree 9 > 3, and 37 distinct pairs must be
  realised on a graph with 23 edges. At least 4 SWAPs.
- `ladder_trotter` — degree-feasible, but VF2 proves no subgraph monomorphism
  exists, so no placement makes all 16 pairs adjacent at once. At least 1 SWAP.

Total: **at least 8 SWAPs are mathematically required.** Zero SWAPs *is*
achievable on `chain_trotter` and `vqe_layers`, where the interaction graph
embeds — and we already achieve it on both, optimally. So a reported "0 SWAPs"
can only refer to those instances, not to the set.

Where the headroom actually is: 56 of our 91.5 is SWAP count, and 37 of the
59.5 gap is `dense_random` alone.

## Correctness

The scorer strips inserted SWAPs and requires the remainder to equal the
original program **in exact order**. Two consequences drove the design:

- **No gate reordering.** A faithful SABRE port executes whatever is in the
  DAG front layer and therefore reorders commuting gates, scoring ∞. Our front
  layer is always the single next gate; parallelism comes from the ASAP
  scheduler, never from us reordering.
- **Gate argument order is load-bearing.** For program op `("2Q", i, j)` the
  emitted pair must have the qubit holding logical `i` first. The graph is
  undirected so `has_edge` passes either way, but tuple comparison does not.

Both are handled structurally by the IR: a solution is `(placement, ops)` with
`ops` holding `("SWAP", p, q)` and `("GATE", k)`, where `k` indexes the
original program. Physical arguments are never stored — they are derived in
`materialize()`. Every edit (delete, move, insert a SWAP) is therefore safe,
and the argument-order trap cannot occur.

## Layout

```
submission.py            the required solve(program, hardware_graph)
qroute/
  ir.py                  solution IR, materialize, evaluate
  mdp.py                 depth-aware routing MDP
  search.py              move-set branching, rollouts, beam search
  placement.py           VF2 embedding, constructive, random
  portfolio.py           run-everything-take-the-min solver
  gnn.py                 RoutingNet: multi-stream graph transformer value fn
  features.py            state -> node / edge / gate-token tensors
  collect.py             expert trajectories + sibling groups (multiprocess)
  train.py               ranking + value + decomposition training
  bounds.py              provable per-instance lower bounds
  generate.py            random program generator / held-out test set
  eval.py                benchmark harness
starter_kit/             organisers' code, vendored unmodified
```

## Running it

```bash
pip install -r requirements.txt

python -m qroute.eval                      # benchmark table
python -m qroute.eval -v --only dense_random
python submission.py                       # scores via the organisers' scorer
```

```bash
python -m qroute.bounds                    # provable lower bounds
```

Training the value net (not yet run):

```bash
# full run on one 5090
python -m qroute.train --preset base --programs 4000 --workers 12 --benchmark

# add DAgger rounds on the net's own state distribution
python -m qroute.train --preset base --dagger 2 --resume models/value_base.pt

# CPU-affordable student distilled from it
python -m qroute.train --preset small --distill-from models/value_base.pt
```

## Still open

- Train the value net and measure it against the hand heuristic. The code is
  written and smoke-tested end to end; nothing has been fitted yet.
- Placement is the lever on SWAP count, and SWAP count is 56 of our 91.5. The
  zero-SWAP check is currently all-or-nothing (VF2 monomorphism); a
  *maximum-coverage* placement — maximise the weighted interaction edges that
  are adjacent at time zero — should help every instance that fails to embed.
- Large neighbourhood search — rip out a 4–6 gate window, re-route it
  exhaustively, repeat. This is the main remaining lever on `dense_random`,
  which is 47 of our 91.5.
- Commutation bubbling as a post-pass: a SWAP and an adjacent GATE sharing no
  physical qubit can be transposed freely, moving SWAPs into idle layers.
- Peephole SWAP deletion.
- `ghz_star` currently scores 7.5 under move-set branching but reached 7.0
  under an earlier single-SWAP beam; both should be portfolio members.
- Exhaustive/A\* search on the small benchmarks to certify optimality.
