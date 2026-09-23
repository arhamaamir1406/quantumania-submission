# Quantumania — QSITE 2026 Computational Track

Placement, routing and scheduling for the Quantum Coalition challenge at
Q-SITE 2026.

**Current result: 67.5 across the six public benchmarks, down from the
provided baseline's 283.5 (−76.2%). All six validate, and four are provably
optimal** (`ghz_star`, `chain_trotter`, `ladder_trotter`, `vqe_layers`).

| benchmark | 2Q gates | baseline | ours | swaps | depth | floor | proven optimal |
|---|---|---|---|---|---|---|---|
| `ghz_star` | 7 | 14.0 | **6.5** | 2 | 9 | 6.5 | yes — coupled hub bound |
| `chain_trotter` | 9 | 15.0 | **4.5** | 0 | 9 | 4.5 | yes — embeds |
| `ladder_trotter` | 16 | 35.5 | **6.5** | 3 | 7 | 4.0 | yes — `qroute.certify`, 11 s |
| `qaoa_random` | 18 | 39.0 | 11.5 | 7 | 9 | 5.0 | no ≤ 11.0 up to depth 11; none found at 12–14 |
| `dense_random` | 40 | 122.0 | 35.5 | 24 | 23 | 10.0 | — |
| `vqe_layers` | 45 | 58.0 | **3.0** | 0 | 6 | 3.0 | yes — embeds |
| **total** | 135 | **283.5** | **67.5** | 36 | | 33.0 | |

Measured with the default 60 s budget per benchmark on 24 cores.
`dense_random` varies between runs (35.5–39.0 over 16 runs, 35.5 four times,
median 37.5) — the placement search is multi-process, so the same seed
explores a different path each run; the other five return the same score
every run. `floor` is the analytical bound from `qroute/bounds.py`;
the last column adds the computational proofs from `qroute/certify.py`.

`floor` is a provable lower bound, computed by `qroute/bounds.py` — see
*Lower bounds* below. `chain_trotter` and `vqe_layers` sit exactly on it, so
those two are **provably optimal**.

### Against Qiskit's SABRE

The organisers publish no target score, only the deliberately weak baseline
above, so SABRE (Qiskit 2.5.2, `SabreLayout` doing both layout and routing) is
the external reference point. 2000 seeds per benchmark, 20 layout trials and
20 swap trials each, ~3 s per benchmark, scored by the organisers' scorer.

| benchmark | SABRE, valid | SABRE, order rule waived | ours |
|---|---|---|---|
| `ghz_star` | 6.5 | 6.5 | 6.5 |
| `chain_trotter` | 4.5 | 4.5 | 4.5 |
| `ladder_trotter` | 6.5 | 6.5 | 6.5 |
| `qaoa_random` | 12.0 | 11.0 | **11.5** |
| `dense_random` | 43.5 | 36.0 | **35.5** |
| `vqe_layers` | 3.0 | 3.0 | 3.0 |
| **total** | **76.0** | 67.5 | **67.5** |

With the pruning policy on, `dense_random` now lands below SABRE's
order-relaxed 36.0 *while obeying the order rule*.

SABRE routes the circuit's dependency DAG, which is *weaker* than this
challenge's rule: the scorer requires the exact program order, including
between gates that share no qubits (`translated != program` is a hard fail).
Most SABRE runs therefore cannot be submitted at all — on `dense_random` only
2 of 2000 were valid, which is why its valid column is 43.5. The middle column
is what SABRE scores if that rule is dropped; it is not a legal submission.

That middle column is the only place anything beats us, and it motivated two
experiments — SABRE seeding and pre-positioning moves — both recorded below.

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

### Wide move set: pre-positioning SWAPs

`search.py` also offers a wider move set (`lookahead=N`, used by the policy
stage and off elsewhere): each successor may be preceded by one **pre-positioning SWAP** — one that shortens
any of the next `N` gates and touches neither of the active gate's qubits.
Chained across rounds, these start moving qubits for gates that are still
several rounds away, which the narrow move set (shortest paths for the active
gate only) can never do.

It multiplies successors per parent by 5-6x (`dense_random`: 114 -> 692), and
**under the hand-written heuristic it loses** — that heuristic reads a
pre-move as pure progress and lets those states crowd the beam
(`dense_random`, width 64, same placements: 58.2 narrow vs 63-65 wide,
pruned or not). It only pays with a learned policy doing both jobs, below.

### Portfolio

Every strategy runs, self-scores, and the best valid candidate wins:

1. **Zero-SWAP embedding** — VF2 subgraph monomorphism of the interaction
   graph into the hardware graph. Two O(1) rejections first (more edges, or
   higher max degree, than the hardware) so dense programs fail instantly.
   This alone solves `chain_trotter` and `vqe_layers` optimally. Any stage
   that produces a candidate on the provable floor ends the solve at once.
2. **Placement search** (`qroute/placement_search.py`) — the main source of
   score; see below.
3. **Polish** — the 12 best distinct placements are re-routed under 216
   beam settings (width × paths per gate × heuristic weight). The beam is
   non-monotone in all three, so a different setting often routes the same
   placement more cheaply; this replaced a single width-1200 beam that never
   beat the placement search.
4. **Learned value function** (`RoutingNet`) driving both beam search and a
   greedy rollout, when a checkpoint is present — see below.
5. **Exact layered model** (`qroute/exact.py`, needs `ortools`) — on
   instances with ≤ 24 two-qubit gates, CP-SAT solves placement + routing in
   a full-freedom move space (any SWAP on any edge in any layer), seeded with
   the incumbent, for the back half of the budget. It finds trade-offs the
   move-set search cannot express: `qaoa_random` went from 6 SWAPs/depth 12
   (12.0) to 7 SWAPs/depth 9 (11.5). See *Exact model* below.
6. **The provided baseline**, kept as a safety net — so the result can never
   be worse than it, and never invalid.

### Placement search

Routing from a *fixed* placement saturates quickly. Beam widths from 50 to
1200 return the same score; re-routing the tail of the best solution from
random cut points with randomised beam settings (~730k re-routes) found no
improvement on any benchmark; allowing paths one hop longer than shortest
found none either. The starting placement is what moves the score, and good
placements are rare — only 2–15% of jittered constructive placements beat
the old portfolio's result.

So the solver spends most of its budget searching placements, using a narrow
width-16 beam (1–60 ms) as the fitness function:

1. **Sampling** — a few hundred constructive placements at a spread of jitter
   levels, each scored by the narrow beam. Purely random placements were
   tried and are far worse (e.g. 23.5 vs 13.0 on `qaoa_random`).
2. **Iterated local search** from the three best distinct placements. The
   neighbourhood swaps two logicals, or moves one to a free physical qubit,
   within 2 hops of where it sits (~9 moves per logical on this degree-3
   graph). First-improvement over shuffled batches sized to the worker
   count; on a local optimum, perturb the incumbent with 2–3 random swaps.

Beam width is non-monotone in quality here: at a fixed time budget, more
placements at width 16 beat fewer at widths (16, 48) on every worker count
tried. Evaluation runs in a fork-based process pool, falling back to serial
where fork is unavailable or only one core exists; `beam_search` takes a
deadline so the later stages stay inside the budget.

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

Dataset size is requested in **states**, not programs (`--target-states`,
default 300k): yield per program swings by an order of magnitude across the
families, since a 5-gate chain and an 80-gate dense instance are drawn with
equal probability. At the defaults that is ~10.5k programs, of which

| | |
|---|---|
| states / program | 28.4 |
| in ranking groups | 60% (~29k groups, avg 3.7 members) |
| trajectory states | 40% |
| collection | ~38 min on 8 cores, ~19 min on 16 |
| resident on GPU | ~700 MB (fp16) + ~290 MB model and optimiser |

Programs are synthetic (`qroute/generate.py`): eight families — star, chain,
ladder, brickwork, sparse, dense, tree, cycle — sampled uniformly over 6–20
logical qubits, on the one 20-qubit hardware graph. The six public benchmarks
are **not** among them; the families mirror their shapes, plus two the
benchmarks do not contain. Labels are the teacher's cost-to-go, so they are
upper bounds rather than optima: the net can beat the teacher by ordering
better under a wider beam, but the value *scale* is anchored to beam quality.

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

> **Status: trained, and it does not move the score.** Two independent
> end-to-end runs (300,529 states each; `base` 18M and a distilled `small`
> 2.2M, both shipped under `models/`) left the old portfolio's total at
> exactly 91.5, and with placement search the net wins no instance outright.
> Best validation regret was 0.2505 (`base`) and 0.2514 (`small`), reached
> around epoch 15; after that validation regret climbs to ~0.30 while
> training loss keeps falling, i.e. it overfits labels it cannot beat.
>
> The reason is structural, not architectural. With the move-set search
> space, a value function only orders candidates, and on three of the four
> routed benchmarks a *random* value function reaches the same score as the
> trained net (on `dense_random`: zero 62.0, random 63.0, hand heuristic
> 53.0, net 50.0 from the same placements). The 2.2M and 18M nets score
> identically. Labels are the teacher beam's own cost-to-go, which caps what
> the net can learn. A larger net, an ensemble, or a different architecture
> (GAT etc.) would not change this; the placement is the lever.

### Pruning policy

A second, separately trained checkpoint (`models/policy_small_r2.pt`, `small`
preset, 2.2M parameters) drives the wide move set; round 1's
`models/policy_small.pt` is kept for comparison. Same architecture and
the same three losses; what differs is the data and the job it does in search:

* **Data** — `collect.py --lookahead N` builds sibling groups over the *wide*
  successor list, half ordinary moves and half pre-positioning ones, so the
  net sees both under one label. 1.2M labelled states from 20,291 programs
  (94,388 ranking groups) in 14 min on 22 cores.
* **Job** — `beam_search(prune=m, prune_fn=..., lookahead=N)` scores every
  successor of the round in one batched call and keeps each parent's best `m`
  extras, *plus* all of its ordinary moves, so pruning can only add candidates
  to the narrow search, never remove them. The same net then ranks the beam.

From fixed placements it is the first thing that makes pre-positioning pay,
and it also edges out the 18M value net at a ninth of the size (round-1
checkpoint, held-out decision accuracy 0.835, regret 0.377):

| mean / best over 10 placements | heuristic | 18M value net | policy, narrow | policy, wide + prune |
|---|---|---|---|---|
| `dense_random` (w64) | 58.25 / 51.0 | 51.50 / 48.5 | 51.50 / 47.0 | **49.45 / 46.0** |
| `qaoa_random` (w64) | 21.95 / 18.5 | 21.65 / 18.5 | 21.65 / 18.5 | **20.95 / 17.5** |
| `ladder_trotter` (w64) | 18.00 / 15.0 | 17.80 / 15.0 | 17.75 / 15.0 | **17.15 / 14.5** |

> **Status: on by default (`use_policy=True`), after a second training
> round.** Round 1 labelled every sibling with a *narrow* beam, so the net
> learned what a pre-move is worth when only ordinary moves follow it — that
> version tied at the 60 s budget (1 win, 6 ties, 0 losses). Round 2
> re-collected with `--wide-labels`: the policy's own pruned wide search
> computes the labels, so pre-moves are valued by what further pre-moves can
> make of them. 143,835 new states (12,867 groups) on 8 GPU workers in 45 min,
> trained on those plus round 1 (1.34M total). Validation regret 0.377 ->
> **0.326**, decision accuracy 0.838.
>
> That version moves the score. A/B at 60 s on `dense_random`, 6 seeds:
> **4 wins, 1 tie, 1 loss** (39.5/**37.5**, 37.5/37.5, 39.0/**37.5**,
> 37.5/**35.0**, 39.5/**37.5**, **37.0**/37.5), and the policy stage itself
> produced the winning candidate in 4 of the 6. Over 16 further runs the new
> best is **35.5** (24 swaps, depth 23; four times, median 37.5) against a
> previous best of 36.5 that needed 10 minutes offline. `qaoa_random`,
> `ladder_trotter` and the three optimal benchmarks are unchanged.
>
> Runs are not reproducible seed-for-seed: the placement search is
> multi-process, so the same seed takes a different path each run. Every
> score quoted here was validated by the organisers' scorer.
>
> Without `torch`, or without the checkpoint, the stage is skipped silently
> and the total returns to ~69.0.

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
- **Coupled hub bound.** For a logical `v` with `g(v)` gates and `d(v)`
  partners, let `c` be the SWAPs that move `v`. Everything touching `v` is
  serialised, so `depth >= g(v) + c`; the `c` moves add at most `(D-1)c`
  partners and every other SWAP at most one. Minimising
  `swaps + 0.5*depth` over `c` couples the two terms, which the bounds above
  never do. On `ghz_star` (hub with 7 partners and 7 gates) it gives
  `min over c of 3.5 + 1.5c + max(0, 4 - 2c)` = **6.5** at `c = 2` — exactly
  our result, so `ghz_star` is provably optimal.
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

Where the headroom actually is: 37 of our 69.5 is SWAP count, and 27 of the
36.5 gap is `dense_random` alone. The floors are loose there, so not all of
that gap is achievable.

## Exact model

`qroute/exact.py` is an OLSQ-style layered model: `x[q,p,t]` (logical `q` on
physical `p` at layer `t`), `s[e,t]` (SWAP on edge `e`), `y[i,e,t]` (gate `i`
on edge `e`), with disjointness per layer and SWAP-driven mapping
transitions; objective `2·swaps + depth`.

- **Relaxed ordering** — only gates sharing a logical qubit are ordered. The
  ASAP layering of any valid routed program is feasible, so the optimum is a
  **lower bound** on the true optimum. `qroute/certify.py` uses it in depth
  slices (for each depth a better solution could have, prove no layout
  reaches a lower score). `python -m qroute.certify ladder_trotter 6.5`
  proves `ladder_trotter` optimal in ~11 s.
- **Emission** — a relaxed solution becomes a routed program by topological
  sort over per-physical-qubit layer order plus program order. If that has
  a cycle, it contains a qubit chain from a later gate back to an earlier
  one; the solver adds `layer(i) <= layer(j)` for exactly those pairs and
  re-solves (lazy ordering). A fully strict order (all gates monotone) is
  too restrictive — it is infeasible for `qaoa_random` even at 11.5.
- **Scale** — whole-instance solves are fast up to ~20 gates. On
  `dense_random` (40 gates) CP-SAT does not improve the incumbent in 8
  minutes. A windowed variant (`qroute/window.py`: solve 12 gates exactly
  with the mapping fixed, re-route the rest by beam) did improve a 39.5
  solution to 38.5, but at a 60 s budget it loses to spending that time on
  placement search, so it is opt-in (`use_window=True`). Given 10 minutes
  offline it took a 37.0 solution to 36.5; at a 300 s budget, window on and
  off both score 37.5 (two seeds), no better than 60 s.

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
  placement_search.py    sampled placements + iterated local search
  portfolio.py           run-everything-take-the-min solver
  gnn.py                 RoutingNet: multi-stream graph transformer value fn
  features.py            state -> node / edge / gate-token tensors
  collect.py             expert trajectories + sibling groups (multiprocess)
  train.py               ranking + value + decomposition training
  bounds.py              provable per-instance lower bounds
  exact.py               CP-SAT layered model: solver + relaxation
  certify.py             optimality proofs by depth-sliced relaxation
  window.py              window LNS (exact windows + beam), opt-in
  sabre_seed.py          Qiskit SABRE initial layouts as seeds, opt-in
  pipeline.py            one-command collect -> train -> distil -> score
setup.sh                 venv + CUDA torch for a fresh Linux/NVIDIA box
  generate.py            random program generator / held-out test set
  eval.py                benchmark harness
starter_kit/             organisers' code, vendored unmodified
```

## Running it

### Scoring

Only `networkx` and `numpy` are needed to run the solver — `torch` is optional,
and the portfolio skips the learned strategy when it is absent.

```bash
pip install -r requirements.txt

python -m qroute.eval                      # benchmark table
python -m qroute.eval -v --only dense_random
python submission.py                       # scores via the organisers' scorer
python -m qroute.bounds                    # provable lower bounds
```

### Training

One command builds the dataset and trains everything:

```bash
python -m qroute.pipeline
```

It collects 300k labelled states across every core, caches them to
`runs/data.npz`, trains the 18M-parameter `base` net, distils a `small`
student from it, and scores the six benchmarks through the real portfolio. The
dataset cache is checked first, so an interrupted run resumes without paying
for collection twice. `--dagger N` adds rounds on the net's own state
distribution; each one re-collects, so budget the collection time again.

#### On a fresh Linux box with an NVIDIA GPU

```bash
git clone https://github.com/arhamaamir1406/quantumania-submission
cd quantumania-submission
./setup.sh
source .venv/bin/activate
mkdir -p runs && python -m qroute.pipeline 2>&1 | tee runs/train.log
```

`setup.sh` was written against Arch Linux with an RTX 5090 and handles the two
things that break a default install there:

- **Blackwell needs CUDA 12.8+.** An RTX 50xx is compute capability `sm_120`,
  and only cu128 wheels carry `sm_120` kernels. An older wheel still reports
  `torch.cuda.is_available() == True` and then dies on the first matmul with
  *no kernel image is available for execution on the device*. `pipeline.py`
  therefore runs a real kernel at startup and refuses to begin on a wheel that
  cannot execute one — a 20-minute collection is not a good time to discover
  this. Override the channel with `CUDA_CHANNEL=cu129 ./setup.sh`.
- **Arch tracks Python ahead of the PyTorch wheel index.** The script looks for
  an interpreter in 3.10–3.13 and tells you what to install if there is none,
  rather than letting pip resolve nothing or start a source build.

Blackwell also needs driver 570 or newer (`nvidia` / `nvidia-open` on Arch);
the script reports what `nvidia-smi` sees.

#### Cost

Collection is CPU-bound beam search, so cores set that stage's wall clock and
the GPU is idle for it:

| | |
|---|---|
| 300k states | ~38 min on 8 cores, ~19 min on 16 |
| dataset on disk | ~90 MB compressed (`runs/data.npz`) |
| dataset on GPU | ~700 MB fp16, plus ~290 MB model and optimiser |

Individual stages, if you want them separately:

```bash
python -m qroute.train --preset base --target-states 300000 --benchmark
python -m qroute.train --preset base --data runs/data.npz --epochs 90    # retrain, no re-collection
python -m qroute.train --preset base --dagger 2 --resume models/value_base.pt
python -m qroute.train --preset small --distill-from models/value_base.pt --data runs/data.npz
```

## Still open

- **`qaoa_random` optimality.** No solution ≤ 11.0 exists with depth ≤ 11
  (proved); the relaxed model admits depth-12 layouts ≤ 11.0, but none has
  yet been made emittable (20 min of lazy-ordering search at depth 12, and
  150 s rounds at 13–14, found none). Closing it needs longer proof runs.
- **`dense_random` sat at ~37 under every non-learned method tried**:
  placement search + polish (60–300 s), whole-instance CP-SAT, window LNS,
  tail re-routing, longer paths, reverse-traversal seeding, per-core chains,
  SABRE seeding. The policy-pruned wide search broke that plateau (35.5), so
  the move set, not the placement, was the binding constraint there. Its
  analytical floor (10.0) is still far too loose to say what remains.
- **A third policy round.** Round 2 (`--wide-labels`) took `dense_random`
  from ~37.5 to 35.5; its labels are now the *round-1* policy's searches, so
  the same trick can be iterated with round 2 as the labeller. Whether that
  keeps paying is untested. Validation regret still bottoms out around epoch
  15-20 and then drifts up, so more epochs are not the lever.
- **SABRE's relaxed 36.0 on `dense_random`** is not reachable under the order
  rule: its solutions run a later gate before a SWAP an earlier gate needs, so
  no reordering of those ops is legal. Seeding our search with SABRE's initial
  layouts (`use_sabre=True`) ties — 4 ties, 1 win, 1 loss over 6 runs at 60 s.
- Stretch goals (decomposition, 1Q optimisation) are not attempted.
