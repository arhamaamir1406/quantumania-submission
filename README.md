# Quantumania — QSITE 2026 Computational Track

Placement, routing and scheduling for the Quantum Coalition challenge at
Q-SITE 2026.

**Current result: 91.5 across the six public benchmarks, down from the
provided baseline's 283.5 (−67.7%). All six validate.**

| benchmark | 2Q gates | baseline | ours | swaps | depth | floor | gap |
|---|---|---|---|---|---|---|---|
| `ghz_star` | 7 | 14.0 | 7.5 | 4 | 7 | 3.5 | 4.0 |
| `chain_trotter` | 9 | 15.0 | **4.5** | 0 | 9 | 4.5 | **0.0** |
| `ladder_trotter` | 16 | 35.5 | 11.0 | 6 | 10 | 3.0 | 8.0 |
| `qaoa_random` | 18 | 39.0 | 18.5 | 12 | 13 | 4.0 | 14.5 |
| `dense_random` | 40 | 122.0 | 47.0 | 34 | 26 | 6.0 | 41.0 |
| `vqe_layers` | 45 | 58.0 | **3.0** | 0 | 6 | 3.0 | **0.0** |
| **total** | 135 | **283.5** | **91.5** | | | 24.0 | 67.5 |

`floor` is a provable lower bound: SWAP count ≥ 0, and physical depth ≥ the
logical program's ASAP depth. `chain_trotter` and `vqe_layers` sit exactly on
it, so those two are **provably optimal**.

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
5. **GraphSAGE value function** (optional, see below).
6. **The provided baseline**, kept as a safety net — so the result can never
   be worse than it, and never invalid.

### Learned value function

`qroute/gnn.py` implements a small inductive GraphSAGE net (28.5k params, 3
layers, hidden 64) predicting cost-to-go from a routing state. It replaces the
hand-written lookahead in rollouts and beam search: `f(t) = t.cost + V(t)`.

Node features are per physical qubit — occupancy, depth slack
`max(tau) − tau[p]`, active-gate endpoint flags, distance to the resident
qubit's next partner, urgency of that next gate, degree, and distances to both
active endpoints. Six globals carry gates remaining, current depth, swaps,
active-gate distance, near-term mean distance and fill ratio.

SAGE is chosen over an attention model deliberately: it is **inductive**, so a
net trained on this 20-qubit graph transfers to other topologies unchanged,
and mean aggregation over a degree-3 graph needs nothing more expressive. It
is implemented densely (a row-normalised adjacency matmul), so the only added
dependency is `torch` — no PyTorch Geometric.

> **Status: implemented and smoke-tested, not yet trained.** With random
> weights it produces valid routings at 22.0 on `qaoa_random` against the hand
> heuristic's 18.5, which confirms the plumbing. Training is the next step;
> `qroute/train.py` collects labelled trajectories from beam-search solutions
> and regresses on true cost-to-go. Until a checkpoint exists at
> `models/value_sage.pt`, the portfolio simply skips this strategy.

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
  gnn.py                 GraphSAGE value function
  train.py               trajectory collection + training
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

Training the value net (not yet run):

```bash
python -m qroute.train --programs 250 --width 250 --epochs 60
```

## Still open

- Train the value net and measure it against the hand heuristic.
- Large neighbourhood search — rip out a 4–6 gate window, re-route it
  exhaustively, repeat. This is the main remaining lever on `dense_random`,
  which is 47 of our 91.5.
- Commutation bubbling as a post-pass: a SWAP and an adjacent GATE sharing no
  physical qubit can be transposed freely, moving SWAPs into idle layers.
- Peephole SWAP deletion.
- `ghz_star` currently scores 7.5 under move-set branching but reached 7.0
  under an earlier single-SWAP beam; both should be portfolio members.
- Exhaustive/A\* search on the small benchmarks to certify optimality.
