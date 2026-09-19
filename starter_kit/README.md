# Computational Starter Kit

This folder contains the starter-kit code used by `starter.ipynb`.

- `hardware.py`: the 20-qubit heavy-hex-style teaching graph
- `benchmarks.py`: toy benchmark programs used in the notebook and starter kit
- `scorer.py`: correctness checks, ordered layer scheduling, and core scoring
- `baseline_routing.py`: intentionally weak routing baseline
- `visualize.py`: graph drawing and simple notebook animations

The baseline router is meant to be obviously beatable. It uses a fixed identity-style placement, greedy shortest-path SWAP insertion, and no placement optimization.
