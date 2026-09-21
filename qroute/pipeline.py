"""One command that builds the dataset and trains everything.

    python -m qroute.pipeline

Stages, in order:

1. **Collect** `--target-states` labelled states across every core and cache
   them to `--data`. Skipped if that file already exists, so an interrupted
   run resumes without paying for collection twice.
2. **Train** the `base` network (18M parameters) on them.
3. **DAgger** `--dagger N` extra rounds, each re-collecting from the current
   network's own state distribution. Off by default: every round pays the
   collection cost again.
4. **Distil** a `small` student from the trained teacher, reusing the same
   cached dataset. This is the checkpoint that makes the learned strategy
   usable if the submission is ever scored without a GPU, where an 18M
   forward pass is ~3 ms/state.
5. **Score** the six public benchmarks through the real portfolio.

Every stage is individually skippable, and the checkpoints land in `models/`
where `qroute.portfolio` looks for them.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:d}h{(s % 3600) // 60:02d}m{s % 60:02d}s"


def _run_train(argv: list[str]) -> None:
    """Invoke qroute.train as if from the command line."""
    from . import train
    old = sys.argv
    sys.argv = ["qroute.train", *argv]
    try:
        train.main()
    finally:
        sys.argv = old


CUDA_INSTALL_HINT = """    python -m venv .venv && source .venv/bin/activate
    pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
    pip install -r requirements.txt"""


def cuda_smoke_test() -> tuple[bool, str]:
    """Actually run a kernel on the GPU.

    `torch.cuda.is_available()` only says the driver found a CUDA device. It
    says nothing about whether the installed wheel was *compiled* for that
    device: a Blackwell card (RTX 5090, sm_120) under a wheel built for
    sm_90 and below reports True here and then dies on the first matmul with
    "no kernel image is available for execution on the device". Training
    would abort minutes in, after collection has already been paid for, so
    the check belongs at startup.
    """
    import torch
    try:
        a = torch.randn(64, 64, device="cuda")
        b = (a @ a).sum().item()
        if b != b:                                    # NaN
            return False, "GPU matmul returned NaN"
        return True, "ok"
    except Exception as exc:
        return False, str(exc).strip().splitlines()[0]


def environment_report(require_cuda: bool = False) -> str:
    """Print what we are about to train on, and return the device string."""
    import torch

    cores = os.cpu_count() or 1
    print("=" * 72)
    print(f"torch {torch.__version__}   python {sys.version.split()[0]}   "
          f"{cores} cores   {sys.platform}")

    if not torch.cuda.is_available():
        print("CUDA NOT available -- training would run on CPU.")
        if "+cpu" in torch.__version__:
            print("\n  This is a CPU-only torch build. Install a CUDA build:\n")
            print(CUDA_INSTALL_HINT)
        print("\n  An 18M-parameter model on CPU is impractically slow "
              "(days, not hours).")
        if require_cuda:
            raise SystemExit("aborting: --require-cuda set and CUDA is unavailable")
        print("  Continuing anyway. Ctrl-C now if that was not intended.")
        print("=" * 72)
        return "cpu"

    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    cap = f"sm_{props.major}{props.minor}"
    mem = props.total_memory / 1e9
    built = ""
    try:
        built = " ".join(torch.cuda.get_arch_list())
    except Exception:
        pass
    print(f"CUDA {torch.version.cuda}: {name} ({cap}, {mem:.0f} GB)")
    if built:
        print(f"wheel built for: {built}")

    ok, msg = cuda_smoke_test()
    if not ok:
        print(f"\n  GPU KERNEL TEST FAILED: {msg}")
        if built and f"sm_{props.major}{props.minor}" not in built:
            print(f"  This wheel has no {cap} kernels, which is what your "
                  f"{name} needs.")
        print("\n  Install a build with kernels for this card:\n")
        print(CUDA_INSTALL_HINT)
        print()
        if require_cuda:
            raise SystemExit("aborting: GPU present but unusable")
        print("  Falling back to CPU, which is impractically slow. "
              "Ctrl-C now if that was not intended.")
        print("=" * 72)
        return "cpu"

    print("GPU kernel test: ok")
    print("=" * 72)
    return "cuda"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-states", type=int, default=300_000)
    ap.add_argument("--data", default="runs/data.npz",
                    help="dataset cache; reused if it already exists")
    ap.add_argument("--preset", default="base",
                    help="teacher preset to train (tiny/small/base/large)")
    ap.add_argument("--student", default="small",
                    help="student preset to distil; empty string to skip")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--student-epochs", type=int, default=40)
    ap.add_argument("--steps-per-epoch", type=int, default=600)
    ap.add_argument("--workers", type=int, default=0, help="0 = cpu_count-1")
    ap.add_argument("--dagger", type=int, default=0,
                    help="extra collect+train rounds on the net's own states")
    ap.add_argument("--groups-per-program", type=int, default=6)
    ap.add_argument("--siblings", type=int, default=8)
    ap.add_argument("--teacher-width", type=int, default=400)
    ap.add_argument("--label-width", type=int, default=60)
    ap.add_argument("--collect-cap", type=float, default=14400.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--require-cuda", action="store_true",
                    help="abort instead of falling back to CPU training")
    ap.add_argument("--skip-collect", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-distill", action="store_true")
    ap.add_argument("--skip-benchmark", action="store_true")
    args = ap.parse_args()

    t_start = time.monotonic()
    environment_report(args.require_cuda)

    Path("models").mkdir(exist_ok=True)
    Path(args.data).parent.mkdir(parents=True, exist_ok=True)
    teacher_ckpt = f"models/value_{args.preset}.pt"

    common = ["--workers", str(args.workers),
              "--steps-per-epoch", str(args.steps_per_epoch),
              "--groups-per-program", str(args.groups_per_program),
              "--siblings", str(args.siblings),
              "--teacher-width", str(args.teacher_width),
              "--label-width", str(args.label_width),
              "--collect-cap", str(args.collect_cap),
              "--seed", str(args.seed)]

    # -- stages 1 + 2: collect (cached) and train the teacher ---------------
    if not args.skip_train:
        have = os.path.exists(args.data)
        if have and not args.skip_collect:
            print(f"\n[1/4] dataset {args.data} already exists "
                  f"({os.path.getsize(args.data)/1e6:.0f} MB) -- reusing it")
        argv = ["--preset", args.preset, "--epochs", str(args.epochs),
                "--out", teacher_ckpt, *common]
        if have:
            argv += ["--data", args.data]
        else:
            print(f"\n[1/4] collecting {args.target_states:,} states "
                  f"-> {args.data}")
            argv += ["--target-states", str(args.target_states),
                     "--save-data", args.data]
        if args.dagger:
            argv += ["--dagger", str(args.dagger),
                     "--target-states", str(args.target_states)]
        print(f"[2/4] training '{args.preset}' -> {teacher_ckpt}\n")
        t0 = time.monotonic()
        _run_train(argv)
        print(f"\n[2/4] done in {_hms(time.monotonic() - t0)}")

    # -- stage 3: distil a CPU-affordable student ---------------------------
    if args.student and not args.skip_distill:
        if args.student == args.preset:
            # both would write models/value_<preset>.pt and the student would
            # clobber its own teacher
            print(f"\n[3/4] student preset '{args.student}' equals the "
                  f"teacher's -- skipping distillation")
        elif not os.path.exists(teacher_ckpt):
            print(f"\n[3/4] no teacher at {teacher_ckpt} -- skipping distillation")
        elif not os.path.exists(args.data):
            print(f"\n[3/4] no dataset at {args.data} -- skipping distillation")
        else:
            print(f"\n[3/4] distilling '{args.student}' from {teacher_ckpt}\n")
            t0 = time.monotonic()
            _run_train(["--preset", args.student, "--distill-from", teacher_ckpt,
                        "--data", args.data, "--epochs", str(args.student_epochs),
                        "--out", f"models/value_{args.student}.pt", *common])
            print(f"\n[3/4] done in {_hms(time.monotonic() - t0)}")

    # -- stage 4: score the benchmarks through the real portfolio -----------
    if not args.skip_benchmark:
        print("\n[4/4] scoring the six public benchmarks\n")
        from .eval import main as eval_main
        old = sys.argv
        sys.argv = ["qroute.eval"]
        try:
            eval_main()
        finally:
            sys.argv = old

    print(f"\npipeline finished in {_hms(time.monotonic() - t_start)}")
    for p in sorted(Path("models").glob("*.pt")):
        print(f"  {p}  {p.stat().st_size/1e6:.0f} MB")


if __name__ == "__main__":
    main()
