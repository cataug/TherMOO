"""Foreground CLI with bounded work submission and completion-based restart."""
from .runtime import ROOT
from .plan import Job, jobs, PROBLEMS, PANEL, METHODS, BLOCKS
from .engine import trial, atomic_json, experiment_fingerprint
from dataclasses import asdict, replace
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import argparse
import itertools
import json
import multiprocessing
import os
import time


def execute(all_jobs, workers, output):
    all_jobs = list(all_jobs)
    start = time.monotonic()
    counts = Counter()
    print(f"[QUEUE] {len(all_jobs)} trials, workers={workers}, output={output}, fingerprint={experiment_fingerprint()}", flush=True)
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
    iterator = iter(all_jobs)
    pending = {}
    def refill():
        while len(pending) < 2*workers:
            item = next(iterator, None)
            if item is None:
                break
            pending[pool.submit(trial, item, output, False)] = item
    refill()
    heartbeat = time.monotonic()
    try:
        while pending:
            done, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    result = {"status":"failed","error":repr(error)}
                counts[result["status"]] += 1
                completed = sum(counts.values())
                text = f"[{completed}/{len(all_jobs)}] {job.block} {job.problem} M={job.objectives} D={job.variables} {job.label} seed={job.seed}: {result['status']}"
                if "seconds" in result:
                    text += f" {result['seconds']:.2f}s"
                print(text, flush=True)
                if result["status"] == "failed":
                    print(result["error"], flush=True)
                    # Stop submitting new expensive work after an implementation failure.
                    raise RuntimeError("Trial failed; error.json contains the traceback. Completed trials are retained.")
            refill()
            if time.monotonic()-heartbeat > 25:
                print(f"[LIVE] completed={sum(counts.values())}/{len(all_jobs)} active/queued={len(pending)} elapsed={(time.monotonic()-start)/60:.1f} min", flush=True)
                heartbeat = time.monotonic()
    except BaseException:
        for future in pending:
            future.cancel()
        # Ctrl-C must not leave a queue of thousands of experiments running.
        for process in list((pool._processes or {}).values()):
            if process.is_alive():
                process.terminate()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    print(f"[DONE] {dict(counts)}, elapsed={(time.monotonic()-start)/60:.2f} min", flush=True)


def main():
    parser = argparse.ArgumentParser(prog="thermoo")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("verify")
    sub.add_parser("plan")
    for command in ("smoke", "pilot", "run"):
        p = sub.add_parser(command)
        p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
        p.add_argument("--max-jobs", type=int)
        p.add_argument("--problems", nargs="+", choices=PROBLEMS)
        if command == "run":
            p.add_argument("--blocks", nargs="+", choices=BLOCKS, default=BLOCKS)
            p.add_argument("--seeds", nargs="+", type=int, default=list(range(30)))
        if command == "pilot":
            p.add_argument("--budget", type=int, default=8000)
    p = sub.add_parser("analyze")
    p.add_argument("--output", default="full")
    p.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        from .native import build
        from .problems import prepare_references
        build()
        prepare_references()
    elif args.command == "verify":
        from .verify import verify
        verify()
    elif args.command == "plan":
        matrix = list(jobs())
        for job in matrix:
            assert job.budget % job.population == 0
        folder = ROOT / "configs"
        folder.mkdir(exist_ok=True)
        path = folder / "experiment_plan.jsonl"
        with path.open("w") as stream:
            for job in matrix:
                stream.write(json.dumps(asdict(job),sort_keys=True)+"\n")
        summary = dict(Counter(j.block for j in matrix))
        atomic_json(ROOT/"reports/experiment_plan_summary.json",summary)
        print(json.dumps(summary,indent=2))
        print(f"[PLAN] Total {len(matrix)} trials: {path}")
    elif args.command == "analyze":
        from .analysis import analyze
        analyze(args.output, args.plots)
    else:
        if args.workers < 1:
            parser.error("workers must be positive")
        if args.max_jobs is not None and args.max_jobs < 1:
            parser.error("max-jobs must be positive")
        if args.command == "pilot" and (args.budget < 800 or args.budget % 400):
            parser.error("pilot budget must be a multiple of 400 and at least 800")
        if args.command == "run":
            matrix = jobs(args.blocks, args.seeds)
            output = "full"
        elif args.command == "pilot":
            selected = args.problems or PANEL
            matrix = [replace(Job(p,m,0), block="pilot", budget=args.budget) for p in selected for m in METHODS]
            if args.problems is None:
                matrix += [replace(j,block="pilot") for j in jobs(["many"],seeds=[0])]
                matrix = [replace(j,budget=args.budget) for j in matrix]
            output = "pilot"
        else:
            matrix = [replace(j,block="smoke",population=40,archive=40,budget=400,interval=2) for j in jobs(["main"],seeds=[0])]
            # Exercise every additional mechanism and every solver beyond the core table.
            extras = []
            for block in BLOCKS[1:]:
                candidates = list(jobs([block],seeds=[0]))
                if block == "ablation":
                    candidates = [j for j in candidates if j.problem == "DTLZ7"]
                elif block == "homogeneous":
                    candidates = [j for j in candidates if j.problem == "DTLZ7" and j.method == "FULL"]
                elif block == "islands":
                    candidates = [j for j in candidates if j.problem == "DTLZ7" and j.method == "FULL"]
                elif block == "many":
                    candidates = [j for j in candidates if j.problem == "WFG9"]
                elif block == "dimensions":
                    candidates = [j for j in candidates if j.problem == "ZCAT20" and j.method == "FULL"]
                else:
                    candidates = [j for j in candidates if j.problem == "DTLZ7"]
                extras += candidates
            for j in extras:
                size=max(40,5*j.islands)
                matrix.append(replace(j,block="smoke",population=size,archive=size,budget=(max(10,2*j.interval)+1)*size))
            output = "smoke"
        if args.problems:
            matrix = (j for j in matrix if j.problem in args.problems)
        if args.max_jobs:
            matrix = itertools.islice(matrix,args.max_jobs)
        execute(matrix,args.workers,output)


if __name__ == "__main__":
    main()
