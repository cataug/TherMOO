"""Budget-checked trials, job-level restart, common archives and action logs."""
from .runtime import ROOT, stable_seed, code_hash
from .plan import Job
from .problems import Benchmark, initial_population, load_reference
from .algorithms import create, evolve, inject
from .control import profile, select_actions, choose_migrants, crowding
from dataclasses import asdict
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from functools import lru_cache
import importlib.metadata

import numpy as np
from scipy.spatial.distance import cdist
from moocore import is_nondominated, hypervolume


def atomic_json(path, value):
    temporary = Path(str(path)+f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def truncate_archive(X, F, cap):
    _, idx = np.unique(F, axis=0, return_index=True)
    X, F = X[idx], F[idx]
    keep = is_nondominated(F)
    X, F = X[keep], F[keep]
    if len(F) > cap:
        choose = np.argsort(-crowding(F), kind="stable")[:cap]
        X, F = X[choose], F[choose]
    return X, F


def indicators(F, pf):
    lo = pf.min(axis=0)
    scale = np.maximum(pf.max(axis=0)-lo, 1e-12)
    f, target = (F-lo)/scale, (pf-lo)/scale
    hv = float(hypervolume(f, ref=np.full(F.shape[1], 1.1)))/(1.1**F.shape[1])
    # IGD+: a reference point is covered if the candidate weakly dominates it.
    total = 0.
    for start in range(0, len(target), 512):
        delta = np.maximum(f[None, :, :]-target[start:start+512, None, :], 0.)
        total += np.sqrt(np.sum(delta*delta, axis=2)).min(axis=1).sum()
    return dict(hv=hv, igd_plus=float(total/len(target)))


@lru_cache(maxsize=1)
def experiment_fingerprint():
    h = hashlib.sha256(code_hash().encode())
    for package in ("numpy", "scipy", "moocore", "autograd"):
        h.update((package+importlib.metadata.version(package)).encode())
    for name in ("reports/download_manifest.json", "build/thermoo/source_hashes.json"):
        path = ROOT / name
        if path.exists():
            # Exclude download timestamp/platform: source commits and actual benchmark bytes matter.
            obj = json.loads(path.read_text())
            if "repositories" in obj:
                obj = {k: v["commit"] for k, v in obj["repositories"].items()}
            h.update(json.dumps(obj, sort_keys=True).encode())
    for file in sorted((ROOT / "data/thermoo_references").glob("*.npy")):
        h.update(file.name.encode())
        h.update(file.read_bytes())
    return h.hexdigest()[:16]


def trial(job, output="full", verbose=False):
    fingerprint = experiment_fingerprint()
    folder = ROOT / "outputs" / output / fingerprint / job.signature()
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "result.json"
    with (folder / "run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "locked", "job": job.signature()}
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result.get("status") == "complete" and (folder/"population.npz").exists():
                return {"status": "cached", "job": job.signature(), "seconds": result["seconds"]}
        started = time.perf_counter()
        try:
            result = _compute(job, folder, started, verbose)
            result.update(status="complete", job=asdict(job), fingerprint=fingerprint, run_id=job.signature())
            atomic_json(result_path, result)
            return {"status": "complete", "job": job.signature(), "seconds": result["seconds"], "hv": result["metrics"]["hv"]}
        except Exception:
            error = traceback.format_exc()
            atomic_json(folder / "error.json", {"job": asdict(job), "error": error})
            return {"status": "failed", "job": job.signature(), "error": error}


def _compute(job, folder, started, verbose):
    if job.budget % job.population or job.population % job.islands:
        raise ValueError("Budget must be divisible by total population, and population by island count")
    problem = Benchmark(job)
    pf = load_reference(job)
    singleton = job.method in ("NSGA2", "MOEAD", "GDE3", "SMS", "NSGA3", "RVEA")
    k = 1 if singleton else job.islands
    n = job.population//k
    if n < 5:
        raise ValueError("At least five individuals per island are required")
    kinds = [job.method] if singleton else [job.composition if job.composition != "mixed" else ["NSGA2", "MOEAD", "GDE3", "SMS"][i % 4] for i in range(k)]
    X0 = initial_population(problem, job.population, job.seed)
    initial_hash = hashlib.sha256(X0.tobytes()).hexdigest()
    local_budget = job.budget//k
    algorithms = [create(kind, problem, X0[i*n:(i+1)*n], stable_seed(job.problem, job.objectives, job.variables, job.seed, i, "algorithm"), local_budget) for i, kind in enumerate(kinds)]
    allF = np.vstack([a.pop.get("F") for a in algorithms])
    online_scale = np.maximum(np.ptp(allF, axis=0), 1e-8)
    online_lower = allF.min(axis=0)-.1*online_scale
    archiveX, archiveF = truncate_archive(np.vstack([a.pop.get("X") for a in algorithms]), allF.copy(), job.archive)
    rng = np.random.default_rng(stable_seed(job.problem, job.seed, "migration"))
    profiles = [profile(a.pop, problem, online_lower, online_scale, job) for a in algorithms] if not singleton else []
    history, events = [], []
    control_seconds = diagnostic_seconds = integration_seconds = 0.
    total_sent = total_accepted = total_bytes = 0
    checkpoints = sorted(set([min(v, job.budget) for v in (10000, 25000, 50000, 100000)] + [job.budget]))
    cursor, last_record = 0, -1
    evaluations = sum(a.evaluator.n_eval for a in algorithms)

    def observe():
        nonlocal diagnostic_seconds, last_record
        t = time.perf_counter()
        values = indicators(archiveF, pf)
        diagnostic_seconds += time.perf_counter()-t
        row = dict(fe=evaluations, seconds=time.perf_counter()-started, sent=total_sent,
                   accepted=total_accepted, payload_bytes=total_bytes, **values)
        row["profiles"] = [p.record() for p in profiles]
        history.append(row)
        last_record = evaluations
        if verbose:
            print(f"[{job.problem}/{job.label}/s{job.seed}] FE={evaluations}/{job.budget} HV={values['hv']:.6g} IGD+={values['igd_plus']:.6g} migrants={total_sent}", flush=True)

    observe()
    epochs = 0
    while evaluations < job.budget:
        for alg in algorithms:
            evolve(alg, n)
        epochs += 1
        evaluations = sum(a.evaluator.n_eval for a in algorithms)
        joinedX = np.vstack([archiveX]+[a.pop.get("X") for a in algorithms])
        joinedF = np.vstack([archiveF]+[a.pop.get("F") for a in algorithms])
        archiveX, archiveF = truncate_archive(joinedX, joinedF, job.archive)
        coordination = not singleton and epochs % job.interval == 0
        if coordination:
            t = time.perf_counter()
            profiles = [profile(a.pop, problem, online_lower, online_scale, job, prev) for a, prev in zip(algorithms, profiles)]
            actions = select_actions(profiles, job, rng)
            control_seconds += time.perf_counter()-t
            # Read all sources before modifying any recipient (synchronous exchange).
            prepared = []
            for source, target, ratio, mixing, score in actions:
                count = max(1, min(int(np.floor(job.rho*n)), int(np.floor(ratio*n))))
                migrants = copy.deepcopy(choose_migrants(algorithms[source].pop, profiles[source], count, job, mixing, rng))
                prepared.append((source, target, migrants, ratio, mixing, score))
            for source, target, migrants, ratio, mixing, score in prepared:
                before_count = problem.counter
                t = time.perf_counter()
                accepted = inject(algorithms[target], migrants, problem, job.method == "CROWDING")
                integration_seconds += time.perf_counter()-t
                if problem.counter != before_count:
                    raise AssertionError("Migration re-evaluated cached objective vectors")
                payload = migrants.get("X").nbytes + migrants.get("F").nbytes
                total_sent += len(migrants)
                total_accepted += accepted
                total_bytes += payload
                events.append(dict(fe=evaluations, source=source, target=target, sent=len(migrants), accepted=accepted, payload_bytes=payload, ratio=ratio, mixing=mixing, score=score))
            if prepared:
                archiveX, archiveF = truncate_archive(np.vstack([archiveX]+[a.pop.get("X") for a in algorithms]), np.vstack([archiveF]+[a.pop.get("F") for a in algorithms]), job.archive)
        at_threshold = cursor < len(checkpoints) and evaluations >= checkpoints[cursor]
        if at_threshold:
            while cursor < len(checkpoints) and evaluations >= checkpoints[cursor]:
                cursor += 1
        if coordination or at_threshold:
            observe()
    if last_record != evaluations:
        observe()
    if evaluations != job.budget or problem.counter != job.budget:
        raise AssertionError(("Budget mismatch", evaluations, problem.counter, job.budget))
    temporary = folder / f"population.{os.getpid()}.npz"
    np.savez_compressed(temporary, X=archiveX, F=archiveF, population_X=np.vstack([a.pop.get("X") for a in algorithms]), population_F=np.vstack([a.pop.get("F") for a in algorithms]))
    temporary.replace(folder/"population.npz")
    atomic_json(folder/"history.json", history)
    atomic_json(folder/"migration.json", events)
    return dict(metrics=history[-1] | {}, seconds=time.perf_counter()-started,
                initial_hash=initial_hash, evaluations=evaluations, sent=total_sent,
                accepted=total_accepted, payload_bytes=total_bytes, control_seconds=control_seconds,
                integration_seconds=integration_seconds, diagnostic_seconds=diagnostic_seconds,
                algorithms=[a.thermoo_kind for a in algorithms], archive_size=len(archiveF),
                controller="explicit heuristic; no neural training", island_execution="sequential within trial")
