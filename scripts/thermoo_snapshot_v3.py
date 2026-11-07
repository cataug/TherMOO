#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Immutable, versioned TherMOO statistics and compact publication figures.

Run from the existing experiment environment. No optimizer is started or changed.
"""
import time
START_NS = time.time_ns()  # The cutoff is fixed before scientific imports.

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import traceback
import warnings
import zipfile

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from scipy import stats

VERSION = "3.0.0"
METRICS = ("hv", "igd_plus")
METRIC_NAMES = {"hv": "HV  ↑", "igd_plus": "IGD+  ↓"}
BLOCKS = ("main", "ablation", "sensitivity", "homogeneous", "islands", "many", "dimensions")
STUDIES = ("main", "ablation", "sensitivity_interval", "sensitivity_rho",
           "sensitivity_temperature", "sensitivity_quality", "homogeneous",
           "composition_FULL", "islands", "islands_FULL", "dimensions", "many")
METHODS = ("NSGA2", "MOEAD", "GDE3", "SMS", "NO_MIG", "RANDOM", "ELITE", "HV_GATE", "CROWDING", "FULL", "NSGA3", "RVEA")
POLICIES = ("NO_MIG", "ELITE", "FULL")
DIM_METHODS = ("NSGA2", "MOEAD", "GDE3", "SMS", "NO_MIG", "ELITE", "HV_GATE", "FULL")
DIM_PROBLEMS = ("ZDT4", "DTLZ2", "DTLZ7", "WFG9", "UF1", "ZCAT20")
PANEL = ("ZDT1", "ZDT3", "DTLZ2", "DTLZ7", "WFG1", "WFG9", "LZ09_F1", "UF8", "ZCAT14", "ZCAT20")
FAMILIES = ("ZDT", "DTLZ", "WFG", "LZ09", "UF", "ZCAT")
SINGLETONS = {"NSGA2", "MOEAD", "GDE3", "SMS", "NSGA3", "RVEA"}
LABELS = {"NSGA2": "NSGA-II", "MOEAD": "MOEA/D", "GDE3": "GDE3", "SMS": "SMS-EMOA",
          "NSGA3": "NSGA-III", "NO_MIG": "No migration", "HV_GATE": "HV gate",
          "CROWDING": "Crowding", "RANDOM": "Random", "ELITE": "Elite", "FULL": "FULL",
          "conventional": "Conventional state", "no_entropy": "Without entropy block",
          "no_free_energy": "Without free energy", "no_js": "Without JS",
          "fixed_rate": "Fixed migration rate", "random_source": "Random source",
          "boltzmann_only": "Boltzmann only", "mixed": "Mixed optimizers"}


def log(message):
    print(message, flush=True)


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], "little")


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(safe_json(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows, columns=()):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if frame.empty and len(frame.columns) == 0:
        frame = pd.DataFrame(columns=columns or ["status"])
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")
    return frame


def allocate_version(root):
    """Reserve paired directories under a shared, short-lived flock."""
    stats_root, figures_root = root / "STATS", root / "FIGURES"
    stats_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)
    with (stats_root / ".versions.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        versions = [int(p.name[1:]) for parent in (stats_root, figures_root)
                    for p in parent.iterdir() if re.fullmatch(r"v[1-9][0-9]*", p.name)]
        version = f"v{max(versions, default=0) + 1}"
        stats_dir, figures_dir = stats_root / version, figures_root / version
        stats_dir.mkdir()
        figures_dir.mkdir()
    return version, stats_dir, figures_dir


class CutoffError(Exception):
    pass


def stable_read(path, cutoff_ns):
    """Reject files modified, replaced or atomically published after the cutoff."""
    before = path.stat()
    if max(before.st_mtime_ns, before.st_ctime_ns) > cutoff_ns:
        raise CutoffError(f"published_after_cutoff:{path.name}")
    data = path.read_bytes()
    after = path.stat()
    identity = lambda s: (s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if identity(before) != identity(after):
        raise CutoffError(f"changed_during_read:{path.name}")
    return data, {"size": before.st_size, "mtime_ns": before.st_mtime_ns,
                  "ctime_ns": before.st_ctime_ns, "sha256": hashlib.sha256(data).hexdigest()}


def normalize_job(job, family):
    row = asdict(job)
    row.update(run_id=job.signature(), requested_m=job.m, requested_d=job.d,
               m=job.objectives, d=job.variables, family=family(job.problem))
    return row


def discover_input_roots(output_root, requested=None):
    candidates = requested or [output_root, Path('/home/tahiti/Malashin_Projects/TherMOO')]
    roots = list(dict.fromkeys(Path(p).expanduser().resolve() for p in candidates))
    roots = [p for p in roots if p.is_dir()]
    active = set()
    for process in Path('/proc').glob('[0-9]*'):
        try:
            if int(process.name) == os.getpid() or process.stat().st_uid != os.getuid():
                continue
            args = (process/'cmdline').read_bytes().decode(errors='replace').split('\0')
            if not any('thermoo' in a.lower() for a in args):
                continue
            if any('snapshot' in a.lower() for a in args):
                continue
            location = (process/'cwd').resolve()
            for item in (process/'environ').read_bytes().split(b'\0'):
                if item.startswith(b'THERMOO_ROOT='):
                    location = Path(os.fsdecode(item.split(b'=', 1)[1])).resolve()
            if '--root' in args:
                location = Path(args[args.index('--root')+1]).resolve()
            if location in roots:
                active.add(location)
        except (OSError, ValueError, IndexError):
            continue
    engines = [p for p in roots if (p/'thermoo/engine.py').is_file() and (p/'thermoo/plan.py').is_file()]
    if not engines:
        raise RuntimeError('No experiment code found in input roots: '+', '.join(map(str, roots)))
    def completed_count(p):
        return sum(1 for _ in (p/'outputs/full').glob('*/*/result.json'))
    engines.sort(key=lambda p: (p in active, completed_count(p)), reverse=True)
    return roots, engines[0]


def validated_result(folder, job, fingerprint, cutoff_ns):
    run_id = job.signature()
    result_file = folder / "result.json"
    raw, meta = stable_read(result_file, cutoff_ns)
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("expected_result_object")
    if data.get("status") != "complete":
        raise ValueError("status_not_complete")
    if data.get("fingerprint") != fingerprint or data.get("run_id") != run_id:
        raise ValueError("identity_or_fingerprint_mismatch")
    if data.get("job") != asdict(job):
        raise ValueError("planned_job_metadata_mismatch")
    if data.get("evaluations") != job.budget or data.get("metrics", {}).get("fe") != job.budget:
        raise ValueError("evaluation_budget_mismatch")
    metrics = data["metrics"]
    if not all(math.isfinite(float(metrics[m])) and float(metrics[m]) >= 0 for m in METRICS):
        raise ValueError("nonfinite_or_negative_metric")
    if not re.fullmatch(r"[0-9a-f]{64}", str(data.get("initial_hash", ""))):
        raise ValueError("missing_initial_population_hash")
    for field in ("seconds", "sent", "accepted", "payload_bytes", "control_seconds", "integration_seconds", "diagnostic_seconds"):
        if not math.isfinite(float(data[field])) or float(data[field]) < 0:
            raise ValueError(f"invalid_counter_or_time:{field}")
    if data["seconds"] <= 0 or data["accepted"] > data["sent"]:
        raise ValueError("inconsistent_counters_or_runtime")
    algorithms = data.get("algorithms")
    expected_k = 1 if job.method in SINGLETONS else job.islands
    if not isinstance(algorithms, list) or len(algorithms) != expected_k or not all(isinstance(a, str) for a in algorithms):
        raise ValueError("actual_island_algorithm_metadata_mismatch")
    pop = (folder / "population.npz").stat()
    if max(pop.st_mtime_ns, pop.st_ctime_ns) > cutoff_ns:
        raise CutoffError("population_published_after_cutoff")
    if pop.st_size == 0:
        raise ValueError("empty_population_archive")
    # Re-stat the commit marker after validating the population file.
    stable_read(result_file, cutoff_ns)
    return raw, meta, data


def source_result(root, roots, job, fingerprint, cutoff_ns, issues):
    valid, failures = [], []
    for candidate_root in roots:
        folder = candidate_root / "outputs/full" / fingerprint / job.signature()
        if not (folder / "result.json").is_file():
            continue
        try:
            raw, meta, data = validated_result(folder, job, fingerprint, cutoff_ns)
            # Timers can differ on duplicate reruns; scientific outcomes may not.
            science = {k: data.get(k) for k in ("fingerprint", "run_id", "initial_hash", "evaluations", "sent", "accepted", "payload_bytes", "algorithms", "archive_size")}
            science["metrics"] = {k: v for k, v in data["metrics"].items() if k != "seconds"}
            science["job"] = dict(data["job"])
            digest = hashlib.sha256(json.dumps(science, sort_keys=True).encode()).hexdigest()
            valid.append((folder, raw, meta, data, digest))
        except (OSError, ValueError, KeyError, TypeError, CutoffError) as error:
            failures.append(str(error))
            issues.append(dict(run_id=job.signature(), file=str(folder/"result.json"), reason=str(error), severity="source_copy_excluded"))
    if not valid:
        raise ValueError(failures[0] if failures else "not_complete_at_cutoff")
    if len({item[4] for item in valid}) > 1:
        for folder, *_ in valid:
            issues.append(dict(run_id=job.signature(), file=str(folder/"result.json"), reason="conflicting_scientific_results_for_same_run", severity="conflict"))
        raise ValueError("conflicting_scientific_results_for_same_run")
    # Prefer the active/code root, then any compatible copy from the other root.
    valid.sort(key=lambda x: (x[0].parents[3] != root, x[2]["ctime_ns"]))
    folder, raw, meta, data, _ = valid[0]
    return folder, raw, meta, data, len(valid)


def capture(root, fingerprint, jobs, family, stats_dir, cutoff_ns, source_roots=None):
    """Capture only immutable, successful, planned trials. Audit every exclusion."""
    source = root / "outputs" / "full" / fingerprint
    source_roots = list(dict.fromkeys(Path(p).resolve() for p in (source_roots or [root])))
    records, histories, issues, missing, provenance = [], {}, [], [], []
    expected = {job.signature(): job for job in jobs}
    snapshot_dir = stats_dir / "snapshot"
    snapshot_dir.mkdir()
    paths = {name: gzip.open(snapshot_dir / f"{name}.jsonl.gz", "wt", encoding="utf-8", compresslevel=3)
             for name in ("results", "histories", "migrations", "scheduler")}
    try:
        for i, (run_id, job) in enumerate(expected.items(), 1):
            folder = source / run_id
            result_file = folder / "result.json"
            try:
                folder, raw, meta, data, copies = source_result(root, source_roots, job, fingerprint, cutoff_ns, issues)
                result_file = folder / "result.json"
                metrics = data["metrics"]
            except (OSError, ValueError, KeyError, TypeError, CutoffError) as error:
                reason = str(error)
                if reason != "not_complete_at_cutoff":
                    issues.append(dict(run_id=run_id, file=str(result_file), reason=reason, severity="excluded"))
                missing.append(dict(run_id=run_id, block=job.block, problem=job.problem,
                                    method=job.method, seed=job.seed, reason=reason))
                continue
            row = normalize_job(job, family)
            row.update({m: float(metrics[m]) for m in METRICS})
            for key in ("seconds", "sent", "accepted", "payload_bytes", "control_seconds", "integration_seconds", "diagnostic_seconds", "archive_size"):
                row[key] = data.get(key, np.nan)
            row.update(actual_evaluations=data["evaluations"], initial_hash=data["initial_hash"],
                       actual_algorithms="|".join(data.get("algorithms", [])),
                       actual_islands=len(data.get("algorithms", [])),
                       source_result=str(result_file), source_root=str(folder.parents[3]), source_copies=copies, result_sha256=meta["sha256"])
            for key in ("H", "E", "free", "temperature", "hv", "progress", "drift", "diversity", "nd_fraction", "dH", "dF", "dDiv"):
                vals = [p[key] for p in metrics.get("profiles", []) if isinstance(p, dict)
                        and key in p and isinstance(p[key], (int, float)) and math.isfinite(p[key])]
                row[f"profile_{key}_mean"] = float(np.mean(vals)) if vals else np.nan
            paths["results"].write(json.dumps(dict(run_id=run_id, source=str(result_file), meta=meta,
                                                   result_json=raw.decode("utf-8")), ensure_ascii=False) + "\n")
            provenance.append(dict(run_id=run_id, file="result.json", **meta))
            for name, filename in (("histories", "history.json"), ("migrations", "migration.json")):
                file = folder / filename
                try:
                    blob, aux_meta = stable_read(file, cutoff_ns)
                    values = json.loads(blob)
                    if not isinstance(values, list):
                        raise ValueError("expected_json_list")
                    if name == "histories":
                        # Keep only points needed for the compact convergence figure in RAM.
                        histories[run_id] = {int(v["fe"]): {m: float(v[m]) for m in METRICS}
                                             for v in values if int(v.get("fe", -1)) in (400, 10000, 25200, 50000, 100000)
                                             and all(m in v and math.isfinite(float(v[m])) for m in METRICS)}
                        if not values or int(values[-1]["fe"]) != job.budget:
                            raise ValueError("history_missing_final_budget")
                    paths[name].write(json.dumps(dict(run_id=run_id, meta=aux_meta,
                                                       data_json=blob.decode("utf-8")), ensure_ascii=False) + "\n")
                    provenance.append(dict(run_id=run_id, file=filename, **aux_meta))
                except (OSError, ValueError, KeyError, TypeError, CutoffError) as error:
                    histories.pop(run_id, None) if name == "histories" else None
                    issues.append(dict(run_id=run_id, file=str(file), reason=str(error), severity="auxiliary_unavailable"))
            row["cpu_seconds"], row["peak_rss_gib"] = np.nan, np.nan
            for scheduler in ("scheduler", "dynamic"):
                sidecar = folder.parents[3] / "reports" / scheduler / fingerprint / f"{run_id}.json"
                if not sidecar.is_file():
                    continue
                try:
                    blob, aux_meta = stable_read(sidecar, cutoff_ns)
                    side = json.loads(blob)
                    if side.get("status") not in ("complete", "cached"):
                        continue
                    row["cpu_seconds"] = side.get("cpu_seconds", np.nan)
                    row["peak_rss_gib"] = side.get("peak_rss_bytes", np.nan) / 2**30
                    paths["scheduler"].write(json.dumps(dict(run_id=run_id, meta=aux_meta,
                                                              data_json=blob.decode("utf-8"))) + "\n")
                    break
                except (OSError, ValueError, TypeError, CutoffError):
                    continue
            records.append(row)
            if len(records) % 500 == 0:
                log(f"[SNAPSHOT] accepted={len(records)} checked={i}/{len(expected)}")
        if source.is_dir():
            for file in source.glob("*/result.json"):
                if file.parent.name not in expected:
                    issues.append(dict(run_id=file.parent.name, file=str(file), severity="excluded",
                                       reason="not_in_current_full_plan"))
    finally:
        for stream in paths.values():
            stream.close()
    return pd.DataFrame(records), histories, issues, missing, provenance


def case_identity(row, omit=()):
    fields = ("problem", "m", "d", "budget", "population", "islands", "composition",
              "interval", "rho", "temperature", "quality", "ablation", "archive", "hv_epsilon", "js_epsilon")
    payload = {k: row[k] for k in fields if k not in omit}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:20], text


def make_views(planned):
    """Predeclare every comparison, including reuse of exact main control IDs."""
    rows = []
    defaults = {"interval": 10, "rho": .1, "temperature": 1., "quality": "rank_crowding"}
    def add(row, study, level=None, facet="all", omit=(), control="FULL", reused=False):
        case, parameters = case_identity(row, omit)
        rows.append(dict(row, study=study, facet=facet, case=case, case_parameters=parameters,
                         level=str(row["method"] if level is None else level), control=str(control),
                         comparison_family=("sensitivity" if study.startswith("sensitivity_") else
                                            {"composition_FULL": "homogeneous", "islands_FULL": "islands"}.get(study, study)),
                         is_reused_control=reused))
    for row in planned:
        b, method, problem = row["block"], row["method"], row["problem"]
        if b == "main":
            add(row, "main")
            if problem in PANEL:
                if method == "FULL":
                    add(row, "ablation", omit=("ablation",), reused=True)
                    for parameter, default in defaults.items():
                        add(row, f"sensitivity_{parameter}", level=default, omit=(parameter,), control=default, reused=True)
                    add(row, "composition_FULL", level="mixed", omit=("composition",), control="mixed", reused=True)
                    add(row, "islands_FULL", level="K=4", omit=("islands",), control="K=4", reused=True)
                if method in POLICIES:
                    add(row, "homogeneous", facet="mixed", reused=True)
                    add(row, "islands", facet="K=4", reused=True)
            if problem in DIM_PROBLEMS and method in DIM_METHODS:
                add(row, "dimensions", facet="Default", reused=True)
        elif b == "ablation":
            add(row, "ablation", level=row["ablation"], omit=("ablation",))
        elif b == "sensitivity":
            changed = [k for k, v in defaults.items() if row[k] != v]
            if len(changed) != 1:
                raise ValueError(f"Sensitivity must change exactly one factor: {row['run_id']}")
            key = changed[0]
            add(row, f"sensitivity_{key}", level=row[key], omit=(key,), control=defaults[key])
        elif b == "homogeneous":
            add(row, "homogeneous", facet=row["composition"])
            if method == "FULL":
                add(row, "composition_FULL", level=row["composition"], omit=("composition",), control="mixed")
        elif b == "islands":
            add(row, "islands", facet=f"K={row['islands']}")
            if method == "FULL":
                add(row, "islands_FULL", level=f"K={row['islands']}", omit=("islands",), control="K=4")
        elif b == "dimensions":
            add(row, "dimensions", facet=f"D={row['d']}")
        elif b == "many":
            add(row, "many")
    result = pd.DataFrame(rows)
    if result.duplicated(["study", "facet", "case", "level", "seed"]).any():
        raise ValueError("Duplicate planned view cells")
    return result


def holm(pvalues, planned_count=None):
    p = np.asarray(pvalues, float)
    m = len(p) if planned_count is None else int(planned_count)
    if m < len(p):
        raise ValueError("Holm family smaller than observed hypotheses")
    order = np.argsort(p, kind="stable")
    out = np.empty(len(p))
    maximum = 0.
    for position, index in enumerate(order):
        maximum = max(maximum, (m - position) * p[index])
        out[index] = min(1., maximum)
    return out


def bootstrap_mean(values, rng, count):
    x = np.asarray(values, float)
    if len(x) < 2:
        return np.nan, np.nan
    samples = x[rng.integers(0, len(x), size=(count, len(x)))].mean(axis=1)
    return tuple(np.quantile(samples, [.025, .975]))


def permutation_p(differences, rng, count):
    d = np.asarray(differences, float)
    if not len(d) or np.all(d == 0):
        return 1.
    observed = abs(float(d.mean()))
    tolerance = max(1e-300, observed * 1e-12)
    greater = 0
    # Exact enumeration is cheap for the six/ten-problem panels.
    exact = len(d) <= 16 and 2**len(d) <= max(count, 65536)
    total = 2**len(d) if exact else count
    for start in range(0, total, 4096):
        size = min(4096, total - start)
        if exact:
            signs = 2 * (((np.arange(start, start + size, dtype=np.uint64)[:, None] >>
                            np.arange(len(d), dtype=np.uint64)) & 1).astype(float)) - 1
        else:
            signs = 2 * rng.integers(0, 2, size=(size, len(d))) - 1
        greater += int(np.count_nonzero(np.abs((signs * d).mean(axis=1)) >= observed - tolerance))
    return greater / total if exact else (greater + 1) / (total + 1)


def paired_description(delta, rng, bootstrap, infer, permutations):
    d = np.asarray(delta, float)
    n = len(d)
    low, high = bootstrap_mean(d, rng, bootstrap)
    nonzero = d[d != 0]
    ranks = stats.rankdata(np.abs(nonzero))
    effect = float(np.sum(ranks * np.sign(nonzero)) / ranks.sum()) if len(nonzero) else 0.
    constant = not len(d) or np.ptp(d) == 0
    shapiro = float(stats.shapiro(d).pvalue) if n >= 3 and not constant else np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        symmetric = len(nonzero) >= 8 and (np.ptp(nonzero) == 0 or stats.skewtest(nonzero).pvalue >= .05)
    p_t = (float(stats.ttest_1samp(d, 0).pvalue) if infer and not constant and shapiro >= .05 else np.nan)
    p_w = (float(stats.wilcoxon(d, zero_method="pratt", method="approx").pvalue)
           if infer and len(nonzero) and symmetric else (1. if infer and not len(nonzero) else np.nan))
    sign_p = float(stats.binomtest(int(np.sum(nonzero > 0)), len(nonzero)).pvalue) if infer and len(nonzero) else (1. if infer else np.nan)
    return dict(n_pairs=n, advantage_mean=float(d.mean()) if n else np.nan,
                advantage_ci_low=low, advantage_ci_high=high,
                paired_win_probability=float(np.mean((d > 0) + .5 * (d == 0))) if n else np.nan,
                rank_biserial=effect if n else np.nan, shapiro_difference_p=shapiro,
                shapiro_status="constant" if constant else ("tested" if n >= 3 else "too_few"),
                p_primary=permutation_p(d, rng, permutations) if infer else np.nan,
                p_ttest_sensitivity=p_t, p_wilcoxon_sensitivity=p_w, p_sign_sensitivity=sign_p)


def coverage_table(planned, raw):
    plan = pd.DataFrame(planned)
    counts = plan.groupby(["block", "method"], sort=False).size().rename("planned")
    completed = raw.groupby(["block", "method"]).size().rename("completed") if len(raw) else pd.Series(0, index=counts.index, name="completed")
    out = counts.to_frame().join(completed).fillna({"completed": 0}).reset_index()
    out["completed"] = out.completed.astype(int)
    out["fraction"] = out.completed / out.planned
    return out


def cell_statistics(views, observed, bootstrap):
    keys = ["study", "facet", "case", "level"]
    planned = {k: g for k, g in views.groupby(keys, sort=False)}
    actual = {k: g for k, g in observed.groupby(keys, sort=False)} if len(observed) else {}
    rows = []
    for index, (key, expected) in enumerate(planned.items(), 1):
        g = actual.get(key)
        first = expected.iloc[0]
        base = {k: first[k] for k in ("study", "facet", "case", "level", "control", "comparison_family", "problem", "family", "m", "d", "case_parameters")}
        n = len(g) if g is not None else 0
        base.update(n=n, expected_n=len(expected), complete=n == len(expected),
                    reused_control=bool(first.is_reused_control))
        for metric in METRICS:
            x = g[metric].to_numpy(float) if n else np.array([])
            low, high = bootstrap_mean(x, np.random.default_rng(stable_seed("cell", *key, metric)), bootstrap)
            constant = n > 0 and np.ptp(x) == 0
            rows.append(dict(base, metric=metric, mean=float(x.mean()) if n else np.nan,
                             sd=float(x.std(ddof=1)) if n >= 2 else np.nan,
                             median=float(np.median(x)) if n else np.nan,
                             q25=float(np.quantile(x, .25)) if n else np.nan,
                             q75=float(np.quantile(x, .75)) if n else np.nan,
                             mean_ci_low=low, mean_ci_high=high,
                             shapiro_raw_p=float(stats.shapiro(x).pvalue) if n >= 3 and not constant else np.nan,
                             shapiro_status="constant" if constant else ("tested" if n >= 3 else "too_few")))
        if index % 500 == 0:
            log(f"[CELLS] {index}/{len(planned)}")
    return pd.DataFrame(rows)


def pair_comparisons(views, observed, bootstrap, permutations, issues):
    rows = []
    last_log = time.monotonic()
    actual = {k: g.set_index("seed") for k, g in observed.groupby(["study", "facet", "case", "level"], sort=False)} if len(observed) else {}
    groups = list(views.groupby(["study", "facet", "case"], sort=False))
    for index, (key, expected) in enumerate(groups, 1):
        first = expected.iloc[0]
        control = first.control
        required = set(expected[expected.level == control].seed)
        control_group = actual.get((*key, control))
        for level in expected.level.unique():
            if level == control:
                continue
            variant = actual.get((*key, level))
            paired_seeds = sorted(set(control_group.index) & set(variant.index)) if control_group is not None and variant is not None else []
            valid = [seed for seed in paired_seeds if control_group.loc[seed, "initial_hash"] == variant.loc[seed, "initial_hash"]]
            if len(valid) != len(paired_seeds):
                issues.append(dict(run_id="", file="pairing", severity="pair_excluded",
                                   reason=f"initial_hash_mismatch:{key}:{level}:{len(paired_seeds)-len(valid)}"))
            infer = len(required) >= 30 and set(valid) == required and len(valid) == len(required)
            for metric in METRICS:
                sign = 1 if metric == "hv" else -1
                delta = sign * (control_group.loc[valid, metric].to_numpy(float) - variant.loc[valid, metric].to_numpy(float)) if valid else np.array([])
                description = paired_description(delta, np.random.default_rng(stable_seed("pair", *key, level, metric)), bootstrap, infer, permutations)
                rows.append(dict(study=key[0], facet=key[1], case=key[2], problem=first.problem,
                                 family=first.family, m=first.m, d=first.d, control=control, level=level,
                                 comparison_family=first.comparison_family, metric=metric, expected_pairs=len(required),
                                 eligible_inference=infer, **description))
        if index % 25 == 0 or time.monotonic()-last_log >= 15:
            log(f"[PAIRS] cases={index}/{len(groups)}")
            last_log = time.monotonic()
    result = pd.DataFrame(rows)
    result["p_holm"] = np.nan
    result["sign_sensitivity_holm"] = np.nan
    result["planned_hypotheses"] = 0
    result["decision"] = "insufficient_runs"
    for family, group in result.groupby("comparison_family", sort=False):
        total = len(group)  # Unfinished planned hypotheses are conservatively padded with p=1.
        mask = group.eligible_inference
        idx = group.index[mask]
        result.loc[group.index, "planned_hypotheses"] = total
        if len(idx):
            result.loc[idx, "p_holm"] = holm(result.loc[idx, "p_primary"], total)
            result.loc[idx, "sign_sensitivity_holm"] = holm(result.loc[idx, "p_sign_sensitivity"], total)
            for row_id in idx:
                row = result.loc[row_id]
                result.loc[row_id, "decision"] = ("win" if row.advantage_mean > 0 else "loss") if row.p_holm < .05 else "no_difference_detected"
    return result


def balanced_ranks(views, observed, bootstrap, permutations):
    """A case enters only when all planned methods have all matched seeds."""
    case_rows, availability = [], []
    actual = {k: g for k, g in observed.groupby(["study", "facet", "case"], sort=False)} if len(observed) else {}
    for key, expected in views.groupby(["study", "facet", "case"], sort=False):
        first = expected.iloc[0]
        # The binary benchmark substitutes SPEA2 into the GDE3 slot.
        excluded_binary = first.problem == "ZDT5"
        levels = list(expected.level.unique())
        seeds = set(expected.seed)
        found = actual.get(key)
        complete = found is not None and len(seeds) >= 30
        if complete:
            for level in levels:
                if set(found[found.level == level].seed) != seeds:
                    complete = False
            if complete and found.groupby("seed").initial_hash.nunique().max() != 1:
                complete = False
        availability.append(dict(study=key[0], facet=key[1], case=key[2], problem=first.problem,
                                 family=first.family, eligible=bool(complete and not excluded_binary),
                                 excluded_binary=excluded_binary, levels=len(levels), seeds=len(seeds)))
        if not complete or excluded_binary:
            continue
        for metric in METRICS:
            medians = found.groupby("level")[metric].median().reindex(levels).to_numpy()
            ranks = stats.rankdata(-medians if metric == "hv" else medians, method="average")
            for level, rank, median in zip(levels, ranks, medians):
                case_rows.append(dict(study=key[0], facet=key[1], case=key[2], problem=first.problem,
                                      family=first.family, metric=metric, level=level, rank=rank, median=median))
    cases = pd.DataFrame(case_rows, columns=["study", "facet", "case", "problem", "family", "metric", "level", "rank", "median"])
    avail = pd.DataFrame(availability)
    summaries, omnibus, posthoc = [], [], []
    for (study, facet, metric), sub in cases.groupby(["study", "facet", "metric"], sort=False):
        log(f"[RANKS] {study}/{facet}/{metric}: complete_problems={sub.case.nunique()}")
        expected = avail[(avail.study == study) & (avail.facet == facet) & ~avail.excluded_binary]
        levels = list(sub.level.unique())
        matrix = sub.pivot(index="case", columns="level", values="rank").reindex(columns=levels)
        meta = sub.drop_duplicates("case").set_index("case").loc[matrix.index]
        family_indices = [np.flatnonzero(meta.family.to_numpy() == f) for f in meta.family.unique()]
        rng = np.random.default_rng(stable_seed("ranks", study, facet, metric))
        samples = np.zeros((bootstrap, len(levels)))
        for indices in family_indices:
            picks = indices[rng.integers(0, len(indices), size=(bootstrap, len(indices)))]
            samples += matrix.to_numpy()[picks].mean(axis=1) / len(family_indices)
        mean = np.mean([matrix.to_numpy()[i].mean(axis=0) for i in family_indices], axis=0)
        low, high = np.quantile(samples, [.025, .975], axis=0)
        for j, level in enumerate(levels):
            summaries.append(dict(study=study, facet=facet, metric=metric, level=level,
                                  family_balanced_rank=mean[j], ci_low=low[j] if len(matrix) >= 2 else np.nan,
                                  ci_high=high[j] if len(matrix) >= 2 else np.nan,
                                  mean_rank=float(matrix[level].mean()), cases=len(matrix), planned_cases=len(expected),
                                  families=len(family_indices), interim=len(matrix) < len(expected)))
        if len(levels) < 3 or len(matrix) < 3:
            continue
        a = matrix.to_numpy()
        n, k = a.shape
        statistic = float(np.sum((a.mean(axis=0) - (k + 1) / 2)**2))
        greater = 0
        for start in range(0, permutations, 256):
            count = min(256, permutations - start)
            order = np.argsort(rng.random((count, n, k)), axis=2)
            perm = np.take_along_axis(np.broadcast_to(a, (count, n, k)), order, axis=2)
            simulated = np.sum((perm.mean(axis=1) - (k + 1) / 2)**2, axis=1)
            greater += int(np.count_nonzero(simulated >= statistic - max(1e-300, statistic * 1e-12)))
        p = (greater + 1) / (permutations + 1)
        omnibus.append(dict(study=study, facet=facet, metric=metric, cases=n, methods=k,
                            statistic=12*n*statistic/(k*(k+1)), p_permutation=p,
                            interim=n < len(expected), unit="one_problem_within_fixed_facet"))
        control = views[(views.study == study) & (views.facet == facet)].control.iloc[0]
        for level in levels:
            if level == control:
                continue
            delta = a[:, levels.index(level)] - a[:, levels.index(control)]
            posthoc.append(dict(study=study, facet=facet, metric=metric, control=control, level=level,
                                cases=n, rank_advantage=float(delta.mean()),
                                p_primary=permutation_p(delta, rng, permutations),
                                comparison_family=views[views.study == study].comparison_family.iloc[0]))
    omnibus = pd.DataFrame(omnibus)
    posthoc = pd.DataFrame(posthoc)
    if len(omnibus):
        planned_omnibus = sum(len(METRICS) for _, g in views.groupby(["study", "facet"])
                               if g.level.nunique() >= 3)
        omnibus["p_holm"] = holm(omnibus.p_permutation, planned_omnibus)
    if len(posthoc):
        posthoc["p_holm"] = np.nan
        for family, group in posthoc.groupby("comparison_family"):
            study_names = views[views.comparison_family == family].study.unique()
            planned_total = sum((g.level.nunique()-1)*len(METRICS) for _, g in views[views.study.isin(study_names)].groupby(["study", "facet"]))
            posthoc.loc[group.index, "p_holm"] = holm(group.p_primary, planned_total)
        gates = {(r.study, r.facet, r.metric): r.p_holm < .05 for r in omnibus.itertuples()}
        posthoc["omnibus_gate"] = [gates[(r.study, r.facet, r.metric)] for r in posthoc.itertuples()]
        posthoc["reject"] = posthoc.omnibus_gate & (posthoc.p_holm < .05)
    return cases, pd.DataFrame(summaries), avail, omnibus, posthoc


def family_bootstrap(frame, value, bootstrap, seed):
    """Give each represented family equal weight; resample problems within it."""
    rng = np.random.default_rng(seed)
    samples, centers = [], []
    for _, group in frame.groupby("family", sort=False):
        x = group[value].to_numpy(float)
        centers.append(x.mean())
        samples.append(x[rng.integers(0, len(x), (bootstrap, len(x)))].mean(axis=1))
    low, high = np.quantile(np.mean(samples, axis=0), [.025, .975])
    return float(np.mean(centers)), float(low), float(high)


def paired_effects(comparisons, bootstrap):
    rows = []
    if not len(comparisons):
        return pd.DataFrame()
    # All levels of a study/facet use the same complete problem set in this figure.
    for (study, facet), group in comparisons.groupby(["study", "facet"], sort=False):
        levels = list(group.level.unique())
        common = set(group.case.unique())
        for level in levels:
            common &= set(group[(group.level == level) & group.eligible_inference].case)
        if not common:
            continue
        for (metric, level), sub in group[group.case.isin(common)].groupby(["metric", "level"], sort=False):
            center, low, high = family_bootstrap(sub, "paired_win_probability", bootstrap,
                                                stable_seed("effect", study, facet, metric, level))
            rows.append(dict(study=study, facet=facet, metric=metric, level=level,
                             control=sub.control.iloc[0], paired_win_probability=center,
                             ci_low=low if len(sub) >= 2 else np.nan,
                             ci_high=high if len(sub) >= 2 else np.nan,
                             cases=len(sub), families=sub.family.nunique(),
                             planned_cases=group.case.nunique(), interim=len(sub) < group.case.nunique()))
    return pd.DataFrame(rows)


def convergence_table(views, observed, histories, bootstrap):
    selected = ("NSGA2", "MOEAD", "GDE3", "SMS", "NO_MIG", "ELITE", "FULL")
    grid = (400, 10000, 25200, 50000, 100000)
    plan = views[(views.study == "main") & views.level.isin(selected) & (views.problem != "ZDT5")]
    actual = observed[(observed.study == "main") & observed.level.isin(selected)] if len(observed) else observed
    grouped = {key: g for key, g in actual.groupby("case")} if len(actual) else {}
    rows = []
    for case, expected in plan.groupby("case", sort=False):
        g = grouped.get(case)
        if g is None or len(g) != len(expected) or g.groupby("seed").initial_hash.nunique().max() != 1:
            continue
        if any(not set(grid).issubset(histories.get(rid, {})) for rid in g.run_id):
            continue
        for fe in grid:
            for metric in METRICS:
                medians = [np.median([histories[rid][fe][metric] for rid in g[g.level == level].run_id]) for level in selected]
                ranks = stats.rankdata(-np.asarray(medians) if metric == "hv" else medians)
                for level, rank in zip(selected, ranks):
                    rows.append(dict(case=case, problem=g.problem.iloc[0], family=g.family.iloc[0],
                                     fe=fe, metric=metric, level=level, rank=rank))
    case_rows = pd.DataFrame(rows)
    output = []
    if len(case_rows):
        for (fe, metric, level), g in case_rows.groupby(["fe", "metric", "level"], sort=False):
            mean, low, high = family_bootstrap(g, "rank", bootstrap, stable_seed("convergence", fe, metric, level))
            output.append(dict(fe=fe, metric=metric, level=level, family_balanced_rank=mean,
                               ci_low=low if len(g) >= 2 else np.nan, ci_high=high if len(g) >= 2 else np.nan,
                               cases=len(g), planned_cases=plan.case.nunique(), families=g.family.nunique()))
    return pd.DataFrame(output), case_rows


def migration_timing(raw, rank_cases):
    """Descriptive costs, using exactly the same main cases for every method."""
    rows = []
    if not len(raw) or not len(rank_cases):
        return pd.DataFrame()
    eligible = set(rank_cases[rank_cases.study == "main"].problem)
    main = raw[(raw.block == "main") & raw.problem.isin(eligible)]
    for method, group in main.groupby("method", sort=False):
        per_case = []
        for (problem, family), g in group.groupby(["problem", "family"]):
            record = dict(problem=problem, family=family,
                          sent_per_10k=float(np.median(g.sent / g.actual_evaluations * 10000)),
                          accepted_per_10k=float(np.median(g.accepted / g.actual_evaluations * 10000)),
                          payload_kib_per_10k=float(np.median(g.payload_bytes / 1024 / g.actual_evaluations * 10000)),
                          wall_seconds=float(np.median(g.seconds)), cpu_seconds=float(g.cpu_seconds.median()) if g.cpu_seconds.notna().any() else np.nan)
            for component in ("control", "integration", "diagnostic"):
                record[f"{component}_fraction"] = float(np.median(g[f"{component}_seconds"] / g.seconds))
            per_case.append(record)
        frame = pd.DataFrame(per_case)
        balanced = frame.groupby("family").mean(numeric_only=True).mean()
        rows.append(dict(method=method, problems=len(frame), families=frame.family.nunique(),
                         cpu_samples=int(group.cpu_seconds.notna().sum()), **balanced.to_dict()))
    return pd.DataFrame(rows)


BG = "#FAF8F2"
AX_BG = "#F4F1E8"
GRID = "#D8D3C5"


def plot_pair_cases(observed):
    """Descriptive, seed-matched outcomes; no 30-run gate on displaying data."""
    output = []
    if not len(observed):
        return pd.DataFrame()
    for (study, facet, case), group in observed.groupby(["study", "facet", "case"], sort=False):
        control = str(group.control.iloc[0])
        base = group[group.level == control].set_index("seed")
        if not len(base):
            continue
        for level, variant in group[group.level != control].groupby("level", sort=False):
            other = variant.set_index("seed")
            seeds = sorted(set(base.index) & set(other.index))
            seeds = [s for s in seeds if base.loc[s, "initial_hash"] == other.loc[s, "initial_hash"]]
            if not seeds:
                continue
            for metric in METRICS:
                a, b = base.loc[seeds, metric].to_numpy(float), other.loc[seeds, metric].to_numpy(float)
                delta = (a-b) * (1 if metric == "hv" else -1)
                output.append(dict(study=study, facet=facet, case=case, problem=group.problem.iloc[0],
                                   family=group.family.iloc[0], metric=metric, control=control, level=str(level),
                                   n=len(seeds), wins=int(np.sum(delta > 0)), ties=int(np.sum(delta == 0)),
                                   losses=int(np.sum(delta < 0)), win_probability=float(np.mean((delta > 0)+.5*(delta == 0))),
                                   advantage_mean=float(delta.mean())))
    return pd.DataFrame(output)


def aggregate_plot_pairs(pair_cases, bootstrap, by_family=False):
    """Equal-family means with a hierarchical problem/paired-seed bootstrap."""
    if not len(pair_cases):
        return pd.DataFrame()
    keys = ["study", "facet", "level", "metric"] + (["family"] if by_family else [])
    output = []
    for key, group in pair_cases.groupby(keys, sort=False):
        rng = np.random.default_rng(stable_seed("publication_effect", *key))
        centers, draws = [], []
        for _, family in group.groupby("family", sort=False):
            means, boot = [], []
            for row in family.itertuples():
                p = np.array([row.wins, row.ties, row.losses], float)/row.n
                counts = rng.multinomial(row.n, p, size=bootstrap)
                boot.append((counts[:, 0]+.5*counts[:, 1])/row.n)
                means.append(row.win_probability)
            values = np.asarray(boot).T
            picks = rng.integers(0, len(means), size=(bootstrap, len(means)))
            draws.append(np.take_along_axis(values, picks, axis=1).mean(axis=1))
            centers.append(np.mean(means))
        low, high = np.quantile(np.mean(draws, axis=0), [.025, .975])
        output.append(dict(zip(keys, key), win_percent=100*float(np.mean(centers)),
                           ci_low=100*float(low) if group.n.sum() >= 2 else np.nan,
                           ci_high=100*float(high) if group.n.sum() >= 2 else np.nan,
                           problems=group.case.nunique(), seed_pairs=int(group.n.sum()),
                           min_pairs=int(group.n.min()), max_pairs=int(group.n.max()),
                           families=group.family.nunique(), control=group.control.iloc[0],
                           all_cases_30_pairs=bool((group.n >= 30).all())))
    return pd.DataFrame(output)


def read_visual_history(stats_dir, raw):
    """Read the frozen logs once; do not touch live experiment files."""
    path = stats_dir / "snapshot/histories.jsonl.gz"
    histories, associations = {}, []
    if not path.is_file() or not len(raw):
        return histories, pd.DataFrame()
    main = raw[raw.block == "main"].set_index("run_id")
    ids = set(main.index)
    grid = {400, 10000, 25200, 50000, 100000}
    features = ("H", "E", "free", "drift", "diversity", "progress")
    for line in gzip.open(path, "rt", encoding="utf-8"):
        entry = json.loads(line)
        rid = entry["run_id"]
        if rid not in ids:
            continue
        history = json.loads(entry["data_json"])
        histories[rid] = {int(row["fe"]): {m: float(row[m]) for m in METRICS}
                          for row in history if int(row.get("fe", -1)) in grid
                          and all(m in row and math.isfinite(float(row[m])) for m in METRICS)}
        if main.loc[rid, "method"] != "FULL":
            continue
        points = []
        previous = None
        for row in history:
            profiles = row.get("profiles", [])
            if profiles and previous is not None and row["fe"] > previous["fe"]:
                point = {k: float(np.mean([p[k] for p in profiles if k in p])) for k in features
                         if any(k in p for p in profiles)}
                sent = row.get("sent", 0)-previous.get("sent", 0)
                accepted = row.get("accepted", 0)-previous.get("accepted", 0)
                point.update(migration_intensity=sent/(row["fe"]-previous["fe"])*10000,
                             acceptance=accepted/sent if sent > 0 else np.nan,
                             hv_gain=row.get("hv", np.nan)-previous.get("hv", np.nan))
                points.append(point)
            previous = row
        if len(points) < 5:
            continue
        frame = pd.DataFrame(points)
        for feature in features:
            if feature not in frame:
                continue
            for outcome in ("migration_intensity", "acceptance", "hv_gain"):
                sub = frame[[feature, outcome]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(sub) < 5 or sub[feature].nunique() < 2 or sub[outcome].nunique() < 2:
                    continue
                associations.append(dict(run_id=rid, problem=main.loc[rid, "problem"],
                                         family=main.loc[rid, "family"], feature=feature, outcome=outcome,
                                         spearman=float(stats.spearmanr(sub[feature], sub[outcome]).statistic),
                                         observations=len(sub)))
    return histories, pd.DataFrame(associations)


def publication_convergence(raw, histories, bootstrap):
    """Normalized actual-metric traces on a common complete-history seed panel."""
    if not len(raw):
        return pd.DataFrame(), pd.DataFrame()
    methods = ("NSGA2", "MOEAD", "GDE3", "SMS", "NO_MIG", "ELITE", "FULL")
    grid = (400, 10000, 25200, 50000, 100000)
    main = raw[(raw.block == "main") & (raw.problem != "ZDT5") & raw.method.isin(methods)]
    rows = []
    for problem, g in main.groupby("problem", sort=False):
        if set(g.method) != set(methods):
            continue
        complete = []
        for seed, panel in g.groupby("seed"):
            if len(panel) == len(methods) and panel.initial_hash.nunique() == 1 and all(set(grid).issubset(histories.get(rid, {})) for rid in panel.run_id):
                complete.append(panel)
        if not complete:
            continue
        for panel in complete:
            best_hv = max(histories[rid][fe]["hv"] for rid in panel.run_id for fe in grid)
            for run in panel.itertuples():
                initial_hv = histories[run.run_id][400]["hv"]
                initial_igd = histories[run.run_id][400]["igd_plus"]
                hv_span = best_hv-initial_hv
                for fe in grid:
                    values = histories[run.run_id][fe]
                    rows.append(dict(problem=problem, family=run.family, method=run.method, seed=run.seed,
                                     fe=fe, hv_gain_percent=100*(values["hv"]-initial_hv)/hv_span if hv_span > 1e-14 else np.nan,
                                     igd_relative=values["igd_plus"]/initial_igd if initial_igd > 1e-14 else np.nan))
    traces = pd.DataFrame(rows)
    summary = []
    if len(traces):
        per_problem = traces.groupby(["problem", "family", "method", "fe"], sort=False)[["hv_gain_percent", "igd_relative"]].median().reset_index()
        for (method, fe), group in per_problem.groupby(["method", "fe"], sort=False):
            for metric in ("hv_gain_percent", "igd_relative"):
                sub = group.dropna(subset=[metric])
                if not len(sub):
                    continue
                center, low, high = family_bootstrap(sub, metric, bootstrap, stable_seed("trace", method, fe, metric))
                summary.append(dict(method=method, fe=fe, metric=metric, value=center,
                                    ci_low=low if len(sub) > 1 else np.nan, ci_high=high if len(sub) > 1 else np.nan,
                                    problems=sub.problem.nunique()))
    return pd.DataFrame(summary), traces


def descriptive_costs(raw):
    if not len(raw):
        return pd.DataFrame()
    main = raw[(raw.block == "main") & ~raw.method.isin(SINGLETONS)]
    rows = []
    for method, group in main.groupby("method", sort=False):
        cases = []
        for (problem, family), g in group.groupby(["problem", "family"]):
            row = dict(problem=problem, family=family,
                       sent=float(np.median(g.sent/g.actual_evaluations*10000)),
                       accepted=float(np.median(g.accepted/g.actual_evaluations*10000)))
            for key in ("control", "integration", "diagnostic"):
                row[key] = float(np.median(g[f"{key}_seconds"]/g.seconds*100))
            cases.append(row)
        c = pd.DataFrame(cases)
        if not len(c):
            continue
        balanced = c.groupby("family").mean(numeric_only=True).mean()
        rows.append(dict(method=method, problems=c.problem.nunique(), runs=len(group), **balanced.to_dict()))
    return pd.DataFrame(rows)


class Figures:
    """Research figures in the user's cream/viridis/boxed-value template."""
    def __init__(self, folder, version, cutoff_ns):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt, self.folder, self.version = plt, folder, version
        self.index = []
        self.cmap = plt.get_cmap("viridis")
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 24,
                             "axes.titlesize": 28, "axes.labelsize": 25,
                             "xtick.labelsize": 22, "ytick.labelsize": 23,
                             "legend.fontsize": 20, "figure.titlesize": 30,
                             "axes.linewidth": 1.2, "lines.linewidth": 2.8,
                             "axes.edgecolor": "black", "axes.labelcolor": "black",
                             "text.color": "black", "xtick.color": "black", "ytick.color": "black",
                             "pdf.fonttype": 42, "ps.fonttype": 42})
        self.box = dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="black", linewidth=.9, alpha=.96)

    def canvas(self, title, columns=2, height=9.5, width=None, top=.85, bottom=.14):
        fig, axes = self.plt.subplots(1, columns, figsize=(width or (12 if columns == 1 else 18), height),
                                      facecolor=BG, squeeze=False)
        fig.suptitle(title, weight="bold", y=.975)
        fig.subplots_adjust(left=.22 if columns == 1 else .19, right=.97, top=top, bottom=bottom,
                            wspace=.20)
        for ax in axes.flat:
            ax.set_facecolor(AX_BG)
            ax.grid(color=GRID, lw=.8, alpha=.7, zorder=0)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(1.1)
        return fig, list(axes.flat)

    def label_box(self, ax, x, y, text, fontsize=20, **kwargs):
        kwargs.setdefault("ha", "center")
        kwargs.setdefault("va", "center")
        return ax.text(x, y, text, fontsize=fontsize, weight="bold",
                       color="black", bbox=self.box, zorder=12, **kwargs)

    def gradient_barh(self, ax, y, width, height=.55, left=0., c0=.12, c1=.92, alpha=.78):
        from matplotlib.patches import Rectangle
        if abs(width) < 1e-14:
            return
        low, high = sorted((left, left+width))
        gradient = np.linspace(c0, c1, 256).reshape(1, -1)
        ax.imshow(gradient, extent=(low, high, y-height/2, y+height/2), aspect="auto",
                  cmap=self.cmap, vmin=0, vmax=1, alpha=alpha, interpolation="bicubic", zorder=2)
        ax.add_patch(Rectangle((low, y-height/2), high-low, height, facecolor="none",
                               edgecolor="black", linewidth=1., zorder=3))

    def save(self, fig, number, slug, caption, note=""):
        stem = f"{number:02d}_{slug}"
        # Sample sizes/protocol belong in captions, not tiny dashboard footers.
        for extension in ("png", "pdf"):
            fig.savefig(self.folder / f"{stem}.{extension}", dpi=320, bbox_inches="tight", facecolor=BG, pad_inches=.16)
        self.plt.close(fig)
        self.index.append(dict(number=number, file=stem, status="created", caption=caption, note=note))
        log(f"[FIGURE] {stem}.png / .pdf")

    def skip(self, number, slug, reason):
        self.index.append(dict(number=number, file=f"{number:02d}_{slug}", status="unavailable", caption=reason, note=""))
        log(f"[FIGURE] {number:02d}_{slug}: {reason}")

    @staticmethod
    def scientific(value):
        if not np.isfinite(value):
            return "—"
        if value == 0:
            return "0"
        if .01 <= abs(value) < 1000:
            return f"{value:.3g}"
        return f"{value:.1e}".replace("e-0", "e−").replace("e+0", "e+").replace("e-", "e−")

    def effects(self, data, number, slug, title, rows):
        """Each row names study, facet, level, and its human label."""
        if not len(data) or not any(len(data[(data.study == s) & (data.facet == f) & (data.level == v)]) for s, f, v, _ in rows):
            return self.skip(number, slug, "No matching control/variant seed pair is present in this snapshot.")
        fig, axes = self.canvas(title, height=max(8.4, 3.2+.66*len(rows)))
        sizes = []
        for ax, metric in zip(axes, METRICS):
            for i, (study, facet, level, label) in enumerate(rows):
                sub = data[(data.study == study) & (data.facet == facet) & (data.level == level) & (data.metric == metric)]
                if not len(sub):
                    self.label_box(ax, 111, i, "—")
                    continue
                row = sub.iloc[0]
                value, low, high = row.win_percent, row.ci_low, row.ci_high
                self.gradient_barh(ax, i, value-50, left=50, height=.48)
                if np.isfinite(low) and np.isfinite(high):
                    ax.plot([low, high], [i, i], lw=3.5, color="black", alpha=.65, zorder=4)
                    ax.plot([low, high], [i, i], ls="none", marker="|", ms=14, color="black", zorder=4)
                ax.scatter(value, i, s=115, color=self.cmap(.82 if value >= 50 else .18), edgecolor="black", lw=1., zorder=5)
                self.label_box(ax, 112, i, f"{value:.0f}%", fontsize=21)
                sizes.append(f"{study}/{facet}/{level}/{metric}: {row.problems} problems, {row.seed_pairs} pairs ({row.min_pairs}–{row.max_pairs}/problem)")
            ax.axvline(50, color="black", lw=1.2, ls="--", alpha=.8)
            ax.set_xlim(-3, 123)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_ylim(len(rows)-.45, -.6)
            ax.set_yticks(range(len(rows)), [r[3] for r in rows])
            ax.set_title(METRIC_NAMES[metric], weight="bold", pad=18)
            ax.set_xlabel("Control win probability (%)", weight="bold", labelpad=12)
            ax.grid(axis="y", visible=False)
        axes[1].tick_params(axis="y", labelleft=False)
        self.save(fig, number, slug,
                  "P(control better)+½P(tie), computed from matching seeds and initial-population hashes. "
                  "Equal weight per represented family. Lines are hierarchical 95% bootstrap intervals, resampling problems within families and seed pairs within problems. "
                  "Each contrast uses its own available matched panel; descriptive plots do not imply significance. "
                  "Significance tests remain separate and require all 30 planned pairs. Right of 50% favors the named control.", "\n".join(dict.fromkeys(sizes)))

    def heatmap(self, ax, values, labels_y, labels_x, texts=None, cmap="viridis", limits=(0, 100), fontsize=22):
        color = self.plt.get_cmap(cmap).copy()
        color.set_bad("#E2DED3")
        matrix = np.asarray(values, float)
        ax.imshow(np.ma.masked_invalid(matrix), cmap=color, vmin=limits[0], vmax=limits[1],
                  aspect="auto", interpolation="nearest", alpha=.85)
        ax.grid(False)
        ax.set_xticks(range(len(labels_x)), labels_x, rotation=42, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(labels_y)), labels_y)
        for (i, j), value in np.ndenumerate(matrix):
            text = texts[i][j] if texts is not None else (f"{value:.0f}" if np.isfinite(value) else "—")
            self.label_box(ax, j, i, text, fontsize=fontsize)

    def family(self, family_effects):
        sub = family_effects[family_effects.study == "main"] if len(family_effects) else family_effects
        if not len(sub):
            return self.skip(2, "families", "No main seed-paired outcome is present.")
        methods = [m for m in METHODS if m != "FULL" and m in set(sub.level)]
        fig, axes = self.canvas("FULL vs baselines across benchmark families", height=9.7, width=19, bottom=.27)
        for ax, metric in zip(axes, METRICS):
            g = sub[sub.metric == metric]
            matrix = g.pivot(index="family", columns="level", values="win_percent").reindex(index=FAMILIES, columns=methods).to_numpy()
            self.heatmap(ax, matrix, FAMILIES, [LABELS.get(m, m) for m in methods], fontsize=20)
            ax.set_title(METRIC_NAMES[metric], weight="bold", pad=18)
        axes[1].tick_params(axis="y", labelleft=False)
        self.save(fig, 2, "families", "Seed-paired FULL win probability (%) within each family, averaged across available matched problems. "
                  "50 denotes a balanced outcome, not equivalence. Brighter cells favor FULL; boxes contain measured probabilities. "
                  "Different contrasts may use different available panels. The binary GDE3 slot uses SPEA2 on ZDT5.")

    def architecture(self, data, number, slug, title, settings):
        rows = ("FULL vs\nno migration", "FULL vs\nelite exchange", "Default FULL vs\nthis FULL")
        matrices, details = {}, []
        for metric in METRICS:
            matrix = np.full((3, len(settings)), np.nan)
            for j, (label, study, facet, full_study, full_level) in enumerate(settings):
                for i, level in enumerate(("NO_MIG", "ELITE")):
                    g = data[(data.study == study) & (data.facet == facet) & (data.level == level) & (data.metric == metric)] if len(data) else data
                    if len(g):
                        matrix[i, j] = g.win_percent.iloc[0]
                        details.append(f"{label}/{rows[i].replace(chr(10), ' ')}/{metric}: {int(g.problems.iloc[0])} problems, {int(g.seed_pairs.iloc[0])} pairs")
                if full_level is None:
                    matrix[2, j] = 50.
                else:
                    g = data[(data.study == full_study) & (data.level == full_level) & (data.metric == metric)] if len(data) else data
                    if len(g):
                        matrix[2, j] = g.win_percent.iloc[0]
                        details.append(f"{label}/default FULL/{metric}: {int(g.problems.iloc[0])} problems, {int(g.seed_pairs.iloc[0])} pairs")
            matrices[metric] = matrix
        if not any(np.isfinite(m[:2]).any() or np.isfinite(m[2, 1:]).any() for m in matrices.values()):
            return self.skip(number, slug, "No matched architecture or policy contrast is present.")
        fig, axes = self.canvas(title, height=8.7, width=18.5, bottom=.25)
        for ax, metric in zip(axes, METRICS):
            texts = [["=" if i == 2 and settings[j][4] is None else (f"{v:.0f}" if np.isfinite(v) else "—")
                      for j, v in enumerate(row)] for i, row in enumerate(matrices[metric])]
            self.heatmap(ax, matrices[metric], rows, [s[0] for s in settings], texts, fontsize=24)
            ax.set_title(METRIC_NAMES[metric], weight="bold", pad=18)
        axes[1].tick_params(axis="y", labelleft=False)
        self.save(fig, number, slug, "Seed-paired win probability (%). Top two rows compare FULL against policies inside each setting. "
                  "Bottom row compares mixed K=4 FULL against FULL in the indicated setting; '=' is the same control, not an estimated effect. "
                  "Total population and FE budget are fixed. Different cells may use different available matched panels. Descriptive outcomes; see CSV for n.", "\n".join(details))

    def dimensions(self, data):
        sub = data[data.study == "dimensions"] if len(data) else data
        if not len(sub):
            return self.skip(7, "dimension", "No within-dimension matched pair is present.")
        methods = [m for m in DIM_METHODS if m != "FULL"]
        facets = ("Default", "D=50", "D=100")
        fig, axes = self.canvas("Robustness to decision-space dimension", height=10.3, bottom=.19)
        for ax, metric in zip(axes, METRICS):
            g = sub[sub.metric == metric]
            values = g.pivot(index="level", columns="facet", values="win_percent").reindex(index=methods, columns=facets).to_numpy()
            self.heatmap(ax, values, [LABELS.get(m, m) for m in methods], facets, fontsize=25)
            ax.set_title(METRIC_NAMES[metric], weight="bold", pad=18)
        axes[1].tick_params(axis="y", labelleft=False)
        self.save(fig, 7, "dimension", "FULL win probability (%) against each baseline, calculated separately within each dimension. "
                  "The default dimension is problem-specific. No pairing across different D is performed. "
                  "Averages give represented families equal weight. Available matched panels may differ across cells.")

    def many(self, cells):
        sub = cells[cells.study == "many"]
        methods = list(METHODS)
        problems = list(dict.fromkeys(sub.problem))
        for number, metric in ((8, "hv"), (9, "igd_plus")):
            g = sub[sub.metric == metric]
            if not len(g) or not (g.n > 0).any():
                self.skip(number, f"many_{metric}", "No measured M=5 outcomes are present.")
                continue
            values = g.pivot(index="problem", columns="level", values="median").reindex(index=problems, columns=methods).to_numpy()
            display = np.full_like(values, np.nan)
            for i, row in enumerate(values):
                valid = np.isfinite(row)
                x = np.log10(np.maximum(row[valid], 1e-12)) if metric == "igd_plus" else row[valid]
                if len(x):
                    scaled = (x-x.min())/(x.max()-x.min()) if x.max() > x.min() else np.full(len(x), .5)
                    display[i, valid] = 1-scaled if metric == "igd_plus" else scaled
            texts = [[self.scientific(v) for v in row] for row in values]
            fig, (ax,) = self.canvas(f"Five objectives · {METRIC_NAMES[metric]}", columns=1, width=18.5, height=9.5, bottom=.27)
            fig.subplots_adjust(left=.12)
            self.heatmap(ax, display, problems, [LABELS.get(m, m) for m in methods], texts, limits=(0, 1), fontsize=20)
            self.save(fig, number, f"many_{metric}", "Boxes contain actual per-problem seed medians, not completion counts or ranks. "
                      "Colors are scaled within each problem; brighter means better. IGD+ uses log color scaling with a display floor of 1e-12, while labels retain actual zeros. "
                      "All available completed seeds enter each median; sample sizes are in cells.csv. Missing outcomes remain grey. This is descriptive until the planned matrix is complete.")

    def convergence(self, data):
        if not len(data):
            return self.skip(10, "convergence", "No common seed panel has all seven methods and the exact required FE observations.")
        methods = [m for m in METHODS if m in set(data.method)]
        colors = {m: self.cmap(c) for m, c in zip(methods, np.linspace(.08, .88, len(methods)))}
        colors["FULL"] = self.cmap(.9)
        fig, axes = self.canvas("Search dynamics at identical evaluation budgets", height=10.2, top=.76, bottom=.18)
        for ax, metric, title, ylabel in zip(axes, ("hv_gain_percent", "igd_relative"),
                                            ("Hypervolume gain", "IGD+ reduction"),
                                            ("Within-problem HV gain (%)", "IGD+ / initial IGD+")):
            for method in methods:
                g = data[(data.method == method) & (data.metric == metric)].sort_values("fe")
                if not len(g):
                    continue
                x, y = g.fe.to_numpy(float), g.value.to_numpy(float)
                ax.plot(x, y, color=colors[method], marker="o", ms=7, markeredgecolor="black", markeredgewidth=.7,
                        lw=4.2 if method == "FULL" else 2.6, label=LABELS.get(method, method), zorder=4 if method == "FULL" else 3)
                ax.fill_between(x, g.ci_low.to_numpy(float), g.ci_high.to_numpy(float), color=colors[method], alpha=.10)
            ax.set_xscale("log")
            ax.set_xticks([400, 10000, 100000], ["400", "10k", "100k"])
            ax.set_xlabel("Objective evaluations", weight="bold", labelpad=12)
            ax.set_ylabel(ylabel, weight="bold", labelpad=12)
            ax.set_title(title, weight="bold", pad=18)
            full = data[(data.method == "FULL") & (data.metric == metric)].sort_values("fe")
            if len(full):
                row = full.iloc[-1]
                self.label_box(ax, .96, .9, f"FULL: {row.value:.1f}" + ("%" if metric == "hv_gain_percent" else ""),
                               transform=ax.transAxes, fontsize=21, ha="right")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.56, .925), ncol=4,
                   frameon=True, facecolor="white", edgecolor="black", fontsize=20)
        self.save(fig, 10, "convergence", "Same matched seeds, methods and problems at every actual FE point: 400, 10000, 25200, 50000, 100000. "
                  "HV gain is (HV(t)-this run's initial HV)/(best observed HV over this complete method/seed panel minus this run's initial HV). "
                  "This retrospective display normalization is never used by optimizers. IGD+ is divided by this run's initial value. "
                  "Zero-span/zero-initial cases are excluded from the corresponding display metric. "
                  "Seed medians are averaged with equal family weights; bands resample problems within families and are descriptive.")

    def costs(self, data):
        if not len(data):
            return self.skip(11, "migration_costs", "No migration-policy costs are present.")
        methods = [m for m in METHODS if m in set(data.method)]
        t = data.set_index("method")
        fig, axes = self.canvas("Migration activity and controller overhead", height=9.5, top=.80, bottom=.15)
        maximum = max(1., float(t.sent.max()))
        for i, method in enumerate(methods):
            row = t.loc[method]
            self.gradient_barh(axes[0], i-.15, row.sent, height=.24, c0=.1, c1=.65)
            self.gradient_barh(axes[0], i+.15, row.accepted, height=.24, c0=.5, c1=.95)
            self.label_box(axes[0], maximum*1.2, i, f"{row.sent:.0f} / {row.accepted:.0f}", fontsize=20)
            overhead = row.control+row.integration
            self.gradient_barh(axes[1], i, overhead, height=.5)
            self.label_box(axes[1], max(1., float((t.control+t.integration).max()))*1.28, i, f"{overhead:.1f}%", fontsize=22)
        axes[0].set_xlim(0, maximum*1.55)
        axes[1].set_xlim(0, max(1., float((t.control+t.integration).max()))*1.55)
        for ax, title, xlabel in zip(axes, ("Sent / accepted migrants", "Controller + integration"),
                                     ("Individuals per 10,000 FE", "Share of trial wall time (%)")):
            ax.set_yticks(range(len(methods)), [LABELS.get(m, m) for m in methods])
            ax.set_ylim(len(methods)-.6, -.65)
            ax.set_title(title, weight="bold", pad=18)
            ax.set_xlabel(xlabel, weight="bold", labelpad=12)
            ax.grid(axis="y", visible=False)
        axes[1].tick_params(axis="y", labelleft=False)
        self.save(fig, 11, "migration_costs", "Per-problem seed medians averaged with equal family weights, for all available main policy runs. "
                  "Each policy may have a different available panel; exact n and diagnostic fractions are exported to figure_costs.csv. "
                  "Left boxes show sent / accepted individuals. Payload is logical array traffic. "
                  "Time composition is descriptive under shared-server load and does not establish physical parallel speedup.")

    def descriptors(self, associations):
        if not len(associations):
            return self.skip(12, "state_descriptors", "Frozen FULL histories do not contain enough varying state/action observations.")
        features = ("H", "E", "free", "drift", "diversity", "progress")
        outcomes = ("migration_intensity", "acceptance", "hv_gain")
        per_problem = associations.groupby(["problem", "family", "feature", "outcome"]).spearman.median().reset_index()
        summary = per_problem.groupby(["family", "feature", "outcome"]).spearman.mean().reset_index().groupby(["feature", "outcome"]).spearman.mean().unstack()
        matrix = summary.reindex(index=features, columns=outcomes).to_numpy()
        texts = [[f"{x:+.2f}" if np.isfinite(x) else "—" for x in row] for row in matrix]
        fig, (ax,) = self.canvas("Population state, migration and search progress", columns=1, width=12.5, height=9.6, bottom=.25)
        fig.subplots_adjust(left=.30)
        self.heatmap(ax, matrix, ("Entropy H", "Expected energy E", "Free energy F", "JS drift", "Diversity", "HV progress"),
                     ("Migration\nintensity", "Accepted\nfraction", "Archive\nHV gain"), texts, limits=(-1, 1), fontsize=25)
        self.save(fig, 12, "state_descriptors", "Descriptive Spearman association within each completed main FULL run. "
                  "Descriptors are means over island profiles at each recorded step; outcomes use same-step migration increments and archive-HV increments. "
                  "Acceptance is undefined when nothing was sent. At least five varying paired observations are required. "
                  "Run correlations are reduced to per-problem seed medians and then equal-family means. "
                  "No time point is treated as an independent experimental run; no p-values or causal claims are made. Some inputs/outcomes are endogenous by construction.")

    def finish(self):
        write_csv(self.folder / "figure_index.csv", self.index)
        paragraphs = [f"# TherMOO research figures · {self.version}", "",
                      "Cream #FAF8F2, light cream panels, viridis gradients, black axes, large type, white value boxes with black borders. PNG 320 dpi; vector PDF.", "",
                      "Figures show scientific outcomes. Completion/remaining-work summaries are only in STATS. Incomplete samples are descriptive; inferential decisions still require 30 matched pairs.", ""]
        for row in self.index:
            paragraphs += [f"## {row['file']} · {row['status']}", "", row["caption"], "", row["note"], ""]
        (self.folder / "captions.md").write_text("\n".join(paragraphs))


def draw_research_figures(stats_dir, figures_dir, manifest, raw, views, observed, cells, bootstrap):
    pair_cases = plot_pair_cases(observed)
    effects = aggregate_plot_pairs(pair_cases, bootstrap)
    families = aggregate_plot_pairs(pair_cases, bootstrap, by_family=True)
    histories, associations = read_visual_history(stats_dir, raw)
    convergence, traces = publication_convergence(raw, histories, bootstrap)
    costs = descriptive_costs(raw)
    for name, frame in (("figure_pair_cases", pair_cases), ("figure_effects", effects), ("figure_family_effects", families),
                        ("figure_convergence", convergence), ("figure_convergence_runs", traces),
                        ("figure_costs", costs), ("figure_state_associations", associations)):
        write_csv(stats_dir/f"{name}.csv", frame)
    fig = Figures(figures_dir, manifest["version"], manifest["cutoff_ns"])
    fig.effects(effects, 1, "main_results", "FULL vs single solvers and migration baselines",
                [("main", "all", m, "GDE3 / SPEA2*" if m == "GDE3" else LABELS.get(m, m)) for m in METHODS[:10] if m != "FULL"])
    fig.family(families)
    ablations = ("conventional", "no_entropy", "no_free_energy", "no_js", "fixed_rate", "random_source", "boltzmann_only")
    short = ("Conventional state", "No entropy block", "No free energy", "No JS", "Fixed rate", "Random source", "Boltzmann only")
    fig.effects(effects, 3, "ablations", "Controller ablations · FULL vs reduced variants",
                [("ablation", "all", level, label) for level, label in zip(ablations, short)])
    sensitivity = [("interval", "5", "Interval: 5"), ("interval", "20", "Interval: 20"),
                   ("rho", "0.05", "Migration: 0.05"), ("rho", "0.2", "Migration: 0.20"),
                   ("temperature", "0.5", "Temperature: 0.5"), ("temperature", "2.0", "Temperature: 2.0"),
                   ("quality", "rank_hv", "Quality: HV"), ("quality", "rank_novelty", "Quality: novelty")]
    # CSV round trips can represent integer-valued parameter levels as '5.0'.
    if len(effects):
        for study in ("sensitivity_interval", "sensitivity_rho", "sensitivity_temperature"):
            mask = effects.study == study
            effects.loc[mask, "level"] = effects.loc[mask, "level"].map(lambda x: format(float(x), ".12g"))
    fig.effects(effects, 4, "sensitivity", "One-factor sensitivity · default vs variation",
                [("sensitivity_"+s, "all", format(float(v), ".12g") if s != "quality" else v, label) for s, v, label in sensitivity])
    fig.architecture(effects, 5, "composition", "Optimizer composition and migration policy",
                     [("Mixed", "homogeneous", "mixed", "composition_FULL", None)] +
                     [(LABELS.get(c, c), "homogeneous", c, "composition_FULL", c) for c in ("NSGA2", "MOEAD", "GDE3", "SMS")])
    fig.architecture(effects, 6, "islands", "Island count under a fixed population and FE budget",
                     [(f"K={k}", "islands", f"K={k}", "islands_FULL", None if k == 4 else f"K={k}") for k in (4, 8, 16)])
    fig.dimensions(effects)
    fig.many(cells)
    fig.convergence(convergence)
    fig.costs(costs)
    fig.descriptors(associations)
    fig.finish()
    return fig.index


def provenance_archive(root, stats_dir, manifest):
    """Save the implementation and parameter protocol without duplicating populations."""
    paths = [Path(__file__).resolve()]
    paths += sorted((root / "thermoo").glob("*.py"))
    paths += [root / n for n in ("README.md", "LICENSE", "reports/download_manifest.json",
                                "build/thermoo/source_hashes.json")]
    paths += sorted((root / "configs").glob("*"))
    hashes = []
    with zipfile.ZipFile(stats_dir / "provenance.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            if not path.is_file():
                continue
            name = "analyzer/thermoo_snapshot.py" if path == Path(__file__).resolve() else str(path.relative_to(root))
            blob = path.read_bytes()
            archive.writestr(name, blob)
            hashes.append(dict(file=name, bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest()))
        archive.writestr("analysis_protocol.json", json.dumps(safe_json(manifest), indent=2))
    return hashes


PROTOCOL = """# TherMOO snapshot protocol

This is a versioned interim snapshot, not a new experiment or a final acceptance decision.
It never launches an optimizer, changes a seed, truncates a budget, or stops workers.

## Snapshot and provenance

The cutoff is the start time of the Python analyzer, before heavy imports. Only
planned full-budget results, committed before this cutoff, enter the snapshot.
Both modification and change times are checked; an atomic rename published
after the cutoff is excluded. The active experiment fingerprint is computed by
the original engine. Other fingerprints, smoke trials and timing pilots are
never combined. Each paired STATS/vN and FIGURES/vN reservation is atomic.
Failed/interrupted snapshot versions are retained and never overwritten.

The exact accepted result JSON, histories, migration events and available
scheduler sidecars are retained in gzip JSONL files, with source byte hashes.
Population NPZ files are checked for existence and publication time but not
copied, decompressed, or re-scored. This report analyzes recorded metrics.
Missing auxiliary logs do not discard an otherwise valid result; they remove
that trial only from analyses which need those logs. Raw populations and
reference fronts remain in the experiment directory.

## Experimental levels

- main: all 61 functions and ten planned solver/policy slots.
- ablation: seven variants against the same main FULL control, ten-function panel.
- sensitivity: one factor at a time; interval, rho, temperature, and quality.
- homogeneous: three migration policies within each optimizer composition,
  plus the original mixed-island controls; FULL-only composition contrasts.
- islands: policies compared separately at K=4, 8, 16; FULL-only K contrasts.
- dimensions: comparisons within each D setting, including the default-D main
  controls. Default D is problem-specific. D variants are never pooled as
  independent benchmark problems.
- many: six M=5 problems and twelve methods, including NSGA-III and RVEA.

Reused controls retain their original run IDs and are not counted as additional
independent experiments. Planned parameters and actual island optimizers are
exported. Standalone methods use one population; Job.islands denotes the
shared experimental configuration, not their actual island count.
The binary ZDT5 GDE3 slot actually uses SPEA2. ZDT5 is retained in its own
cells and paired comparisons, but excluded from cross-problem ranks, omnibus
tests, convergence ranks, and matched timing summaries. GDE3* flags this
substitution in the main paired-comparison figure.

## Descriptive summaries and intervals

Every cell reports its available and planned n, mean, sample SD, median, IQR,
raw-sample Shapiro diagnostic and a percentile bootstrap 95% interval for the
mean. Intervals with n<30 are descriptive and explicitly interim. No interval
is claimed with fewer than two observations. Constant samples are flagged,
not assigned an invented normality p-value. Values are never clipped or
converted to invalid percentage improvements when HV=0.

## Paired inference

Comparisons use the intersection of the exact planned seeds and require equal
initial-population hashes. Positive advantage always favors the control:
HV_control - HV_variant, or IGD+_variant - IGD+_control.
Inferential tests require all 30 planned seed pairs. Fewer pairs still receive
descriptive paired effects; they receive no inferential decision.

The primary test is a two-sided sign-flip permutation test of the mean paired
difference (50,000 resamples by default, plus-one Monte Carlo correction).
Sign flipping assumes exchangeability/symmetry of paired differences under
the null; it is not justified solely by a failed normality test. Seed-paired
bootstrap intervals use 4,000 resamples by default. Raw per-metric values and
paired-difference normality diagnostics are both retained. A paired t test
when difference normality is not rejected, a signed-rank sensitivity test
when symmetry diagnostics permit it, and an exact sign-test sensitivity
result are separate columns. The sign test requires no symmetry assumption.

Holm correction covers every planned contrast and both endpoints within its
experimental block, including auxiliary composition/K views. Unfinished
hypotheses are conservatively treated as p=1, so a cheap early subset never
shrinks the planned multiplicity family. Sign-test sensitivity results have
their own analogous Holm correction. Repeated snapshots are descriptive
monitoring: interim p-values do not license optional stopping or early claims.
Final manuscript conclusions should use the predeclared complete matrix.

The seed-paired win probability is P(control better)+0.5*P(tie), not the
unpaired Vargha-Delaney A12 effect. Paired rank-biserial effects are also
reported. A nonsignificant difference is not evidence of equivalence.

## Cross-problem summaries

Ranks use per-problem seed medians and only cases with 30 matched seeds for
every planned level within a fixed study/facet. Missing algorithms cannot
improve another algorithm's rank. Represented families receive equal weight.
95% rank/effect intervals resample problems within each family; they do not
resample time points or pretend that every seed is a new benchmark problem.
These intervals are conditional on the represented benchmark families.

Omnibus tests permute method labels within independent problem rows of the
rank matrix (Friedman-type statistic). This handles small panels without
relying on a chi-square approximation. The reported statistic is the
uncorrected Friedman statistic; ties are retained in the permutation null.
Omnibus Holm correction includes the predeclared study/facet/endpoints.
Post-hoc control contrasts use paired permutations of problem-level ranks,
Holm correction for the planned block, and the adjusted omnibus gate.
Substudies with fewer than three independent problem rows have no omnibus
or post-hoc significance decision. Composition, K and D facets stay separate.

No raw HV/IGD+ grand average is taken across different objective spaces.
Compact ablation/sensitivity/composition figures use matched problem panels
and family-balanced paired win probabilities; their intervals are descriptive.

## Convergence, migration and runtime

Convergence uses a predefined seven-method subset on the same complete problem
panel and exactly observed FE = 400, 10000, 25200, 50000, 100000. There is no
interpolation, and nominal 25000 is not substituted for measured 25200.
The main full-budget archive metrics and source histories remain available.

Migration tables distinguish sent and accepted individuals, logical array
payload, and controller/integration/diagnostic time. Per-problem seed medians
are averaged with equal family weights. Timing is descriptive: jobs share a
server and priorities/load vary. CPU seconds are available only where a
completed scheduler sidecar recorded them; missing measurements remain NA.
Logical payload is not measured MPI/network traffic. Islands execute
sequentially inside each trial; K is not a physical parallel-speedup test.

## Figures

At most twelve figure pairs (PNG 320 dpi and vector PDF), with cream panels,
viridis, black axes and large fonts. A figure lacking its required data is
listed as not_ready in figure_index.csv instead of inventing results. Detailed
problem/seed cells are tables, not hundreds of separate figures. Five-objective
heatmaps explicitly show n/30 and remain descriptive until all cells finish.
All algorithm operators, parameter defaults and native survival details are
in the captured README.md and thermoo/*.py inside provenance.zip. Every
planned Job parameter is also in planned_runs.csv and per_run.csv.
"""


def build_report(stats_dir, figures_dir, manifest, coverage, comparisons, ranks, issues):
    lines = [f"# TherMOO · {manifest['version']}", "",
             f"Срез: **{manifest['cutoff_utc']}**. Fingerprint: `{manifest['fingerprint']}`.", "",
             f"Готово: **{manifest['accepted_trials']:,} / {manifest['planned_trials']:,}**. "
             f"Осталось на момент среза: **{manifest['planned_trials']-manifest['accepted_trials']:,}**.", "",
             "Это промежуточный срез. Новые результаты после времени среза попадут в следующую версию.", "",
             "## Покрытие", "", "| Блок | Готово | Запланировано | Доля |", "|---|---:|---:|---:|"]
    for block in BLOCKS:
        g = coverage[coverage.block == block]
        n, total = int(g.completed.sum()), int(g.planned.sum())
        lines.append(f"| {block} | {n:,} | {total:,} | {100*n/total:.1f}% |")
    eligible = int(comparisons.eligible_inference.sum()) if len(comparisons) else 0
    lines += ["", "## Как читать результаты", "",
              f"Доступны {eligible:,} парных проверок по метрикам с 30 совпадающими seed и одинаковой начальной популяцией. "
              "Во всех остальных строках — только описательная статистика.", "",
              "Положительный advantage означает преимущество указанного control для обеих метрик. "
              "`p_holm` учитывает оба показателя и все заранее запланированные сравнения блока. "
              "`no_difference_detected` не означает эквивалентность.", "",
              "Общие ранги строятся только по задачам, где завершились все сравниваемые методы. "
              "Семейства имеют одинаковый вес. Неполные семейства и панели отмечены в таблицах. "
              "ZDT5 анализируется отдельно от общих рангов: вместо GDE3 там работает SPEA2.", "",
              "## Файлы", "",
              "| Файл | Содержание |", "|---|---|",
              "| per_run.csv | Все принятые прогоны, параметры, HV, IGD+, мигранты, время, реальные алгоритмы |",
              "| planned_runs.csv, coverage.csv, missing_runs.csv | Полная матрица, покрытие и недостающие прогоны |",
              "| cells.csv | Все уровни анализа: n, среднее, SD, медиана, IQR, 95% CI, нормальность |",
              "| comparisons.csv | Парные эффекты, CI, нормальность разностей, permutation/t/Wilcoxon/sign, Holm |",
              "| ranks.csv, rank_cases.csv, rank_eligibility.csv | Общие/семейные ранги и допуск каждой задачи |",
              "| family_ranks.csv | Ранги отдельно по семействам, число завершённых и запланированных задач |",
              "| omnibus.csv, posthoc.csv | Перестановочный Friedman и скорректированные post-hoc сравнения |",
              "| effects.csv | Сводные эффекты абляций, параметров, составов, K |",
              "| convergence.csv, convergence_cases.csv | Ранги на одинаковых фактических FE |",
              "| migration_timing.csv | Отправленные/принятые мигранты, логический трафик и расходы времени |",
              "| issues.csv, source_files.csv | Исключения, доступность логов и хеши исходных результатов |",
              "| snapshot/*.jsonl.gz | Точные копии принятых JSON и вспомогательных логов |",
              "| provenance.zip, PROTOCOL.md, manifest.json | Код, протокол, параметры и воспроизводимость |", "",
              f"Рисунки: `{figures_dir}`. Описание и готовность каждого рисунка: `figure_index.csv`, `captions.md`.", "",
              f"Создано рисунков: **{manifest.get('created_figures', 0)} из 12**. "
              "Отсутствующий рисунок означает недостаточное покрытие для его панели, а не нулевой эффект.", ""]
    if len(ranks):
        lines += ["## Готовность общих рангов", "", "| Анализ | Фасет | Полных задач | Запланировано | Семейств |", "|---|---|---:|---:|---:|"]
        for row in ranks[ranks.metric == "hv"].drop_duplicates(["study", "facet"]).itertuples():
            lines.append(f"| {row.study} | {row.facet} | {row.cases} | {row.planned_cases} | {row.families} |")
        lines.append("")
    if issues:
        lines += ["## Проверка источников", "", f"Записей аудита: **{len(issues)}**; причины и файлы перечислены в `issues.csv`.", ""]
    lines += ["## Ограничения среза", "",
              "Время отражает общую загрузку сервера и не доказывает ускорение алгоритма. "
              "Острова внутри одного прогона выполняются последовательно. "
              "Не используйте промежуточные p-value для остановки эксперимента или выбора выгодной версии. "
              "Итоговые выводы статьи следует делать по полной заранее заданной матрице.", ""]
    if manifest.get("source_roots"):
        lines += ["## Пути чтения и сохранения", "",
                  f"Код и конфигурация эксперимента: `{manifest.get('engine_root', '')}`.", "",
                  "Источники результатов: " + ", ".join(f"`{p}`" for p in manifest["source_roots"]) + ".", "",
                  f"Все новые отчёты и рисунки: `{manifest['root']}`.", "",
                  "Один run_id учитывается один раз. Разные fingerprint не объединяются; конфликтующие копии исключаются с записью в issues.csv.", ""]
    (stats_dir / "REPORT.md").write_text("\n".join(lines))


FIGURE_PROTOCOL = """# Research figure protocol · version 2

Completion/remaining-work charts are excluded from FIGURES. Coverage remains in
STATS/coverage.csv and REPORT.md. Every exported figure describes measured
optimization quality, a controlled variation, search dynamics, measured cost,
or recorded state/action association.

The rendering layer is descriptive and uses every available valid seed pair
for each control/variant/case. It requires equal initial-population hashes but
does not discard a completed sample just because another method has fewer than
30 seeds. Each contrast has its own matched panel; panels can differ. Exact
problem, family and pair counts are exported in figure_pair_cases.csv,
figure_effects.csv, figure_family_effects.csv and the captions. This layer does
not change the 30-pair inference gate, Holm families, or frozen test results.

Main, ablation and sensitivity plots show P(control better)+0.5*P(tie). Represented
families receive equal weight. Descriptive 95% percentile intervals use a
hierarchical bootstrap: problems within families and seed pairs within each
problem. Bootstrap intervals can collapse at empirical boundaries; they are
not equivalence tests or evidence of certain population probabilities.
No significance stars are added to incomplete descriptive panels.

Architecture figures compare policies within each fixed setting and show the
mixed K=4 FULL control against FULL at the alternative composition/K. Dimension
contrasts are paired only within the same D. Default D is problem-specific.
The M=5 matrices label actual per-cell metric medians, with independent per-row
color scaling; colors are not a common raw metric scale. IGD+ color scaling uses
a 1e-12 display floor; actual numeric zeros are retained.

Convergence uses seven predefined methods with matching initial populations,
the same available seeds/problem panel at all five actual FE points, and no
interpolation. HV display progress is (HV(t)-initial HV)/(best observed HV over
the complete compared method/seed panel minus initial HV). This is retrospective
plot scaling, never an optimization input. IGD+ is divided by its initial value.
Zero-denominator cases are omitted for that display metric. Per-problem seed
medians receive equal family weights. Bands bootstrap problems within families.

Migration costs use available completed main policy runs. Per-problem seed
medians receive equal family weights; method panels may differ. Control plus
integration is a measured share of wall time. Shared-server load prevents
interpreting these descriptive measurements as physical parallel speedup.

Descriptor associations are Spearman coefficients calculated within a completed
main FULL trial from recorded island-mean H, E, F, JS drift, diversity and HV
progress against same-step migration intensity, accepted fraction and archive
HV gain. At least five varying finite observation pairs are needed. No-sending
steps have undefined accepted fraction. Within-run coefficients become
per-problem seed medians and then equal-family means. There are no p-values,
independent-timepoint claims, or causal claims; some quantities are endogenous.

Presentation follows the fractional-lora-unlearning template: cream #FAF8F2,
light #F4F1E8 panels, viridis gradients with transparency, black axes/ticks,
bold labels and white rounded numeric bbox with black borders. Text is 20–30 pt
depending on density. Dates, progress ratios and diagnostic prose are outside
the figure canvas. PNG is 320 dpi; PDF is vector with embedded TrueType fonts.
"""


def read_table(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame()


def replot_snapshot(root, source_version, bootstrap):
    """Make a new paired version from frozen inputs; no optimizer imports/tests."""
    candidates = []
    for directory in (root / "STATS").glob("v*"):
        if re.fullmatch(r"v[1-9][0-9]*", directory.name) and (directory/"manifest.json").is_file():
            value = json.loads((directory/"manifest.json").read_text())
            if value.get("status") == "COMPLETE" and (directory/"per_run.csv").is_file():
                candidates.append(directory)
    if source_version == "latest":
        if not candidates:
            raise RuntimeError(f"No complete frozen STATS/vN snapshot in {root}")
        source = max(candidates, key=lambda p: int(p.name[1:]))
    else:
        if not re.fullmatch(r"v[1-9][0-9]*", source_version):
            raise ValueError("--replot requires 'latest' or a version such as v1")
        source = root / "STATS" / source_version
        if source not in candidates:
            raise RuntimeError(f"Snapshot is missing or incomplete: {source}")
    original = json.loads((source/"manifest.json").read_text())
    version, stats_dir, figures_dir = allocate_version(root)
    manifest = dict(original, status="RUNNING", version=version, root=str(root),
                    statistics=str(stats_dir), figures=str(figures_dir), analyzer_version=VERSION,
                    analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    source_snapshot=str(source), replot_started_utc=datetime.now(timezone.utc).isoformat(),
                    figure_protocol="descriptive_available_matched_pairs_v2")
    atomic_json(stats_dir/"manifest.json", manifest)
    log(f"[REPLOT] {source.name} -> {version}; frozen results and inference tables are reused")
    log(f"[STATS] {stats_dir}")
    log(f"[FIGURES] {figures_dir}")
    try:
        try:
            os.nice(10)
        except OSError:
            pass
        for path in source.iterdir():
            if path.name in ("manifest.json", "REPORT.md") or path.name.startswith("figure_"):
                continue
            destination = stats_dir/path.name
            if path.is_dir():
                shutil.copytree(path, destination)
            else:
                shutil.copy2(path, destination)
        atomic_json(stats_dir/"source_manifest.json", original)
        raw = read_table(stats_dir/"per_run.csv", dtype={"run_id": str, "initial_hash": str})
        planned = read_table(stats_dir/"planned_runs.csv", dtype={"run_id": str})
        if not len(planned):
            raise RuntimeError("Frozen planned_runs.csv is missing or empty")
        views = make_views(planned.to_dict("records"))
        if len(raw):
            extra = [c for c in raw if c not in views]
            observed = views.merge(raw[["run_id"]+extra], on="run_id", how="inner", validate="many_to_one")
        else:
            observed = pd.DataFrame(columns=list(views.columns)+["hv", "igd_plus", "initial_hash"])
        cells = read_table(stats_dir/"cells.csv", dtype={"level": str})
        if not len(cells):
            cells = cell_statistics(views, observed, bootstrap)
            write_csv(stats_dir/"cells.csv", cells)
        index = draw_research_figures(stats_dir, figures_dir, manifest, raw, views, observed, cells, bootstrap)
        manifest.update(created_figures=sum(r["status"] == "created" for r in index), figure_slots=12,
                        status="COMPLETE", figure_bootstrap=bootstrap,
                        finished_utc=datetime.now(timezone.utc).isoformat())
        with zipfile.ZipFile(stats_dir/"provenance.zip", "a", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(f"analyzer/thermoo_snapshot_{VERSION}.py", Path(__file__).read_bytes())
        (stats_dir/"FIGURE_PROTOCOL.md").write_text(FIGURE_PROTOCOL)
        issues = read_table(stats_dir/"issues.csv").to_dict("records")
        build_report(stats_dir, figures_dir, manifest, read_table(stats_dir/"coverage.csv"),
                     read_table(stats_dir/"comparisons.csv"), read_table(stats_dir/"ranks.csv"), issues)
        with (stats_dir/"REPORT.md").open("a") as report:
            report.write("\n## Рисунки версии 2\n\nПоказаны доступные измеренные результаты. Диаграммы выполнения исключены. "
                         "Объём выборок и правила сравнений — в FIGURE_PROTOCOL.md, figure_*.csv и подписях. "
                         "Исходный срез и статистические тесты сохранены без пересчёта.\n")
        atomic_json(stats_dir/"manifest.json", manifest)
        log(f"[DONE] {version}: research figures={manifest['created_figures']}/12; source={source.name}")
        return 0
    except BaseException as error:
        manifest.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED", error=str(error), traceback=traceback.format_exc())
        atomic_json(stats_dir/"manifest.json", manifest)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/tahiti/TherMOO"))
    parser.add_argument("--source-root", action="append", type=Path, help="Input experiment root; repeat to merge compatible roots")
    parser.add_argument("--bootstrap", type=int, default=4000, help="Bootstrap resamples; production default 4000")
    parser.add_argument("--permutations", type=int, default=50000, help="Permutation resamples; production default 50000")
    parser.add_argument("--replot", nargs="?", const="latest", help="Redraw latest or vN frozen snapshot, reusing statistics")
    args = parser.parse_args(argv)
    if args.bootstrap < 200 or args.permutations < 1000:
        parser.error("At least 200 bootstrap and 1000 permutation resamples are required")
    root = args.root.expanduser().resolve()
    if args.replot is not None:
        return replot_snapshot(root, args.replot, args.bootstrap)
    source_roots, engine_root = discover_input_roots(root, args.source_root)
    os.environ["THERMOO_ROOT"] = str(engine_root)
    sys.path.insert(0, str(engine_root))
    version, stats_dir, figures_dir = allocate_version(root)
    manifest = dict(status="RUNNING", version=version, analyzer_version=VERSION,
                    analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    cutoff_ns=START_NS, cutoff_utc=datetime.fromtimestamp(START_NS / 1e9, timezone.utc).isoformat(),
                    root=str(root), engine_root=str(engine_root), source_roots=[str(p) for p in source_roots],
                    statistics=str(stats_dir), figures=str(figures_dir),
                    protocol=dict(bootstrap=args.bootstrap, permutations=args.permutations, alpha=.05,
                                  minimum_paired_runs=30, inference="paired_sign_flip_mean", correction="Holm_planned_block_both_endpoints"),
                    python=sys.version, dependencies={})
    atomic_json(stats_dir / "manifest.json", manifest)
    log(f"[ENGINE] {engine_root}")
    for input_root in source_roots:
        log(f"[READ] {input_root / 'outputs/full'}")
    log(f"[WRITE ONLY] {root}")
    log(f"[VERSION] {version}")
    log(f"[CUTOFF] {manifest['cutoff_utc']}")
    log(f"[STATS] {stats_dir}")
    log(f"[FIGURES] {figures_dir}")
    try:
        try:
            os.nice(10)
        except OSError:
            pass
        from thermoo.plan import jobs as generate_jobs, family
        from thermoo.engine import experiment_fingerprint
        fingerprint = experiment_fingerprint()
        planned_jobs = list(generate_jobs())
        if len({j.signature() for j in planned_jobs}) != len(planned_jobs):
            raise ValueError("The full plan contains duplicate signatures")
        planned = [normalize_job(job, family) for job in planned_jobs]
        manifest.update(fingerprint=fingerprint, planned_trials=len(planned))
        for package in ("numpy", "scipy", "pandas", "matplotlib", "moocore", "autograd"):
            try:
                manifest["dependencies"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                manifest["dependencies"][package] = None
        atomic_json(stats_dir / "manifest.json", manifest)
        log(f"[PLAN] fingerprint={fingerprint} trials={len(planned):,}")
        write_csv(stats_dir / "planned_runs.csv", planned)
        raw, histories, issues, missing, provenance = capture(engine_root, fingerprint, planned_jobs, family, stats_dir, START_NS, source_roots=source_roots)
        manifest.update(accepted_trials=len(raw), missing_trials=len(missing),
                        accepted_by_source=raw.source_root.value_counts().to_dict() if len(raw) else {})
        log(f"[SNAPSHOT] frozen={len(raw):,}/{len(planned):,}; excluded/audit={len(issues)}")
        write_csv(stats_dir / "per_run.csv", raw, ["run_id", "block", "problem", "method", "seed", "hv", "igd_plus"])
        write_csv(stats_dir / "missing_runs.csv", missing, ["run_id", "reason"])
        write_csv(stats_dir / "source_files.csv", provenance, ["run_id", "file", "sha256"])
        coverage = coverage_table(planned, raw)
        write_csv(stats_dir / "coverage.csv", coverage)
        views = make_views(planned)
        # Merge on immutable IDs only. A main control may appear in several views.
        if len(raw):
            extra = [c for c in raw.columns if c not in views.columns]
            observed = views.merge(raw[["run_id"] + extra], on="run_id", how="inner", validate="many_to_one")
        else:
            observed = pd.DataFrame(columns=list(views.columns) + ["hv", "igd_plus", "initial_hash"])
        log(f"[ANALYZE] cells, paired inference, ranks; bootstrap={args.bootstrap}, permutations={args.permutations}")
        cells = cell_statistics(views, observed, args.bootstrap)
        write_csv(stats_dir / "cells.csv", cells)
        comparisons = pair_comparisons(views, observed, args.bootstrap, args.permutations, issues)
        write_csv(stats_dir / "comparisons.csv", comparisons)
        rank_cases, ranks, eligibility, omnibus, posthoc = balanced_ranks(views, observed, args.bootstrap, args.permutations)
        for name, frame in (("rank_cases", rank_cases), ("ranks", ranks), ("rank_eligibility", eligibility), ("omnibus", omnibus), ("posthoc", posthoc)):
            write_csv(stats_dir / f"{name}.csv", frame)
        if len(rank_cases):
            family_ranks = rank_cases.groupby(["study", "facet", "family", "metric", "level"], sort=False).agg(
                mean_rank=("rank", "mean"), median_rank=("rank", "median"), completed_problems=("case", "nunique")).reset_index()
            planned_families = eligibility[~eligibility.excluded_binary].groupby(["study", "facet", "family"]).case.nunique().rename("planned_problems").reset_index()
            family_ranks = family_ranks.merge(planned_families, on=["study", "facet", "family"], validate="many_to_one")
        else:
            family_ranks = pd.DataFrame()
        write_csv(stats_dir / "family_ranks.csv", family_ranks)
        effects = paired_effects(comparisons, args.bootstrap)
        write_csv(stats_dir / "effects.csv", effects)
        convergence, convergence_cases = convergence_table(views, observed, histories, args.bootstrap)
        write_csv(stats_dir / "convergence.csv", convergence)
        write_csv(stats_dir / "convergence_cases.csv", convergence_cases)
        timing = migration_timing(raw, rank_cases)
        write_csv(stats_dir / "migration_timing.csv", timing)
        write_csv(stats_dir / "issues.csv", issues, ["run_id", "file", "reason", "severity"])
        figure_index = draw_research_figures(stats_dir, figures_dir, manifest, raw, views, observed, cells, args.bootstrap)
        manifest.update(created_figures=sum(row["status"] == "created" for row in figure_index),
                        figure_slots=12, audit_records=len(issues),
                        eligible_paired_tests=int(comparisons.eligible_inference.sum()))
        (stats_dir / "PROTOCOL.md").write_text(PROTOCOL)
        (stats_dir / "FIGURE_PROTOCOL.md").write_text(FIGURE_PROTOCOL)
        manifest["provenance_files"] = provenance_archive(engine_root, stats_dir, manifest)
        manifest.update(status="COMPLETE", finished_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=(time.time_ns() - START_NS) / 1e9,
                        analysis_complete_matrix=len(raw) == len(planned))
        atomic_json(stats_dir / "manifest.json", manifest)
        build_report(stats_dir, figures_dir, manifest, coverage, comparisons, ranks, issues)
        log(f"[DONE] {version}: frozen={len(raw):,}/{len(planned):,}, figures={manifest['created_figures']}/12")
        log(f"[REPORT] {stats_dir / 'REPORT.md'}")
        return 0
    except BaseException as error:
        manifest.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED",
                        error=str(error), traceback=traceback.format_exc(),
                        finished_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(stats_dir / "manifest.json", manifest)
        log(f"[FAILED] {version}: {error}; details in {stats_dir / 'manifest.json'}")
        raise


if __name__ == "__main__":
    sys.exit(main())
