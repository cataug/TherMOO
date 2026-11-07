#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""TherMOO deep scientific atlas v4, with immutable dual-source data intake.

Optional --trace-panel runs a separate supplementary mechanistic study.
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

VERSION = "4.0.0"
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
    global LAST_ALLOCATED_VERSION
    LAST_ALLOCATED_VERSION = (version, stats_dir, figures_dir)
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


def snapshot_main(argv=None):
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



"""Deep research atlas extension. Included after the immutable V3 intake core."""
import sqlite3
import zlib
import inspect
from functools import lru_cache
from contextlib import contextmanager
from dataclasses import fields as dataclass_fields

ATLAS_VERSION = "4.0.0"
STATE_FEATURES = ("H", "E", "free", "drift", "diversity", "progress", "nd_fraction", "temperature", "dH", "dF", "dDiv")
TRACE_PANEL = ("ZDT1", "DTLZ2", "WFG9", "LZ09_F1", "UF8", "ZCAT20")
FE_GRID = (400, 10000, 25200, 50000, 100000)


def atlas_slug(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def cohort_name(row):
    return f"{row['problem']}_M{int(row['m'])}_D{int(row['d'])}"


def actual_label(method, problem=""):
    return "SPEA2" if method == "GDE3" and problem == "ZDT5" else LABELS.get(method, method)


def pca_coordinates(matrix, components=3, fit=None, standardize=True):
    x = np.asarray(matrix, float)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("PCA needs a finite two-dimensional matrix with at least two rows")
    if fit is None:
        mean = x.mean(axis=0)
        scale = x.std(axis=0) if standardize else np.ones(x.shape[1])
        scale = np.where(scale > 1e-12, scale, 1.)
        z = (x-mean)/scale
        _, singular, basis = np.linalg.svd(z, full_matrices=False)
        total = float(np.sum(singular**2))
        explained = singular**2/total if total > 0 else np.zeros(len(singular))
        axes = basis[:min(components, len(basis))].T
        for j in range(axes.shape[1]):
            if axes[np.argmax(np.abs(axes[:, j])), j] < 0:
                axes[:, j] *= -1
        fit = dict(mean=mean, scale=scale, axes=axes, explained=explained[:axes.shape[1]])
    coordinates = (x-fit["mean"])/fit["scale"] @ fit["axes"]
    if coordinates.shape[1] < components:
        coordinates = np.pad(coordinates, ((0, 0), (0, components-coordinates.shape[1])))
    return coordinates, fit


def joint_embedding(matrix, cache, key, max_points=1600, seed=17, standardize=True, metric='euclidean'):
    """Fit one common map; quantify neighborhood preservation and restart stability."""
    from sklearn.manifold import TSNE, trustworthiness
    from sklearn.neighbors import NearestNeighbors
    x = np.asarray(matrix, float)
    if len(x) < 8:
        raise ValueError("At least eight points are required for t-SNE")
    if not np.isfinite(x).all() or np.max(np.ptp(x,axis=0)) <= 1e-12:
        raise ValueError('No finite geometric variation for an informative t-SNE map')
    if len(x) > max_points:
        raise ValueError("Sample explicitly before the common t-SNE fit")
    scale=np.where(x.std(axis=0)>1e-12,x.std(axis=0),1.)
    reference=(x-x.mean(axis=0))/scale if standardize and metric!='hamming' else x
    z, _ = pca_coordinates(reference, components=min(50, x.shape[1], len(x)-1), standardize=False) if metric != 'hamming' else (x,None)
    digest = hashlib.sha256(x.tobytes()+repr((key, seed, x.shape, ATLAS_VERSION, standardize, metric,
                                            importlib.metadata.version('scikit-learn'),np.__version__,importlib.metadata.version('scipy'),
                                            'original_space_audit_random_restart')).encode()).hexdigest()
    path = cache/f"tsne_{digest}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            return data["coordinates"], json.loads(str(data["audit"]))
    perplexity = min(30., max(2., (len(x)-1)/4))
    params = dict(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed, metric=metric,
                  method="barnes_hut", angle=.5, n_jobs=1)
    parameter = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
    params[parameter] = 1000
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        first = TSNE(**params).fit_transform(z)
        second = TSNE(**dict(params, random_state=seed+37, init='random')).fit_transform(z)
    neighbors = min(10, max(1, (len(x)-1)//3))
    a = NearestNeighbors(n_neighbors=neighbors).fit(first).kneighbors(return_distance=False)
    b = NearestNeighbors(n_neighbors=neighbors).fit(second).kneighbors(return_distance=False)
    jaccard = np.mean([len(set(i) & set(j))/len(set(i) | set(j)) for i, j in zip(a, b)])
    audit = dict(points=len(x), input_dimensions=x.shape[1], perplexity=perplexity, seed=seed,
                 repeat_seed=seed+37, iterations=1000, neighbors=neighbors,
                 metric=metric, init='pca', repeat_init='random',
                 trustworthiness=float(trustworthiness(reference, first, n_neighbors=neighbors,metric=metric)),
                 repeat_trustworthiness=float(trustworthiness(reference, second, n_neighbors=neighbors,metric=metric)),
                 restart_neighbor_jaccard=float(jaccard), input_sha256=hashlib.sha256(x.tobytes()).hexdigest())
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, coordinates=first, audit=json.dumps(audit))
    return first, audit


def empirical_attainment(fronts, grid_x, grid_y):
    """Probability that at least one point weakly dominates each two-objective grid point."""
    if not fronts:
        return np.full((len(grid_y), len(grid_x)), np.nan)
    count = np.zeros((len(grid_y), len(grid_x)))
    for front in fronts:
        f = np.asarray(front, float)
        f = f[np.isfinite(f).all(axis=1)]
        if not len(f):
            continue
        order = np.argsort(f[:, 0], kind="stable")
        xs, ys = f[order, 0], np.minimum.accumulate(f[order, 1])
        idx = np.searchsorted(xs, grid_x, side="right")-1
        threshold = np.where(idx >= 0, ys[np.maximum(idx, 0)], np.inf)
        count += np.asarray(grid_y)[:, None] >= threshold[None, :]
    return count/len(fronts)


class FrozenHistory:
    def __init__(self, snapshot_dir, target):
        self.path = target/"frozen_history.sqlite"
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("CREATE TABLE IF NOT EXISTS logs(run_id TEXT PRIMARY KEY, history BLOB, migration BLOB)")
        self.connection.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT)")
        self.index_issues=[]
        for kind, column in (("histories", "history"), ("migrations", "migration")):
            source = snapshot_dir/f"{kind}.jsonl.gz"
            if not source.is_file():
                continue
            stamp = f"{source.stat().st_size}:{source.stat().st_mtime_ns}"
            previous = self.connection.execute("SELECT value FROM metadata WHERE key=?", (kind,)).fetchone()
            if previous and previous[0] == stamp:
                continue
            n = 0
            for line in gzip.open(source, "rt", encoding="utf-8"):
                entry = json.loads(line)
                expected=entry.get('meta',{}).get('sha256')
                if expected and hashlib.sha256(entry['data_json'].encode()).hexdigest()!=expected:
                    self.index_issues.append(dict(run_id=entry.get('run_id',''),kind=kind,reason='Frozen auxiliary JSON byte hash mismatch'))
                    continue
                blob = zlib.compress(entry["data_json"].encode(), 2)
                self.connection.execute(f"INSERT INTO logs(run_id,{column}) VALUES (?,?) ON CONFLICT(run_id) DO UPDATE SET {column}=excluded.{column}", (entry["run_id"], blob))
                n += 1
                if n % 2000 == 0:
                    log(f"[INDEX] {kind}: {n} frozen logs")
                    self.connection.commit()
            self.connection.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", (kind, stamp))
            self.connection.commit()

    @lru_cache(maxsize=64)
    def history(self, run_id):
        row = self.connection.execute("SELECT history FROM logs WHERE run_id=?", (str(run_id),)).fetchone()
        return json.loads(zlib.decompress(row[0])) if row and row[0] else []

    @lru_cache(maxsize=64)
    def events(self, run_id):
        row = self.connection.execute("SELECT migration FROM logs WHERE run_id=?", (str(run_id),)).fetchone()
        return json.loads(zlib.decompress(row[0])) if row and row[0] else []

    def close(self):
        self.connection.close()
        self.history.cache_clear()
        self.events.cache_clear()


class AtlasStyle(Figures):
    def __init__(self, folder, stats_dir, manifest):
        super().__init__(folder, manifest["version"], manifest["cutoff_ns"])
        self.stats_dir, self.manifest = stats_dir, manifest
        self.entries, self.audits, self.interactive = [], [], []
        self.tables = stats_dir/"ATLAS"
        self.tables.mkdir(exist_ok=True)
        self.folder.mkdir(exist_ok=True)
        self.plt.rcParams.update({"axes.titleweight": "bold", "axes.labelweight": "bold", "font.size": 23})
        self.colors = {m: self.cmap(c) for m, c in zip(METHODS, np.linspace(.07, .93, len(METHODS)))}
        self.colors["FULL"] = self.cmap(.86)

    def figure(self, title, width=13.5, height=9, projection=None, sidebar=False):
        fig = self.plt.figure(figsize=(width, height), facecolor=BG)
        ax = fig.add_subplot(111, projection=projection)
        fig.subplots_adjust(left=.12, right=.70 if sidebar else .95, bottom=.14, top=.86)
        fig.suptitle(title, fontsize=29, weight="bold", y=.98)
        ax.set_facecolor(AX_BG)
        ax.grid(color=GRID, alpha=.65, linewidth=.8)
        if projection != "3d":
            for spine in ax.spines.values():
                spine.set_edgecolor("black"); spine.set_linewidth(1.1)
        else:
            for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
                axis.pane.set_facecolor(AX_BG);axis.pane.set_edgecolor('black')
        return fig, ax

    def emit(self, fig, name, caption, scope="summary", data=None, metadata=None):
        from matplotlib.ticker import MaxNLocator
        for ax in fig.axes:
            if getattr(ax,'name','')=='3d':
                ax.tick_params(labelsize=20,pad=2)
                for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
                    axis.labelpad=27
                    axis.set_major_locator(MaxNLocator(nbins=4))
                ax.set_box_aspect((1,1,1),zoom=.86)
                legend=ax.get_legend()
                if legend is not None:
                    legend.set_ncols(1);legend.set_loc('upper left');legend.set_bbox_to_anchor((1.12,1))
        path = self.folder/name
        path.parent.mkdir(parents=True, exist_ok=True)
        for extension in ("png", "pdf"):
            fig.savefig(Path(str(path)+'.'+extension), dpi=320, bbox_inches="tight", facecolor=BG, pad_inches=.17)
        self.plt.close(fig)
        if data is not None:
            table = self.tables/(name.replace('/', '__')+'.csv')
            write_csv(table, data)
        entry = dict(file=str(name), scope=scope, caption=caption, **(metadata or {}))
        self.entries.append(entry)
        write_csv(self.folder/"atlas_index.csv", self.entries)
        log(f"[ATLAS] {name}")

    def note(self, scope, message):
        self.audits.append(dict(scope=scope, reason=message))

    def legend_panel(self, fig, rows):
        ax = fig.add_axes([.735, .13, .255, .74], facecolor=BG)
        ax.set_xlim(0, 1); ax.set_ylim(len(rows)-.5, -.6); ax.axis('off')
        for i, (method, label, value, n) in enumerate(rows):
            ax.plot([0, .13], [i, i], color=self.colors.get(method, self.cmap(.5)), lw=4 if method == "FULL" else 2.8)
            ax.text(.18, i-.21, label, ha='left', va='center', fontsize=21, weight='bold' if method == 'FULL' else 'normal')
            self.label_box(ax, .44, i+.24, f"{self.scientific(value)}  · n={n}", fontsize=18)

    def export_3d(self, name, coordinates, labels, title, hover=None):
        self.interactive.append(dict(name=name, title=title, coordinates=np.asarray(coordinates).tolist(), labels=list(map(str, labels)),
                                     hover=list(map(str,hover if hover is not None else labels))))

    def finish_atlas(self):
        write_csv(self.tables/"unavailable.csv", self.audits)
        text = [f"# TherMOO deep atlas · {self.manifest['version']}", "",
                "All plotted data come from the frozen snapshot or explicitly marked supplementary traces. Each figure has PNG and vector PDF.", ""]
        html = ['<!doctype html><meta charset="utf-8"><title>TherMOO research atlas</title>',
                '<style>body{background:#FAF8F2;color:#111;font:18px system-ui;max-width:1400px;margin:30px auto;padding:20px}img{max-width:100%;height:auto}section{margin:50px 0}a{color:#31688e}p{line-height:1.6}</style>',
                '<h1>TherMOO research atlas</h1>',
                '<p id="interactive-link"></p><input id="filter" type="search" placeholder="Filter by function, dimension or analysis" style="font:22px system-ui;width:95%;padding:12px">']
        for entry in self.entries:
            text += [f"## {entry['file']}", "", entry['caption'], ""]
            html += [f"<section><h2>{entry['file']}</h2><a href='{entry['file']}.pdf'>Vector PDF</a>",
                     f"<img loading='lazy' src='{entry['file']}.png'><p>{entry['caption']}</p></section>"]
            animation=Path(entry['file']).parent/'population_animation.html'
            if (self.folder/animation).is_file():
                html[-1]=html[-1].replace('</section>',f'<p><a href="{animation}">Observed 3D population animation</a></p></section>')
        html.append("<script>document.getElementById('filter').oninput=function(){const q=this.value.toLowerCase();document.querySelectorAll('section').forEach(s=>s.hidden=!s.textContent.toLowerCase().includes(q))}</script>")
        (self.folder/"CAPTIONS.md").write_text('\n'.join(text))
        (self.folder/"index.html").write_text('\n'.join(html))
        if self.interactive:
            try:
                from plotly.offline import get_plotlyjs
                assets=self.folder/'assets'; assets.mkdir(exist_ok=True)
                (assets/'plotly.min.js').write_text(get_plotlyjs())
                blob=json.dumps(self.interactive,allow_nan=False).replace('</','<\\/')
                page='''<!doctype html><meta charset="utf-8"><script src="assets/plotly.min.js"></script><body style="background:#FAF8F2;font:20px system-ui"><select id="choose"></select><div id="view" style="height:88vh"></div><script>const DATA=PAYLOAD;const palette=['#481567','#3e4989','#26828e','#35b779','#b5de2b','#fde725'];const s=document.getElementById('choose');DATA.forEach((v,i)=>s.add(new Option(v.title,i)));function draw(){const q=DATA[+s.value],labs=[...new Set(q.labels)],tr=labs.map((lab,j)=>{const ids=q.labels.map((v,i)=>v===lab?i:-1).filter(i=>i>=0),a=ids.map(i=>q.coordinates[i]);return {type:'scatter3d',mode:'markers',name:lab,x:a.map(v=>v[0]),y:a.map(v=>v[1]),z:a.map(v=>v[2]),marker:{size:3,opacity:.7,color:palette[j%palette.length]},text:ids.map(i=>q.hover[i])}});Plotly.newPlot('view',tr,{title:q.title,paper_bgcolor:'#FAF8F2',font:{size:20},scene:{aspectmode:'data'}},{responsive:true})}s.onchange=draw;draw()</script>'''.replace('PAYLOAD',blob)
                (self.folder/'interactive_3d.html').write_text(page)
                html.append('<script>document.getElementById("interactive-link").innerHTML=\'<a href="interactive_3d.html">Offline interactive 3D explorer</a>\'</script>')
            except ImportError:
                self.note('interactive_3d', 'Plotly unavailable; all static 3D projections remain in PNG/PDF.')
        write_csv(self.tables/"unavailable.csv", self.audits)
        (self.folder/"index.html").write_text('\n'.join(html))


class DeepAtlas:
    def __init__(self, root, stats_dir, figures_dir, args):
        self.root, self.stats_dir, self.figures_dir, self.args = root, stats_dir, figures_dir, args
        self.manifest=json.loads((stats_dir/'manifest.json').read_text())
        self.raw=read_table(stats_dir/'per_run.csv',dtype={'run_id':str,'initial_hash':str})
        self.cells=read_table(stats_dir/'cells.csv',dtype={'level':str})
        self.comparisons=read_table(stats_dir/'comparisons.csv',dtype={'level':str})
        self.style=AtlasStyle(figures_dir/'ATLAS',stats_dir,self.manifest)
        self.store=FrozenHistory(stats_dir/'snapshot',self.style.tables)
        for issue in self.store.index_issues:self.style.note(issue['run_id'],issue['reason'])
        self.cache=root/'ANALYSIS_CACHE/v4';self.cache.mkdir(parents=True,exist_ok=True)
        self.pop_dir=self.style.tables/'populations';self.pop_dir.mkdir(exist_ok=True)
        self.population_meta=[];self.embedding_audit=[];self.insights=[]
        self.engine_root=Path(self.manifest.get('engine_root',str(root)))
        self.reference_available=False
        try:
            os.environ['THERMOO_ROOT']=str(self.engine_root)
            sys.path.insert(0,str(self.engine_root))
            from thermoo.engine import experiment_fingerprint
            from thermoo.plan import Job
            from thermoo.problems import load_reference
            self.Job,self.load_reference=Job,load_reference
            experiment_fingerprint.cache_clear()
            self.reference_available=experiment_fingerprint()==self.manifest['fingerprint']
        except Exception as error:
            self.style.note('references',str(error))

    def job(self,row):
        values={f.name:row[f.name] for f in dataclass_fields(self.Job) if f.name in row}
        values['m']=int(row.get('requested_m',row['m']));values['d']=int(row.get('requested_d',row['d']))
        for key in ('seed','m','d','budget','population','islands','interval','archive'):
            if key in values:values[key]=int(values[key])
        return self.Job(**values)

    @lru_cache(maxsize=64)
    def population(self,run_id):
        row=self.raw[self.raw.run_id==run_id].iloc[0]
        cache=self.pop_dir/f'{run_id}.npz'
        if cache.is_file():
            with np.load(cache,allow_pickle=False) as data:return {k:data[k] for k in data.files}
        # Exact source result hash connects the late-read archive to the frozen run.
        result=Path(row.source_result)
        if not result.is_file():
            for root in self.manifest.get('source_roots',[]):
                candidate=Path(root)/'outputs/full'/self.manifest['fingerprint']/run_id/'result.json'
                if candidate.is_file() and hashlib.sha256(candidate.read_bytes()).hexdigest()==row.result_sha256:
                    result=candidate;break
        if not result.is_file() or hashlib.sha256(result.read_bytes()).hexdigest()!=row.result_sha256:
            raise ValueError('Original population cannot be matched to the frozen result hash')
        path=result.parent/'population.npz';before=path.stat()
        if max(before.st_mtime_ns,before.st_ctime_ns)>int(self.manifest['cutoff_ns']):
            raise CutoffError('Population archive was published after this snapshot cutoff')
        blob=path.read_bytes();after=path.stat()
        if (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):
            raise CutoffError('Population changed while reading')
        import io
        with np.load(io.BytesIO(blob),allow_pickle=False) as data:
            arrays={k:data[k].copy() for k in data.files if k in ('X','F','population_X','population_F')}
        if 'F' not in arrays or arrays['F'].shape[1]!=int(row.m) or not all(np.isfinite(v).all() for v in arrays.values()):
            raise ValueError('Invalid final population dimensions or values')
        np.savez_compressed(cache,**arrays)
        self.population_meta.append(dict(run_id=run_id,source=str(path),sha256=hashlib.sha256(blob).hexdigest(),cutoff_ns=self.manifest['cutoff_ns']))
        return arrays

    def reference(self,row):
        if not self.reference_available:return None
        try:return np.asarray(self.load_reference(self.job(row)),float)
        except Exception as error:self.style.note(cohort_name(row),'Reference unavailable: '+str(error));return None

    def cohorts(self):
        allowed=self.raw[self.raw.block.isin(('main','many','dimensions'))]
        if self.args.problems:
            allowed=allowed[allowed.problem.isin(self.args.problems)]
        return allowed.groupby(['block','problem','m','d'],sort=False)

    def convergence(self,g,name):
        records=[];methods=[m for m in METHODS if m in set(g.method)]
        for row in g.itertuples():
            h={int(v['fe']):v for v in self.store.history(row.run_id)}
            if not set(FE_GRID).issubset(h):continue
            for fe in FE_GRID:
                for metric in METRICS:
                    records.append(dict(run_id=row.run_id,method=row.method,seed=row.seed,fe=fe,metric=metric,value=h[fe][metric]))
        frame=pd.DataFrame(records)
        if not len(frame):return
        for metric in METRICS:
            fig,ax=self.style.figure(f"{name} · {'hypervolume' if metric=='hv' else 'IGD+'} convergence",width=15,height=max(10,1.05*len(methods)),sidebar=True)
            legend=[]
            for j,method in enumerate(methods):
                sub=frame[(frame.method==method)&(frame.metric==metric)]
                if not len(sub):continue
                q=sub.groupby('fe').value.quantile([.25,.5,.75]).unstack()
                x=q.index.to_numpy(float)
                ax.plot(x,q[.5],color=self.style.colors[method],lw=4.2 if method=='FULL' else 2.4,
                        marker=('D' if method=='FULL' else 'o'),ms=7,markeredgecolor='black',markeredgewidth=.5,
                        linestyle=('-','--','-.',':')[j%4])
                ax.fill_between(x,q[.25].to_numpy(float),q[.75].to_numpy(float),color=self.style.colors[method],alpha=.10)
                legend.append((method,actual_label(method,g.problem.iloc[0]),float(q[.5].iloc[-1]),sub.seed.nunique()))
            ax.set_xscale('log');ax.set_xticks([400,10000,100000],['400','10k','100k'])
            if metric=='igd_plus' and frame[frame.metric==metric].value.max()/max(frame[frame.metric==metric].value.min(),1e-12)>40:
                if (frame[frame.metric==metric].value>0).all():ax.set_yscale('log')
                else:ax.set_yscale('symlog',linthresh=1e-6)
            ax.set_xlabel('Objective-vector evaluations');ax.set_ylabel('HV ↑' if metric=='hv' else 'IGD+ ↓')
            self.style.legend_panel(fig,legend)
            self.style.emit(fig,f'functions/{name}/convergence_{metric}',
                            'Seed median and IQR at the same five exactly observed FE counts. Bands are IQR, not confidence intervals. '
                            'The side panel gives the final median and available independent seed count. IGD+ uses log scaling when strictly positive and sufficiently wide-ranging; '
                            'otherwise a symlog axis with linear threshold 1e-6 retains exact zero values.',scope=name,data=frame[frame.metric==metric])

    def distributions(self,g,name):
        methods=[m for m in METHODS if m in set(g.method)]
        from matplotlib.ticker import FuncFormatter
        for metric in METRICS:
            fig,ax=self.style.figure(f'{name} · final {"HV" if metric=="hv" else "IGD+"} distributions',width=14,height=max(8,3+.65*len(methods)))
            fig.subplots_adjust(left=.20,right=.78)
            wide=metric=='igd_plus' and float(g[metric].max())/max(float(g[metric].min()),1e-12)>30
            transform=(lambda x:np.arcsinh(np.asarray(x)/1e-6)) if wide else (lambda x:np.asarray(x))
            for i,method in enumerate(methods):
                raw=g[g.method==method][metric].to_numpy(float);x=transform(raw)
                color=self.style.colors[method]
                if len(x)>2 and np.ptp(x)>1e-12:
                    violin=ax.violinplot([x],positions=[i],vert=False,widths=.72,showextrema=False)
                    for body in violin['bodies']:body.set_facecolor(color);body.set_edgecolor('black');body.set_alpha(.25)
                ax.boxplot([x],positions=[i],vert=False,widths=.20,showfliers=False,patch_artist=True,manage_ticks=False,
                           boxprops=dict(facecolor='white',edgecolor='black',alpha=.9),medianprops=dict(color='black',linewidth=2),
                           whiskerprops=dict(color='black'),capprops=dict(color='black'))
                rng=np.random.default_rng(stable_seed('jitter',name,method,metric))
                ax.scatter(x,i+rng.uniform(-.15,.15,len(x)),s=25,color=color,edgecolors='black',linewidths=.3,alpha=.65,zorder=4)
                self.style.label_box(ax,1.05,i,self.style.scientific(float(np.median(raw))),transform=ax.get_yaxis_transform(),ha='left',fontsize=21)
            if wide:ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:self.style.scientific(float(np.sinh(x)*1e-6))))
            ax.set_yticks(range(len(methods)),[actual_label(m,g.problem.iloc[0]) for m in methods]);ax.set_ylim(len(methods)-.4,-.6)
            ax.grid(axis='y',visible=False);ax.set_xlabel(('HV ↑' if metric=='hv' else 'IGD+ ↓')+(' · asinh display scale' if wide else ''))
            self.style.emit(fig,f'functions/{name}/raincloud_{metric}',
                            'All available completed seeds. Violin density, Tukey box/whiskers (1.5 IQR) and deterministic jittered seed points; boxes on the right show raw medians. '
                            'Constant or fewer-than-three samples receive no invented KDE. Where IGD+ spans a wide range, density and box/whisker construction use asinh(IGD+/1e-6), '
                            'tick labels and boxed values remain in original units. This is a distribution display, not a significance decision.',scope=name,
                            data=g[['run_id','method','seed',metric]])

    def state_phase(self,g,name):
        selected=g[(g.method.isin(('FULL','ELITE','NO_MIG')))&(g.seed==self.args.representative_seed)]
        records=[]
        for run in selected.itertuples():
            for row in self.store.history(run.run_id):
                if int(row['fe']) != int(run.population) and (int(row['fe'])-int(run.population)) % (int(run.population)*int(run.interval)) != 0:
                    continue
                for island,profile in enumerate(row.get('profiles',[])):
                    if all(k in profile and math.isfinite(profile[k]) for k in STATE_FEATURES):
                        records.append(dict(run_id=run.run_id,method=run.method,island=island,fe=row['fe'],**{k:profile[k] for k in STATE_FEATURES}))
        frame=pd.DataFrame(records)
        if len(frame)<8:return
        matrix=frame[list(STATE_FEATURES)].to_numpy(float)
        if np.max(np.ptp(matrix,axis=0))<=1e-12:
            self.style.note(name,'Stored island state vectors are constant; no informative state projection is drawn.')
            return
        coordinates,fit=pca_coordinates(matrix,3)
        for dimensions in (2,3):
            fig,ax=self.style.figure(f'{name} · island-state PCA trajectories',width=13.5,height=10,projection='3d' if dimensions==3 else None)
            for (method,island),indices in frame.groupby(['method','island'],sort=False).groups.items():
                idx=np.asarray(list(indices));idx=idx[np.argsort(frame.loc[idx,'fe'].to_numpy())]
                xy=coordinates[idx];color=self.style.colors[method]
                style=('-','--',':')[('FULL','ELITE','NO_MIG').index(method)]
                if dimensions==2:
                    ax.plot(xy[:,0],xy[:,1],color=color,ls=style,alpha=.75,lw=2)
                    ax.scatter(xy[:,0],xy[:,1],c=frame.loc[idx,'fe'],cmap='viridis',s=25,edgecolors='black',linewidths=.3)
                    ax.scatter(*xy[0,:2],marker='o',s=100,facecolor='white',edgecolor='black',zorder=6)
                    ax.scatter(*xy[-1,:2],marker='*',s=190,color=color,edgecolor='black',zorder=7)
                    if method=='FULL':self.style.label_box(ax,*xy[-1,:2],f'I{island+1}',fontsize=18)
                else:
                    ax.plot(xy[:,0],xy[:,1],xy[:,2],color=color,ls=style,alpha=.8,lw=2)
                    ax.scatter(*xy[-1,:3],marker='*',s=120,color=color,edgecolor='black')
            ax.set_xlabel(f'PC1 ({100*fit["explained"][0]:.1f}%)');ax.set_ylabel(f'PC2 ({100*fit["explained"][1]:.1f}%)')
            if dimensions==3:ax.set_zlabel(f'PC3 ({100*fit["explained"][2]:.1f}%)')
            from matplotlib.lines import Line2D
            ax.legend(handles=[Line2D([0],[0],color=self.style.colors[m],label=actual_label(m),ls=('-','--',':')[j]) for j,m in enumerate(('FULL','ELITE','NO_MIG')) if m in set(frame.method)],fontsize=19,frameon=True,facecolor='white',edgecolor='black')
            self.style.emit(fig,f'functions/{name}/state_pca_{dimensions}d',
                            'True temporal trajectories of the stored island descriptor vectors, not reconstructed individual-solution paths. '
                            f'One common standardized PCA fit across policies, islands and all recorded times; representative seed {self.args.representative_seed}, chosen before inspecting performance. '
                            'Circles denote initial states and stars final states. Free energy and derivative descriptors are derived/correlated features; loadings are exported for interpretation.',scope=name,
                            data=frame.assign(PC1=coordinates[:,0],PC2=coordinates[:,1],PC3=coordinates[:,2]))
        loading=pd.DataFrame(fit['axes'],index=STATE_FEATURES,columns=['PC1','PC2','PC3'])
        write_csv(self.style.tables/(name+'__state_pca_loadings.csv'),loading.reset_index(names='feature'))
        fig,ax=self.style.figure(f'{name} · interpretation of state PCA axes',width=11.5,height=11)
        fig.subplots_adjust(left=.25)
        self.style.heatmap(ax,loading.to_numpy(),STATE_FEATURES,loading.columns.tolist(),
                           [[f'{v:+.2f}' for v in row] for row in loading.to_numpy()],limits=(-1,1),fontsize=21)
        self.style.emit(fig,f'functions/{name}/state_pca_loadings','Signed standardized-PCA component coefficients. These explain which recorded descriptors define each common state-projection axis. '
                        'Free energy and temporal derivatives are derived features, so correlated coefficients are not independent causal contributions.',scope=name,data=loading.reset_index(names='feature'))
        self.style.export_3d(name+'_states',coordinates,frame.method.astype(str)+' / I'+(frame.island+1).astype(str),name+' · recorded island states')
        if not self.args.no_tsne:
            ids=self.sample_groups(frame,['method','island'],self.args.max_embedding_points)
            try:
                emb,audit=joint_embedding(matrix[ids],self.cache,name+'_states',self.args.max_embedding_points)
                self.embedding_audit.append(dict(scope=name,space='island_state',**audit))
                self.embedding_plot(emb,frame.iloc[ids].method.to_numpy(),name,'state_tsne',
                                    'Joint t-SNE of recorded state vectors. Connected states are observations at successive times, but t-SNE distances are not migration distances or velocities. '
                                    'The fit pools all selected policies/islands/times; independent per-frame fits are never connected. Neighborhood and restart audits are exported.',frame.iloc[ids])
            except Exception as error:self.style.note(name,'State t-SNE: '+str(error))

    def sample_groups(self,frame,keys,limit):
        groups=list(frame.groupby(keys,sort=False).groups.values());per=max(1,limit//max(1,len(groups)))
        output=[]
        for group in groups:
            indices=np.asarray(list(group));positions=np.linspace(0,len(indices)-1,min(per,len(indices)),dtype=int)
            output.extend(indices[positions])
        return np.asarray(output[:limit],int)

    def embedding_plot(self,coordinates,labels,name,slug,caption,data=None):
        fig,ax=self.style.figure(f'{name} · {slug.replace("_"," ")}',width=13,height=10)
        if data is not None and all(k in data for k in ('method','island','fe')):
            meta=data.reset_index(drop=True)
            for (method,island),idx in meta.groupby(['method','island'],sort=False).groups.items():
                idx=np.asarray(list(idx));idx=idx[np.argsort(meta.loc[idx,'fe'])]
                ax.plot(coordinates[idx,0],coordinates[idx,1],color=self.style.colors.get(method,'black'),alpha=.35,lw=1.5)
        for j,label in enumerate(dict.fromkeys(labels)):
            ids=np.asarray(labels)==label
            color=self.style.colors.get(str(label),self.style.cmap(.15+.7*j/max(1,len(set(labels))-1)))
            ax.scatter(coordinates[ids,0],coordinates[ids,1],s=35,color=color,edgecolor='black',linewidth=.35,alpha=.7,label=actual_label(str(label)))
        ax.set_xlabel('Embedding coordinate 1');ax.set_ylabel('Embedding coordinate 2')
        ax.legend(loc='best',fontsize=18,frameon=True,facecolor='white',edgecolor='black',ncol=2)
        self.style.emit(fig,f'functions/{name}/{slug}',caption,scope=name,
                        data=None if data is None else data.assign(embedding_1=coordinates[:,0],embedding_2=coordinates[:,1]))

    def final_geometry(self,g,name):
        chosen=g[g.seed==self.args.representative_seed]
        arrays={};groups={};reference=self.reference(g.iloc[0].to_dict())
        for row in chosen.itertuples():
            try:
                data=self.population(row.run_id)
                arrays[row.method]=data
                groups[row.method]=row
            except Exception as error:self.style.note(name,f'{row.method} final archive: {error}')
        if not arrays:return
        m=int(g.m.iloc[0]);problem=g.problem.iloc[0]
        if m in (2,3):
            fig,ax=self.style.figure(f'{name} · final objective-space archives',width=14,height=10,projection='3d' if m==3 else None)
            if reference is not None:
                idx=np.linspace(0,len(reference)-1,min(1000,len(reference)),dtype=int)
                if m==2:ax.scatter(reference[idx,0],reference[idx,1],s=8,color='#777777',alpha=.23,label='Reference front')
                else:ax.scatter(reference[idx,0],reference[idx,1],reference[idx,2],s=6,color='#777777',alpha=.12)
            for method,data in arrays.items():
                f=data['F'];idx=np.linspace(0,len(f)-1,min(180,len(f)),dtype=int)
                kwargs=dict(s=28,alpha=.6,color=self.style.colors.get(method,self.style.cmap(.5)),edgecolor='black',linewidth=.3,label=actual_label(method,problem))
                if m==2:ax.scatter(f[idx,0],f[idx,1],**kwargs)
                else:ax.scatter(f[idx,0],f[idx,1],f[idx,2],**kwargs)
            ax.set_xlabel('Objective 1 ↓');ax.set_ylabel('Objective 2 ↓')
            if m==3:ax.set_zlabel('Objective 3 ↓')
            ax.legend(fontsize=18,ncol=2,facecolor='white',edgecolor='black',loc='best')
            self.style.emit(fig,f'functions/{name}/objective_front_{m}d',
                            f'Actual final nondominated output archives, representative seed {self.args.representative_seed}. '
                            'The seed is fixed before inspecting quality; these are illustrative archives, not best-run fronts. '
                            'Original objective coordinates are retained. Reference points are shown only when the current source fingerprint matches the frozen experiment.',scope=name)
        for space,key in (('objectives','F'),('decisions','X')):
            points=[];labels=[]
            for method,data in arrays.items():
                if key not in data:continue
                x=np.asarray(data[key],float)
                idx=np.linspace(0,len(x)-1,min(len(x),max(5,self.args.max_embedding_points//max(1,len(arrays)))),dtype=int)
                points.append(x[idx]);labels.extend([method]*len(idx))
            if not points:continue
            matrix=np.vstack(points)
            scaling='Common empirical range scaling'
            if space=='objectives' and reference is not None:
                lo=reference.min(axis=0);span=np.maximum(np.ptp(reference,axis=0),1e-12)
                scaled=(matrix-lo)/span;scaling='Shared reference-front ideal/nadir normalization'
            elif space=='decisions' and self.reference_available:
                try:
                    from thermoo.problems import Benchmark
                    problem_object=Benchmark(self.job(g.iloc[0].to_dict()))
                    scaled=(matrix-problem_object.xl)/np.maximum(problem_object.xu-problem_object.xl,1e-12)
                    scaling='Shared decision-variable bound normalization'
                except Exception:
                    scaled=(matrix-matrix.min(axis=0))/np.maximum(np.ptp(matrix,axis=0),1e-12)
            else:scaled=(matrix-matrix.min(axis=0))/np.maximum(np.ptp(matrix,axis=0),1e-12)
            coordinates,fit=pca_coordinates(scaled,3,standardize=False)
            self.embedding_plot(coordinates,np.asarray(labels),name,f'final_{space}_pca2',
                                f'{scaling}; one pooled, centered PCA fit across all methods at the fixed representative seed. '
                                'This shows final geometry, not a time trajectory. Projection does not preserve Pareto dominance; quality is assessed in the original objective space.',
                                pd.DataFrame(dict(method=labels)))
            if scaled.shape[1]>=3:
                fig,ax=self.style.figure(f'{name} · final {space} PCA 3D',width=13.5,height=10,projection='3d')
                for method in dict.fromkeys(labels):
                    idx=np.asarray(labels)==method
                    ax.scatter(coordinates[idx,0],coordinates[idx,1],coordinates[idx,2],s=20,color=self.style.colors.get(method,'black'),alpha=.6,label=actual_label(method,problem))
                ax.set_xlabel('PC1');ax.set_ylabel('PC2');ax.set_zlabel('PC3');ax.legend(fontsize=18,ncol=2)
                self.style.emit(fig,f'functions/{name}/final_{space}_pca3',f'{scaling}. Common three-dimensional PCA of final archives, not individual trajectories. Explained variance: {fit["explained"].tolist()}.',scope=name)
                self.style.export_3d(name+'_'+space,coordinates,labels,name+' · final '+space)
            if not self.args.no_tsne and len(matrix)>=8:
                try:
                    metric='hamming' if space=='decisions' and problem=='ZDT5' else 'euclidean'
                    embedded,audit=joint_embedding(scaled,self.cache,name+'_'+space,self.args.max_embedding_points,standardize=False,metric=metric)
                    self.embedding_audit.append(dict(scope=name,space='final_'+space,**audit))
                    self.embedding_plot(embedded,np.asarray(labels),name,f'final_{space}_tsne',
                                        f'{scaling}; a single joint t-SNE fit across all methods. Fixed representative seed; deterministic balanced point sampling. '
                                        'Neighborhood preservation and restart stability are exported. Cluster separation is a visualization, not evidence of better optimization or a distance/velocity estimate.',
                                        pd.DataFrame(dict(method=labels)))
                except Exception as error:self.style.note(name,f'Final {space} t-SNE: {error}')
        if m>=3:
            fig,ax=self.style.figure(f'{name} · objective trade-off profiles',width=13,height=9)
            if reference is not None:lo,span=reference.min(axis=0),np.maximum(np.ptp(reference,axis=0),1e-12)
            else:
                all_f=np.vstack([v['F'] for v in arrays.values()]);lo=all_f.min(axis=0);span=np.maximum(np.ptp(all_f,axis=0),1e-12)
            for method,data in arrays.items():
                f=(data['F']-lo)/span;q=np.quantile(f,[.1,.5,.9],axis=0);x=np.arange(m)
                ax.plot(x,q[1],marker='o',color=self.style.colors.get(method,'black'),label=actual_label(method,problem),lw=3.5 if method=='FULL' else 2)
                ax.fill_between(x,q[0],q[2],color=self.style.colors.get(method,'black'),alpha=.08)
            ax.set_xticks(range(m),[f'$f_{{{j+1}}}$' for j in range(m)]);ax.set_ylabel('Normalized objective value ↓')
            ax.legend(fontsize=18,ncol=2,facecolor='white',edgecolor='black')
            self.style.emit(fig,f'functions/{name}/parallel_objectives','Median and 10th–90th quantiles across archived solutions at the fixed representative seed, on shared objective normalization. '
                            'These describe trade-offs within each final archive, not confidence intervals across independent runs.',scope=name)

    def attainment(self,g,name):
        if int(g.m.iloc[0])!=2:return
        paired=[];fronts={'FULL':[],'ELITE':[]}
        for seed,panel in g[g.method.isin(('FULL','ELITE'))].groupby('seed'):
            if set(panel.method)!={'FULL','ELITE'} or panel.initial_hash.nunique()!=1:continue
            try:
                values={r.method:self.population(r.run_id)['F'] for r in panel.itertuples()}
                paired.append(seed)
                for method in fronts:fronts[method].append(values[method])
            except Exception as error:self.style.note(name,'Attainment: '+str(error))
        if len(paired)<2:return
        reference=self.reference(g.iloc[0].to_dict())
        if reference is None:
            combined=np.vstack(fronts['FULL']+fronts['ELITE']);lo=combined.min(axis=0);span=np.maximum(np.ptp(combined,axis=0),1e-12)
        else:lo=reference.min(axis=0);span=np.maximum(np.ptp(reference,axis=0),1e-12)
        normalized={m:[(f-lo)/span for f in fs] for m,fs in fronts.items()}
        x=y=np.linspace(0,1.15,100)
        maps={m:empirical_attainment(fs,x,y) for m,fs in normalized.items()}
        fig,ax=self.style.figure(f'{name} · empirical attainment difference',width=12.5,height=10)
        image=ax.pcolormesh(x,y,maps['FULL']-maps['ELITE'],vmin=-1,vmax=1,cmap='viridis',shading='auto',rasterized=True)
        for method,linestyle in (('FULL','-'),('ELITE','--')):
            for level in (.25,.5,.75):
                if maps[method].min()<level<maps[method].max():
                    ax.contour(x,y,maps[method],levels=[level],colors='black',linestyles=linestyle,linewidths=2.5 if level==.5 else .9,alpha=1 if level==.5 else .4)
        cb=fig.colorbar(image,ax=ax,pad=.035);cb.set_label('Attainment probability: FULL − Elite')
        ax.set_xlabel('Normalized objective 1 ↓');ax.set_ylabel('Normalized objective 2 ↓')
        self.style.label_box(ax,.03,.97,f'{len(paired)} paired seeds',transform=ax.transAxes,ha='left',va='top',fontsize=21)
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([0],[0],color='black',lw=2.5,ls='-',label='FULL median attainment'),
                           Line2D([0],[0],color='black',lw=2.5,ls='--',label='Elite median attainment')],
                  loc='lower left',fontsize=18,facecolor='white',edgecolor='black')
        if max(maps['FULL'].max(),maps['ELITE'].max())==0:
            self.style.label_box(ax,.5,.5,'Neither method attained this reference window',transform=ax.transAxes,fontsize=19)
        self.style.emit(fig,f'functions/{name}/attainment_difference',
                        'Empirical attainment probabilities from paired FULL/ELITE seed archives in original two-objective dominance space. '
                        'Color is FULL minus ELITE attainment probability; solid/ dashed contours denote FULL/ELITE at 25%, 50%, 75% (thick: median). '
                        'The frame covers normalized [0,1.15] around the reference front, or the jointly scaled observed range if references are unavailable. '
                        'This is a spatial descriptive difference; no grid-wise significance is claimed.',scope=name,
                        data=pd.DataFrame(dict(x=np.tile(x,len(y)),y=np.repeat(y,len(x)),FULL=maps['FULL'].ravel(),ELITE=maps['ELITE'].ravel())))

    def migration_network(self,g,name):
        selected=g[(g.method.isin(('FULL','ELITE','RANDOM','HV_GATE','CROWDING')))&(g.seed==self.args.representative_seed)]
        for run in selected.itertuples():
            events=self.store.events(run.run_id)
            if not events:continue
            k=int(run.actual_islands);sent=np.zeros((k,k));accepted=np.zeros((k,k))
            for event in events:
                i,j=int(event['source']),int(event['target'])
                if 0<=i<k and 0<=j<k:sent[i,j]+=event['sent'];accepted[i,j]+=event['accepted']
            if k>8:
                fig,ax=self.style.figure(f'{name} · {actual_label(run.method)} directed flux',width=16,height=15)
                fig.subplots_adjust(left=.18,bottom=.18)
                fraction=np.divide(accepted,sent,out=np.full_like(sent,np.nan),where=sent>0)
                im=ax.imshow(np.ma.masked_invalid(fraction),vmin=0,vmax=1,cmap='viridis',aspect='equal')
                ax.grid(False);ax.set_xticks(range(k),[f'I{i+1}' for i in range(k)],rotation=45)
                ax.set_yticks(range(k),[f'I{i+1}' for i in range(k)])
                for i,j in zip(*np.where(sent>0)):self.style.label_box(ax,j,i,f'{int(sent[i,j])}',fontsize=20)
                ax.set_xlabel('Recipient island');ax.set_ylabel('Source island')
                cb=fig.colorbar(im,ax=ax,pad=.025);cb.set_label('Accepted / sent')
                self.style.emit(fig,f'functions/{name}/network_{run.method}',
                                'Directed adjacency representation for more than eight islands, avoiding a crowded circular network. '
                                'Numeric boxes are sent migrant counts; color is accepted/sent. Unused links are masked. '
                                f'Representative seed {self.args.representative_seed}; no spatial trajectory is implied by matrix positions.',scope=name,
                                data=pd.DataFrame(dict(source=np.repeat(np.arange(k),k),target=np.tile(np.arange(k),k),sent=sent.ravel(),accepted=accepted.ravel())))
                continue
            fig,ax=self.style.figure(f'{name} · {actual_label(run.method)} migration flux',width=12.5,height=11)
            angles=np.linspace(0,2*np.pi,k,endpoint=False)+np.pi/2;positions=np.c_[np.cos(angles),np.sin(angles)]
            from matplotlib.patches import FancyArrowPatch
            for i in range(k):
                for j in range(k):
                    if sent[i,j]<=0:continue
                    fraction=accepted[i,j]/sent[i,j]
                    ax.add_patch(FancyArrowPatch(positions[i]*.9,positions[j]*.9,arrowstyle='-|>',mutation_scale=15,
                                                connectionstyle='arc3,rad=.15',lw=1+5*sent[i,j]/max(1,sent.max()),
                                                color=self.style.cmap(fraction),alpha=.35+.5*fraction,zorder=2))
            kinds=str(run.actual_algorithms).split('|')
            for i,(x,y) in enumerate(positions):
                ax.scatter(x,y,s=650,color=self.style.cmap(.12+.75*(i%4)/3),edgecolors='black',lw=1.2,zorder=4)
                self.style.label_box(ax,x*1.25,y*1.25,f'I{i+1}\n{kinds[i] if i<len(kinds) else "island"}',fontsize=19)
            ax.set_xlim(-1.65,1.65);ax.set_ylim(-1.65,1.65);ax.set_aspect('equal');ax.axis('off')
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
            cb=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap='viridis'),ax=ax,shrink=.65,pad=.03)
            cb.set_label('Accepted / sent')
            self.style.emit(fig,f'functions/{name}/network_{run.method}',
                            f'Directed, actually recorded migrant counts at representative seed {self.args.representative_seed}. Edge width is sent count; viridis color is accepted/sent. '
                            'Circular node positions describe topology and are not coordinates in genotype or objective space. No-event policies generate no invented edges.',scope=name,
                            data=pd.DataFrame(dict(source=np.repeat(np.arange(k),k),target=np.tile(np.arange(k),k),sent=sent.ravel(),accepted=accepted.ravel())))

    def migration_event_response(self,g,name):
        pairs=[]
        for seed,panel in g[g.method.isin(('FULL','NO_MIG'))].groupby('seed'):
            if set(panel.method)!={'FULL','NO_MIG'} or panel.initial_hash.nunique()!=1:continue
            runs=panel.set_index('method');full=runs.loc['FULL'];control=runs.loc['NO_MIG']
            hf={int(r['fe']):r for r in self.store.history(full.run_id)};hc={int(r['fe']):r for r in self.store.history(control.run_id)}
            step=int(full.population)*int(full.interval)
            episodes=sorted({int(e['fe']) for e in self.store.events(full.run_id)})
            for fe in episodes:
                grid=[fe+lag*step for lag in (-2,-1,0,1,2)]
                if any(t not in hf or t not in hc for t in grid):continue
                for metric in METRICS:
                    sign=1 if metric=='hv' else -1
                    def gap(t):
                        a,b=hf[t][metric],hc[t][metric]
                        return sign*(a-b)/max(abs(a)+abs(b),1e-12)
                    baseline=gap(fe-step)
                    for lag,t in zip((-2,-1,0,1,2),grid):
                        pairs.append(dict(seed=seed,event_fe=fe,lag=lag,metric=metric,value=gap(t)-baseline))
        frame=pd.DataFrame(pairs)
        if not len(frame):return
        per_run=frame.groupby(['seed','lag','metric']).value.median().reset_index()
        fig,axes=self.style.canvas(f'{name} · event-aligned migration diagnostic',height=9.5)
        fig.subplots_adjust(wspace=.40,bottom=.20)
        fig.supxlabel('Coordination steps from migration episode',fontsize=24,weight='bold',y=.04)
        for ax,metric in zip(axes,METRICS):
            sub=per_run[per_run.metric==metric];q=sub.groupby('lag').value.quantile([.25,.5,.75]).unstack()
            ax.plot(q.index,q[.5],color=self.style.cmap(.72),marker='o',lw=3)
            ax.fill_between(q.index,q[.25],q[.75],color=self.style.cmap(.72),alpha=.18)
            ax.axhline(0,color='black',lw=1);ax.axvline(0,color='black',ls='--',lw=1)
            ax.set_xlabel('');ax.set_ylabel('Change in relative quality gap')
            ax.set_title(METRIC_NAMES[metric],weight='bold')
        self.style.emit(fig,f'functions/{name}/migration_event_response',
                        'Same-seed FULL and no-migration archives at identical recorded FE. Unique migration times define episodes. '
                        'The oriented gap is (FULL−control)/(abs(FULL)+abs(control)) for HV, with reversed sign for IGD+. '
                        'It is centered at the pre-episode step (lag −1). Episode medians are computed within each run before summarizing seed medians/IQR. '
                        'Triggers are endogenous and windows can overlap: this is an event-aligned diagnostic, not a causal effect estimate.',scope=name,data=frame)

    def dimension_bumps(self):
        rows=self.raw[(self.raw.block.isin(('main','dimensions')))&self.raw.problem.isin(DIM_PROBLEMS)&self.raw.method.isin(DIM_METHODS)]
        for problem,g in rows.groupby('problem',sort=False):
            dimensions=sorted(g.d.unique());records=[]
            for d,panel in g.groupby('d'):
                if set(panel.method)!=set(DIM_METHODS):continue
                for metric in METRICS:
                    v=panel.groupby('method')[metric].median().reindex(DIM_METHODS)
                    ranks=stats.rankdata(-v.to_numpy() if metric=='hv' else v.to_numpy())
                    records.extend(dict(problem=problem,d=d,method=m,metric=metric,rank=float(r)) for m,r in zip(DIM_METHODS,ranks))
            frame=pd.DataFrame(records)
            if not len(frame) or frame.d.nunique()<2:continue
            fig,axes=self.style.canvas(f'{problem} · rank reversals with dimension',height=10,width=18)
            for ax,metric in zip(axes,METRICS):
                for method in DIM_METHODS:
                    sub=frame[(frame.method==method)&(frame.metric==metric)].sort_values('d')
                    ax.plot(sub.d,sub['rank'],marker='o',color=self.style.colors[method],lw=3.5 if method=='FULL' else 2,label=actual_label(method))
                    for row in sub.itertuples():self.style.label_box(ax,row.d,row.rank,f'{row.rank:g}',fontsize=18)
                ax.set_xticks(dimensions);ax.set_ylim(len(DIM_METHODS)+.5,.5);ax.set_xlabel('Decision variables D');ax.set_ylabel('Rank of seed median ↓');ax.set_title(METRIC_NAMES[metric],weight='bold')
            handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.56,.92),ncol=4,fontsize=18)
            fig.subplots_adjust(top=.76)
            self.style.emit(fig,f'dimensions/{problem}_rank_bumps',
                            'Per-dimension seed-median ranks over the same eight planned methods. A dimension enters only when every method has at least one measured outcome. '
                            'Available seed counts can differ; this is descriptive. Different D values are not independent new benchmark functions, and initial genotypes are not paired across D.',scope='dimensions',data=frame)

    def effect_atlas(self):
        for study in ('main','ablation','sensitivity_interval','sensitivity_rho','sensitivity_temperature','sensitivity_quality'):
            g=self.comparisons[(self.comparisons.study==study)&(self.comparisons.n_pairs>0)] if len(self.comparisons) else pd.DataFrame()
            if not len(g):continue
            sheets = list(g.groupby('family', sort=False)) if study == 'main' else [('panel', g)]
            for sheet, sheet_data in sheets:
                g = sheet_data
                problems=list(dict.fromkeys(g.problem));levels=list(dict.fromkeys(g.level.astype(str)))
                for metric in METRICS:
                    sub=g[g.metric==metric]
                    values=sub.pivot(index='problem',columns='level',values='rank_biserial').reindex(index=problems,columns=levels)
                    fig,ax=self.style.figure(f'{study} · function-level paired effects · {METRIC_NAMES[metric]}',width=max(12,3+1.15*len(levels)),height=max(8,3+.43*len(problems)))
                    ax.grid(False);im=ax.imshow(np.ma.masked_invalid(values.to_numpy(float)),vmin=-1,vmax=1,cmap='viridis',aspect='auto',interpolation='nearest')
                    ax.set_yticks(range(len(problems)),problems,fontsize=19);ax.set_xticks(range(len(levels)),[LABELS.get(v,v) for v in levels],rotation=42,ha='right',fontsize=20)
                    for i,problem in enumerate(problems):
                        for j,level in enumerate(levels):
                            r=sub[(sub.problem==problem)&(sub.level.astype(str)==level)]
                            if not len(r):continue
                            row=r.iloc[0];star='*' if bool(row.eligible_inference) and np.isfinite(row.p_holm) and row.p_holm<.05 else ''
                            self.style.label_box(ax,j,i,f'{row.rank_biserial:+.2f}{star}',fontsize=18)
                    cb=fig.colorbar(im,ax=ax,pad=.03);cb.set_label('Paired rank-biserial effect · positive favors control')
                    self.style.emit(fig,f'summary/effects_{study}_{sheet}_{metric}',
                                    'Effect sizes are calculated from seed-paired differences in each function/configuration. Positive favors the predefined control. '
                                    'An asterisk appears only for a 30-pair-eligible primary comparison with the original planned-block, both-endpoint Holm correction below 0.05. '
                                    'Incomplete comparisons remain descriptive; this atlas does not create a new uncorrected testing family.',data=sub)
        for (study,problem),g in self.comparisons[(self.comparisons.study=='ablation')&(self.comparisons.n_pairs>0)].groupby(['study','problem']) if len(self.comparisons) else []:
            for metric in METRICS:
                sub=g[g.metric==metric].sort_values('advantage_mean')
                fig,ax=self.style.figure(f'{problem} · ablation effect magnitudes · {METRIC_NAMES[metric]}',width=13,height=9)
                fig.subplots_adjust(left=.29,right=.84)
                for i,row in enumerate(sub.itertuples()):
                    self.style.gradient_barh(ax,i,row.advantage_mean,height=.5)
                    if np.isfinite(row.advantage_ci_low) and np.isfinite(row.advantage_ci_high):ax.plot([row.advantage_ci_low,row.advantage_ci_high],[i,i],color='black',lw=2)
                    self.style.label_box(ax,1.03,i,self.style.scientific(row.advantage_mean),transform=ax.get_yaxis_transform(),ha='left',fontsize=20)
                ax.set_yticks(range(len(sub)),[LABELS.get(v,v) for v in sub.level]);ax.set_ylim(len(sub)-.4,-.6);ax.axvline(0,color='black',lw=1)
                ax.set_xlabel('Mean paired advantage of FULL');ax.grid(axis='y',visible=False)
                self.style.emit(fig,f'ablations/{problem}_{metric}_tornado',
                                'Actual oriented metric differences, not just win percentages: FULL−variant for HV, variant−FULL for IGD+. '
                                'Intervals are the frozen paired bootstrap mean-difference intervals. Seed pairing and inference gates are unchanged.',data=sub)

    def performance_profiles(self):
        main=self.raw[(self.raw.block=='main')&(self.raw.problem!='ZDT5')]
        cases=[];feature_rows=[]
        for problem,g in main.groupby('problem',sort=False):
            if set(g.method)!=set(METHODS[:10]):continue
            row=dict(problem=problem,family=g.family.iloc[0],m=int(g.m.iloc[0]),min_n=int(g.groupby('method').size().min()))
            for metric in METRICS:
                med=g.groupby('method')[metric].median().reindex(METHODS[:10]).to_numpy()
                rank=stats.rankdata(-med if metric=='hv' else med)
                for method,value,r in zip(METHODS[:10],med,rank):
                    cases.append(dict(problem=problem,family=row['family'],method=method,metric=metric,value=value,rank=r))
                    row[f'{metric}_{method}']=(len(rank)-r)/(len(rank)-1)
            feature_rows.append(row)
        frame=pd.DataFrame(cases);features=pd.DataFrame(feature_rows)
        if not len(frame):return
        for metric in METRICS:
            sub=frame[frame.metric==metric];matrix=sub.pivot(index='problem',columns='method',values='value').reindex(columns=METHODS[:10])
            x=matrix.to_numpy(float)
            if metric=='hv':
                best=x.max(axis=1);ratios=np.divide(best[:,None],x,out=np.full_like(x,np.inf),where=x>0)
                ratios[best==0]=1
            else:
                best=x.min(axis=1);ratios=np.divide(x,best[:,None],out=np.full_like(x,np.inf),where=best[:,None]>0)
                ratios[(best==0)[:,None]&(x==0)]=1
            finite=ratios[np.isfinite(ratios)];upper=max(2,min(1e4,float(finite.max()) if len(finite) else 2))
            thresholds=np.unique(np.r_[1,np.geomspace(1,upper,180)])
            fig,ax=self.style.figure(f'Across-function performance profiles · {METRIC_NAMES[metric]}',width=14,height=10)
            for j,method in enumerate(METHODS[:10]):
                y=np.mean(ratios[:,j,None]<=thresholds[None,:]*(1+1e-12),axis=0)
                ax.step(thresholds,y,where='post',color=self.style.colors[method],lw=4 if method=='FULL' else 2,label=actual_label(method))
            ax.set_xscale('log');ax.set_ylim(-.02,1.03);ax.set_xlabel('Factor from the best observed method median');ax.set_ylabel('Fraction of benchmark functions ↑')
            ax.legend(fontsize=18,ncol=2,loc='lower right',facecolor='white',edgecolor='black')
            self.style.emit(fig,f'summary/performance_profile_{metric}',
                            'Descriptive performance profiles over common main functions with at least one completed seed for every planned method; binary ZDT5 is separate. '
                            'Ratios use best-HV/method-HV or method-IGD+/best-IGD+. If the best denominator is zero, tied zeros have ratio one and strictly worse values have infinity. '
                            'Ratios beyond the displayed maximum remain in the denominator and are not silently discarded. Per-function medians may use unequal available seed counts.',data=sub)
        # Radar uses the same common-function panel and dimensionless rank scores.
        families=features.family.unique().tolist()
        if len(families)>=3:
            selected=('FULL','ELITE','NO_MIG','NSGA2','MOEAD','SMS')
            fig=self.style.plt.figure(figsize=(12,11),facecolor=BG);ax=fig.add_subplot(111,polar=True);ax.set_facecolor(AX_BG)
            theta=np.linspace(0,2*np.pi,len(families),endpoint=False);closed=np.r_[theta,theta[0]]
            records=[]
            for method in selected:
                values=[]
                for family in families:
                    g=features[features.family==family];v=float(((g['hv_'+method]+g['igd_plus_'+method])/2).mean());values.append(v)
                    records.append(dict(method=method,family=family,rank_score=v,problems=len(g)))
                ax.plot(closed,np.r_[values,values[0]],color=self.style.colors[method],lw=3.6 if method=='FULL' else 2,label=actual_label(method))
                ax.fill(closed,np.r_[values,values[0]],color=self.style.colors[method],alpha=.06)
            ax.set_xticks(theta,families,fontsize=23);ax.set_ylim(0,1);ax.set_yticks([.25,.5,.75,1]);ax.tick_params(pad=14)
            ax.legend(loc='upper left',bbox_to_anchor=(1.02,1.02),fontsize=19);fig.suptitle('Cross-family robustness · rank-normalized quality',fontsize=28,weight='bold')
            self.style.emit(fig,'summary/robustness_radar',
                            'Same common benchmark panel for all curves. Each axis is the family mean of the equally weighted HV and IGD+ rank score (K−rank)/(K−1), with K=10. '
                            'No raw metrics with incompatible units are combined. Larger is better; polygon area is not an additional quality measure. Families absent from the common panel are not filled with zeros.',data=pd.DataFrame(records))
        if len(features)>=4 and np.max(np.ptp(features[[f'{metric}_{method}' for metric in METRICS for method in METHODS[:10]]].to_numpy(),axis=0))>1e-12:
            columns=[f'{metric}_{method}' for metric in METRICS for method in METHODS[:10]]
            coordinates,fit=pca_coordinates(features[columns].to_numpy(),3,standardize=False)
            fig,ax=self.style.figure('Functions grouped by algorithm-performance signatures',width=14,height=11)
            for j,family in enumerate(features.family.unique()):
                idx=features.family==family;ax.scatter(coordinates[idx,0],coordinates[idx,1],s=100,color=self.style.cmap(.1+.8*j/max(1,features.family.nunique()-1)),edgecolor='black',label=family)
            for i,row in features.iterrows():
                if row.problem in PANEL:self.style.label_box(ax,coordinates[i,0],coordinates[i,1],row.problem,fontsize=18)
            ax.set_xlabel(f'PC1 ({100*fit["explained"][0]:.1f}%)');ax.set_ylabel(f'PC2 ({100*fit["explained"][1]:.1f}%)');ax.legend(fontsize=19,ncol=2)
            self.style.emit(fig,'summary/function_signature_pca',
                            'A common PCA of per-function algorithm rank-score profiles over both endpoints, not of pooled objective coordinates from different optimization problems. '
                            'This is an exploratory map of observed solver behavior. Only the predeclared panel is annotated to avoid unreadable overlapping labels; all functions are exported.',
                            data=features.assign(PC1=coordinates[:,0],PC2=coordinates[:,1],PC3=coordinates[:,2]))
            self.style.export_3d('function_signatures',coordinates,features.family,'Algorithm-performance signatures of functions',hover=features.problem)

    def recorded_target_profiles(self):
        main=self.raw[(self.raw.block=='main')&(self.raw.problem!='ZDT5')]
        records=[]
        for run in main.itertuples():
            h={int(v['fe']):v for v in self.store.history(run.run_id)}
            if not set(FE_GRID).issubset(h):continue
            initial=h[FE_GRID[0]]['igd_plus']
            for fraction in (.5,.1,.01):
                hit=next((fe for fe in FE_GRID if h[fe]['igd_plus']<=fraction*initial),np.inf)
                records.append(dict(problem=run.problem,family=run.family,method=run.method,seed=run.seed,target_fraction=fraction,first_observed_fe=hit))
        frame=pd.DataFrame(records)
        if not len(frame):return
        common=set(frame.problem)
        for method in METHODS[:10]:common &= set(frame[frame.method==method].problem)
        frame=frame[frame.problem.isin(common)]
        if not len(frame):return
        for fraction,g in frame.groupby('target_fraction'):
            fig,ax=self.style.figure(f'First recorded IGD+ target attainment · {fraction:g} × initial',width=14,height=10)
            for method in METHODS[:10]:
                sub=g[g.method==method];ys=[]
                for fe in FE_GRID:
                    per_problem=sub.assign(success=sub.first_observed_fe<=fe).groupby(['family','problem']).success.mean().reset_index()
                    ys.append(float(per_problem.groupby('family').success.mean().mean()))
                ax.step(FE_GRID,ys,where='post',color=self.style.colors[method],lw=3.8 if method=='FULL' else 2,label=actual_label(method))
            ax.set_xscale('log');ax.set_xticks([400,10000,100000],['400','10k','100k']);ax.set_ylim(-.02,1.02)
            ax.set_xlabel('Objective-vector evaluations');ax.set_ylabel('Fraction reaching the recorded target ↑');ax.legend(fontsize=18,ncol=2)
            self.style.emit(fig,f'summary/recorded_target_{fraction:g}',
                            'First observed attainment on the same five actual checkpoints, not an exact continuous hitting time. '
                            'The predeclared target is a fraction of each run’s initial IGD+. Unreached trials remain censored at the budget; no artificial successful time is imputed. '
                            'Success fractions average seeds within functions and then functions within equally weighted represented families, using a common function panel for all methods.',data=g)

    def lagged_state_associations(self):
        rows=[]
        for run in self.raw[(self.raw.block=='main')&(self.raw.method=='FULL')].itertuples():
            hist=[r for r in self.store.history(run.run_id) if r['fe']==run.population or (r['fe']-run.population)%(run.population*run.interval)==0]
            pairs=[]
            for current,next_row in zip(hist,hist[1:]):
                profiles=current.get('profiles',[])
                if not profiles:continue
                record={k:float(np.mean([p[k] for p in profiles])) for k in STATE_FEATURES if all(k in p for p in profiles)}
                record.update(fe=current['fe'],current_hv=current['hv'],future_hv_gain=next_row['hv']-current['hv'])
                pairs.append(record)
            frame=pd.DataFrame(pairs)
            if len(frame)<8:continue
            for feature in ('H','E','free','drift','diversity','progress'):
                if feature not in frame:continue
                sub=frame[[feature,'fe','current_hv','future_hv_gain']].replace([np.inf,-np.inf],np.nan).dropna()
                if len(sub)<8:continue
                ranked=sub.apply(lambda x:stats.rankdata(x))
                design=np.c_[np.ones(len(sub)),ranked.fe,ranked.current_hv]
                x=ranked[feature].to_numpy();y=ranked.future_hv_gain.to_numpy()
                rx=x-design@np.linalg.lstsq(design,x,rcond=None)[0];ry=y-design@np.linalg.lstsq(design,y,rcond=None)[0]
                if rx.std()<1e-10 or ry.std()<1e-10:continue
                rows.append(dict(run_id=run.run_id,problem=run.problem,family=run.family,seed=run.seed,feature=feature,
                                 partial_spearman=float(np.corrcoef(rx,ry)[0,1]),observations=len(sub)))
        result=pd.DataFrame(rows)
        write_csv(self.style.tables/'lagged_state_associations.csv',result)
        if not len(result):return
        by_problem=result.groupby(['problem','family','feature']).partial_spearman.median().reset_index()
        mean=by_problem.groupby(['family','feature']).partial_spearman.mean().unstack().reindex(index=FAMILIES,columns=['H','E','free','drift','diversity','progress'])
        fig,ax=self.style.figure('State signals and subsequent archive improvement',width=13,height=10)
        self.style.heatmap(ax,mean.to_numpy(),FAMILIES,['Entropy H','Energy E','Free energy F','JS drift','Diversity','Online progress'],
                           [[f'{v:+.2f}' if np.isfinite(v) else '—' for v in row] for row in mean.to_numpy()],limits=(-1,1),fontsize=22)
        self.style.emit(fig,'summary/lagged_state_associations',
                        'Within-run partial Spearman associations of each current descriptor with next-coordination-step archive HV gain, controlling rank(FE) and rank(current archive HV). '
                        'At least eight finite varying observations are required. Run coefficients become per-function seed medians and then equally weighted function means within each family. '
                        'These are temporal associations, not out-of-sample predictive validation, independent-timepoint tests or causal feature effects. H, E and F are structurally related.',data=by_problem)

    def variant_details(self):
        planned=read_table(self.stats_dir/'planned_runs.csv',dtype={'run_id':str})
        if not len(planned):return
        views=make_views(planned.to_dict('records'))
        extra=[k for k in self.raw if k not in views]
        observed=views.merge(self.raw[['run_id']+extra],on='run_id',how='inner',validate='many_to_one')
        for study in ('sensitivity_interval','sensitivity_rho','sensitivity_temperature','sensitivity_quality'):
            sub=observed[observed.study==study]
            if self.args.problems:sub=sub[sub.problem.isin(self.args.problems)]
            for problem,g in sub.groupby('problem',sort=False):
                levels=list(g.level.astype(str).unique());numeric=study!='sensitivity_quality'
                if len(levels)<2:continue
                if numeric:levels=sorted(levels,key=float)
                else:levels=sorted(levels)
                fig,axes=self.style.canvas(f'{problem} · {study.removeprefix("sensitivity_")} response',width=18,height=9)
                records=[]
                for ax,metric in zip(axes,METRICS):
                    med=[];low=[];high=[]
                    for level in levels:
                        values=g[g.level.astype(str)==level][metric].to_numpy(float)
                        q=np.quantile(values,[.25,.5,.75]);low.append(q[0]);med.append(q[1]);high.append(q[2])
                        records.append(dict(problem=problem,study=study,level=level,metric=metric,n=len(values),q25=q[0],median=q[1],q75=q[2]))
                    x=np.asarray([float(v) for v in levels]) if numeric else np.arange(len(levels))
                    ax.plot(x,med,color=self.style.cmap(.72),marker='o',lw=3)
                    ax.fill_between(x,low,high,color=self.style.cmap(.72),alpha=.18)
                    for xx,value in zip(x,med):self.style.label_box(ax,xx,value,self.style.scientific(value),fontsize=20)
                    ax.set_xticks(x,[v.replace('rank_','') for v in levels],rotation=25 if not numeric else 0)
                    if metric=='igd_plus' and min(low)>0 and max(high)/min(low)>30:ax.set_yscale('log')
                    ax.set_title(METRIC_NAMES[metric],weight='bold');ax.set_xlabel(study.removeprefix('sensitivity_'));ax.set_ylabel('Measured final metric')
                self.style.emit(fig,f'sensitivity/{problem}_{study}',
                                'Actual one-factor levels and the exact reused main FULL default. Points are seed medians; ribbons are IQR. '
                                'Lines connect measured levels only and do not establish a continuous response surface or interactions between factors.',scope='sensitivity',data=pd.DataFrame(records))
        for (block,problem,k,composition),g in self.raw[self.raw.block.isin(('islands','homogeneous'))].groupby(['block','problem','islands','composition'],sort=False):
            if self.args.problems and problem not in self.args.problems:continue
            name=cohort_name(g.iloc[0].to_dict())+f'_K{int(k)}_{composition}'
            self.state_phase(g,name);self.migration_network(g,name)

    def insights_report(self):
        comp=self.comparisons
        if len(comp):
            eligible=comp[(comp.eligible_inference==True)&comp.p_holm.notna()]
            for row in eligible.itertuples():
                if row.p_holm<.05:
                    self.insights.append(dict(study=row.study,facet=row.facet,problem=row.problem,metric=row.metric,
                                              control=row.control,variant=row.level,direction='control better' if row.advantage_mean>0 else 'control worse',
                                              effect=row.advantage_mean,p_holm=row.p_holm,n_pairs=row.n_pairs))
        write_csv(self.style.tables/'supported_pairwise_findings.csv',self.insights)
        lines=['# Measured findings and interpretation limits','',
               'This report retains positive and negative findings. Significance below refers only to the predeclared 30-pair eligible tests already in the snapshot. Projection clusters and state correlations are exploratory.','']
        if self.insights:
            table=pd.DataFrame(self.insights)
            lines+=['## Confirmed paired contrasts','',table.groupby(['study','metric','direction']).size().rename('contrasts').to_string(),'']
            for study,g in table.groupby('study',sort=False):
                lines+=[f'### {study}','']
                # Order is by problem and comparator, not by the most flattering effect.
                for row in g.sort_values(['problem','variant','metric']).itertuples():
                    lines.append(f'- {row.problem}, {row.facet}, {row.metric}: {row.control} vs {row.variant}: {row.direction}; oriented mean difference {row.effect:.6g}, Holm p={row.p_holm:.4g}, n={row.n_pairs}.')
                lines.append('')
        else:lines+=['No eligible corrected contrast supports a significance claim at this snapshot. Descriptive figures remain useful but are not substituted for that evidence.','']
        lines+=['## What the projections represent','',
                'Existing histories contain per-island descriptors over time and only final solution populations. Descriptor trajectories are observed state paths. Final PCA/t-SNE clouds are static archive projections. They do not reconstruct missing solution histories.','',
                'Supplementary traces, if requested, record full island populations at checkpoints and actual migrant genotypes. Migration changes island membership; cached X and F do not move to a new location in a common genotype/objective embedding. Drawn inter-island arrows represent transfers between population centroids, not literal travel of a genotype through feature space.','',
                't-SNE is fitted jointly over all displayed groups/times. It is audited with neighborhood trustworthiness and a second random seed. Its global distances, cluster areas, line lengths and apparent velocities are not optimization metrics.','',
                'One-factor sensitivity is not an interaction experiment: no interpolated response surface or factorial interaction claim is produced from the sparse OAT plan. Shared-server wall time is descriptive; no physical speedup is inferred.','']
        (self.stats_dir/'INSIGHTS.md').write_text('\n'.join(lines))

    def run(self):
        stages=(('function effects',self.effect_atlas),('performance signatures',self.performance_profiles),
                ('recorded target profiles',self.recorded_target_profiles),('dimension reversals',self.dimension_bumps),
                ('lagged state associations',self.lagged_state_associations),('sensitivity and architecture',self.variant_details))
        for name,stage in stages:
            log(f'[ATLAS STAGE] {name}')
            try:stage()
            except Exception as error:self.style.note(name,traceback.format_exc());self.style.plt.close('all');log(f'[ATLAS NOTE] {name}: {error}')
        cohorts=list(self.cohorts())
        for index,(key,g) in enumerate(cohorts,1):
            name=cohort_name(g.iloc[0].to_dict())
            log(f'[FUNCTION {index}/{len(cohorts)}] {name} · {len(g)} measured runs')
            for stage in (self.convergence,self.distributions,self.state_phase,self.final_geometry,self.attainment,self.migration_network,self.migration_event_response):
                try:stage(g,name)
                except Exception as error:self.style.note(name,stage.__name__+': '+traceback.format_exc());self.style.plt.close('all');log(f'[ATLAS NOTE] {name}/{stage.__name__}: {error}')
            write_csv(self.style.tables/'population_sources.csv',self.population_meta)
            write_csv(self.style.tables/'embedding_audit.csv',self.embedding_audit)
        self.insights_report();self.style.finish_atlas();self.store.close()
        self.manifest.update(deep_atlas_version=ATLAS_VERSION,deep_atlas_figures=len(self.style.entries),atlas_finished_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(self.stats_dir/'manifest.json',self.manifest)
        return len(self.style.entries)


@contextmanager
def record_population_trace(engine,folder):
    """Read-only wrappers at actual observation/migration calls, never a global profiler."""
    trace=folder/'trace';trace.mkdir(exist_ok=True)
    original_indicators,original_choose,original_inject=engine.indicators,engine.choose_migrants,engine.inject
    pending={};metadata=dict(protocol=ATLAS_VERSION,populations=[],events=[],timing_comparable=False)
    def write_npz(path,**arrays):
        temporary=path.with_name(path.stem+f'.{os.getpid()}.tmp.npz')
        np.savez_compressed(temporary,**arrays);temporary.replace(path)
    def observe(F,pf):
        values=original_indicators(F,pf)
        caller=sys._getframe(1)
        if caller.f_code.co_name=='observe':
            outer=caller.f_back;local=outer.f_locals
            algorithms=local['algorithms'];fe=int(local['evaluations'])
            x=np.vstack([a.pop.get('X') for a in algorithms]);f=np.vstack([a.pop.get('F') for a in algorithms])
            islands=np.concatenate([np.full(len(a.pop),i,dtype=int) for i,a in enumerate(algorithms)])
            path=trace/f'population_FE{fe:09d}.npz'
            write_npz(path,X=x,F=f,island=islands,fe=np.array(fe))
            metadata['populations'].append(dict(fe=fe,file=path.name))
            problem=local['problem']
            metadata['xl']=np.asarray(problem.xl).tolist();metadata['xu']=np.asarray(problem.xu).tolist()
            metadata['algorithms']=[a.thermoo_kind for a in algorithms]
        return values
    def choose(*args,**kwargs):
        result=original_choose(*args,**kwargs)
        caller=sys._getframe(1);local=caller.f_locals
        if caller.f_code.co_name=='_compute' and all(k in local for k in ('source','target','evaluations','algorithms')):
            key=(int(local['evaluations']),int(local['source']),int(local['target']))
            algorithms=local['algorithms']
            pending[key]=(np.asarray(algorithms[key[1]].pop.get('X'),float).mean(axis=0),
                          np.asarray(algorithms[key[2]].pop.get('X'),float).mean(axis=0))
        return result
    def inject(algorithm,migrants,problem,crowding_replacement=False):
        before={np.asarray(row).tobytes() for row in algorithm.pop.get('X')}
        incoming_x=migrants.get('X').copy();incoming_f=migrants.get('F').copy()
        accepted=original_inject(algorithm,migrants,problem,crowding_replacement)
        caller=sys._getframe(1);local=caller.f_locals
        if caller.f_code.co_name=='_compute':
            fe,source,target=int(local['evaluations']),int(local['source']),int(local['target'])
            now={np.asarray(row).tobytes() for row in algorithm.pop.get('X')}
            retained=np.array([np.asarray(row).tobytes() in now-before for row in incoming_x])
            pair=pending.pop((fe,source,target),None)
            path=trace/f'migration_{len(metadata["events"]):06d}.npz'
            arrays=dict(X=incoming_x,F=incoming_f,retained_new=retained,fe=np.array(fe),source=np.array(source),target=np.array(target),
                        accepted_distinct=np.array(accepted),target_after=np.asarray(algorithm.pop.get('X'),float).mean(axis=0))
            if pair is not None:arrays.update(source_before=pair[0],target_before=pair[1])
            write_npz(path,**arrays)
            metadata['events'].append(dict(fe=fe,source=source,target=target,sent=len(incoming_x),accepted=int(accepted),file=path.name))
        return accepted
    engine.indicators,engine.choose_migrants,engine.inject=observe,choose,inject
    try:yield metadata
    finally:
        engine.indicators,engine.choose_migrants,engine.inject=original_indicators,original_choose,original_inject
        metadata['profile_phase']='Stored descriptors are calculated before migration; population checkpoints are recorded after the step/migration.'
        metadata['no_persistent_lineage_ids']=True
        atomic_json(trace/'trace_manifest.json',metadata)


def run_traced_trial(engine,job,folder):
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'trace.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result_file=folder/'result.json'
        if result_file.is_file():
            result=json.loads(result_file.read_text())
            if result.get('status')=='complete' and result.get('trace_protocol')==ATLAS_VERSION and (folder/'trace/trace_manifest.json').is_file():return result
        started=time.perf_counter()
        with record_population_trace(engine,folder):
            result=engine._compute(job,folder,started,verbose=True)
        result.update(status='complete',job=asdict(job),fingerprint=engine.experiment_fingerprint(),run_id=job.signature(),
                      exploratory_trace=True,trace_protocol=ATLAS_VERSION,timing_comparable=False,
                      tracer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        atomic_json(result_file,result)
        return result


def draw_true_population_trace(atlas,folder,label):
    info=json.loads((folder/'trace/trace_manifest.json').read_text())
    entries=sorted(info['populations'],key=lambda r:r['fe'])
    if len(entries)<2:return
    low=np.asarray(info['xl']);span=np.maximum(np.asarray(info['xu'])-low,1e-12)
    clouds=[];centroids=[];meta=[]
    for entry in entries:
        with np.load(folder/'trace'/entry['file'],allow_pickle=False) as d:
            x=(d['X'].astype(float)-low)/span;islands=d['island']
        clouds.append(x)
        for island in np.unique(islands):
            centroids.append(x[islands==island].mean(axis=0));meta.append(dict(fe=entry['fe'],island=int(island)))
    matrix=np.vstack(clouds);centroids=np.asarray(centroids);meta=pd.DataFrame(meta)
    _,fit=pca_coordinates(matrix,3,standardize=False)
    coordinates,_=pca_coordinates(centroids,3,fit=fit)
    for dimensions in (2,3):
        fig,ax=atlas.style.figure(label+' · observed population trajectories',width=14,height=11,projection='3d' if dimensions==3 else None)
        for island,indices in meta.groupby('island',sort=False).groups.items():
            idx=np.asarray(list(indices));xy=coordinates[idx];color=atlas.style.cmap(.1+.8*int(island)/max(1,meta.island.max()))
            if dimensions==2:
                ax.plot(xy[:,0],xy[:,1],color=color,lw=2.8)
                ax.scatter(xy[:,0],xy[:,1],c=meta.loc[idx,'fe'],cmap='viridis',edgecolor='black',linewidth=.4,s=40)
                ax.scatter(*xy[0,:2],facecolor='white',edgecolor='black',s=130,marker='o',zorder=5)
                ax.scatter(*xy[-1,:2],color=color,edgecolor='black',s=220,marker='*',zorder=6)
                atlas.style.label_box(ax,*xy[-1,:2],f'I{int(island)+1}',fontsize=21)
            else:
                ax.plot(xy[:,0],xy[:,1],xy[:,2],color=color,lw=2.5)
                ax.scatter(*xy[-1,:3],color=color,edgecolor='black',s=150,marker='*')
        if dimensions==2:
            from matplotlib.patches import FancyArrowPatch
            event_indices=np.linspace(0,len(info['events'])-1,min(18,len(info['events'])),dtype=int) if info['events'] else []
            for i in event_indices:
                event=info['events'][int(i)]
                with np.load(folder/'trace'/event['file'],allow_pickle=False) as d:
                    if 'source_before' not in d or 'target_before' not in d:continue
                    pair=(np.vstack([d['source_before'],d['target_before']])-low)/span
                    xy,_=pca_coordinates(pair,3,fit=fit)
                    ax.add_patch(FancyArrowPatch(xy[0,:2],xy[1,:2],arrowstyle='-|>',connectionstyle='arc3,rad=.1',
                                                 color='black',alpha=.25,lw=1.2,mutation_scale=12))
        ax.set_xlabel(f'PC1 ({100*fit["explained"][0]:.1f}%)');ax.set_ylabel(f'PC2 ({100*fit["explained"][1]:.1f}%)')
        if dimensions==3:ax.set_zlabel(f'PC3 ({100*fit["explained"][2]:.1f}%)')
        atlas.style.emit(fig,f'traces/{atlas_slug(label)}/population_pca_{dimensions}d',
                         'Supplementary traced rerun with the original optimizer, RNGs and FE budget, separate from the primary benchmark sample. '
                         'One centered PCA is fitted to pooled bound-normalized populations at all recorded checkpoints. Lines track exact island population centroids; circle=start, star=end. '
                         'Faint arrows, uniformly sampled over recorded events, link pre-migration source/recipient centroids. They represent membership transfers, not movement of unchanged migrant genotypes in feature space. '
                         'Trace I/O affects runtime, which is excluded from primary runtime comparisons. Individual ancestry is not inferred.',scope='supplementary_trace',data=meta.assign(PC1=coordinates[:,0],PC2=coordinates[:,1],PC3=coordinates[:,2]))
    atlas.style.export_3d(label+'_true_population',coordinates,'Island '+(meta.island+1).astype(str),label+' · observed population centroids')
    # Offline animation uses only actually recorded clouds; no invented particle identities or interpolated frames.
    try:
        import plotly
        from matplotlib.colors import to_hex
        animation_frames=[]
        for entry,cloud in zip(entries,clouds):
            projected,_=pca_coordinates(cloud,3,fit=fit)
            with np.load(folder/'trace'/entry['file'],allow_pickle=False) as d:islands=d['island']
            traces=[]
            for island in np.unique(islands):
                pts=projected[islands==island]
                traces.append(dict(type='scatter3d',mode='markers',name=f'Island {int(island)+1}',
                                   x=pts[:,0].tolist(),y=pts[:,1].tolist(),z=pts[:,2].tolist(),
                                   marker=dict(size=3,opacity=.6,color=to_hex(atlas.style.cmap(.1+.8*int(island)/max(1,meta.island.max()))))))
            animation_frames.append(dict(name=str(entry['fe']),data=traces))
        payload=json.dumps(animation_frames,allow_nan=False).replace('</','<\\/')
        projected_all,_=pca_coordinates(matrix,3,fit=fit)
        minima,maxima=projected_all.min(axis=0),projected_all.max(axis=0)
        padding=np.maximum((maxima-minima)*.06,.01)
        scene={axis:dict(range=[float(minima[j]-padding[j]),float(maxima[j]+padding[j])],backgroundcolor=AX_BG,gridcolor=GRID,showbackground=True)
               for j,axis in enumerate(('xaxis','yaxis','zaxis'))}
        scene['aspectmode']='data'
        destination=atlas.style.folder/'traces'/atlas_slug(label)/'population_animation.html'
        page='''<!doctype html><meta charset="utf-8"><script src="../../assets/plotly.min.js"></script><body style="background:#FAF8F2;font:20px system-ui"><h2>Observed population checkpoints</h2><p>Rotate the 3D view and use the FE slider. Points are observed populations; no individual ancestry or between-checkpoint interpolation is implied.</p><div id="view" style="height:82vh"></div><script>const frames=PAYLOAD;Plotly.newPlot('view',frames[0].data,{paper_bgcolor:'#FAF8F2',font:{size:20},scene:{aspectmode:'data'},sliders:[{steps:frames.map(f=>({label:f.name,method:'animate',args:[[f.name],{mode:'immediate',frame:{duration:0,redraw:true},transition:{duration:0}}]}))}],updatemenus:[{type:'buttons',buttons:[{label:'Play',method:'animate',args:[null,{frame:{duration:350,redraw:true},transition:{duration:0},fromcurrent:true}]}]}]}).then(()=>Plotly.addFrames('view',frames));</script>'''.replace('PAYLOAD',payload)
        page=page.replace("scene:{aspectmode:'data'}",'scene:'+json.dumps(scene))
        destination.write_text(page)
    except ImportError:pass
    if not atlas.args.no_tsne and len(centroids)>=8 and len(centroids)<atlas.args.max_embedding_points:
        available=atlas.args.max_embedding_points-len(centroids)
        idx=np.linspace(0,len(matrix)-1,min(available,len(matrix)),dtype=int)
        combined=np.vstack([centroids,matrix[idx]])
        emb,audit=joint_embedding(combined,atlas.cache,label+'_true_population',atlas.args.max_embedding_points,standardize=False)
        atlas.embedding_audit.append(dict(scope=label,space='traced_population',**audit))
        fig,ax=atlas.style.figure(label+' · joint t-SNE population trajectories',width=14,height=10)
        ax.scatter(emb[len(centroids):,0],emb[len(centroids):,1],s=8,color='#777777',alpha=.13)
        for island,indices in meta.groupby('island').groups.items():
            idx=np.asarray(list(indices));xy=emb[idx];color=atlas.style.cmap(.1+.8*island/max(1,meta.island.max()))
            ax.plot(xy[:,0],xy[:,1],color=color,lw=2);ax.scatter(xy[:,0],xy[:,1],c=meta.loc[idx,'fe'],cmap='viridis',s=32,edgecolor='black',linewidth=.3)
            atlas.style.label_box(ax,*xy[-1,:2],f'I{island+1}',fontsize=20)
        ax.set_xlabel('Joint t-SNE coordinate 1');ax.set_ylabel('Joint t-SNE coordinate 2')
        atlas.style.emit(fig,f'traces/{atlas_slug(label)}/population_tsne',
                         'A single joint t-SNE fit of actual bound-normalized centroid vectors and uniformly sampled population observations over all recorded times. '
                         'Lines connect observed centroid vectors; geometric line lengths/speeds in this nonlinear projection are not interpreted as physical search distances. '
                         'Audit includes a second embedding seed, neighborhood preservation and restart consistency.',scope='supplementary_trace',data=meta.assign(TSNE1=emb[:len(centroids),0],TSNE2=emb[:len(centroids),1]))


def supplementary_trace_panel(atlas):
    import thermoo.engine as engine
    engine.experiment_fingerprint.cache_clear()
    if not atlas.reference_available or engine.experiment_fingerprint()!=atlas.manifest['fingerprint']:
        raise RuntimeError('Supplementary traces require the current code/reference fingerprint to match the frozen snapshot')
    problems=[p for p in TRACE_PANEL if not atlas.args.problems or p in atlas.args.problems]
    records=[]
    for problem in problems:
        for method in ('NO_MIG','ELITE','FULL'):
            job=atlas.Job(problem,method,atlas.args.representative_seed,block='trace_v4')
            folder=atlas.root/'TRACES/v4'/engine.experiment_fingerprint()/job.signature()
            log(f'[TRACE] {problem}/{method}/seed={job.seed} · {job.budget} FE · separate supplementary run')
            try:
                result=run_traced_trial(engine,job,folder)
                record=dict(problem=problem,method=method,seed=job.seed,folder=str(folder),hv=result['metrics']['hv'],igd_plus=result['metrics']['igd_plus'],evaluations=result['evaluations'],timing_comparable=False)
                primary=atlas.raw[atlas.raw.run_id==job.signature()]
                if len(primary):
                    record['matches_primary_metrics']=all(np.isclose(float(primary[m].iloc[0]),result['metrics'][m],rtol=1e-10,atol=1e-12) for m in METRICS)
                records.append(record)
                draw_true_population_trace(atlas,folder,f'{problem}_{method}_s{job.seed}')
            except Exception as error:
                atlas.style.note(problem,method+' trace: '+traceback.format_exc());log(f'[TRACE FAILED] {problem}/{method}: {error}')
    write_csv(atlas.style.tables/'supplementary_trace_runs.csv',records)
    write_csv(atlas.style.tables/'embedding_audit.csv',atlas.embedding_audit)
    atlas.style.finish_atlas()


ATLAS_PROTOCOL = """# Deep research atlas v4

Inputs and report destination are independent: both /home/tahiti/TherMOO and
/home/tahiti/Malashin_Projects/TherMOO are inspected by the V3 intake core.
Only compatible full-budget results under one verified fingerprint are joined,
with one observation per planned run ID. Every new deliverable is under
/home/tahiti/TherMOO. Running primary experiments are not moved or modified.

The deep atlas has no fixed figure limit. It includes function-level actual
HV/IGD+ convergence, seed distributions/rainclouds with Tukey whiskers, paired
effect maps with the original Holm eligibility, ablation effect magnitudes,
OAT parameter response curves, dimension rank reversals, cross-function
performance/target profiles, a dimensionless robustness radar, performance
signature maps, island-state PCA/t-SNE trajectories and loadings, final archive
geometry/parallel coordinates, empirical attainment differences for two-objective
problems, directed migration fluxes and event-aligned diagnostics.

No raw objective/genotype coordinates from different functions or dimensions
are pooled into a common geometric fit. Each geometry map uses one common fit
across the displayed methods/times within its configuration. Decision vectors
use common bounds where available, objective vectors use the recorded reference
frame when its fingerprint is verified, and descriptor vectors use one pooled
standardization. t-SNE is seeded, uses one numerical-library thread, and is
audited using a second seed with random initialization, trustworthiness in the
original scaled input space, and neighbor Jaccard agreement across restarts.
The displayed seed is fixed in advance; no most-attractive restart is selected.
FULL is the existing heuristic controller; no trained neural model, learned
controller weights or neural feature attributions are claimed by this atlas.
t-SNE cluster areas, global distances, apparent velocities and path lengths
are not interpreted as optimizer quality. Plot quality uses original-space
HV, IGD+ and empirical attainment, not embedding separation.

Violin/box/seed-point figures use all available completed independent trials.
KDE is not invented for constant/tiny samples. Convergence bands and parameter
response bands are IQR, explicitly distinct from confidence intervals. Primary
inferential results retain the 30-pair gate and the planned both-endpoint Holm
family. Additional geometric and event/correlation displays are exploratory;
time points are not counted as independent runs. OAT sensitivity is not treated
as a factorial interaction study. Dimensions are not new independent functions.

Existing optimization outputs contain final X/F and time-varying island
descriptors, not intermediate solution populations. Accordingly, existing-data
trajectory figures are descriptor-state paths and static final archive maps.

--trace-panel explicitly runs a separate qualitative mechanistic study:
ZDT1, DTLZ2, WFG9, LZ09_F1, UF8, ZCAT20; NO_MIG, ELITE, FULL; fixed seed 0 by
default; original 100000-FE budget and all original operators. Checkpoint X/F,
island membership, actual migrant X/F and distinct-new retention are recorded
through read-only observation/migration wrappers. The original optimizer core
is not edited, and trace results never enter primary per_run.csv or its tests.
Trace I/O changes measured time, so these runtimes are not compared with primary
timing. Metrics are checked against a matching primary run when it exists.
Migration changes membership while cached genotype/objective coordinates remain
unchanged. Inter-island arrows describe centroid-to-centroid transfers, not
fictional point displacement or individual ancestry. Files go to TRACES/v4.

Representative solution/geometry figures use the predeclared seed, never the
best trial. Archive files are linked to the frozen result byte hash, publication
cutoff and source checks; copied derived arrays are retained for reproducibility.
Missing/changed archives are logged, not replaced by generated points.

References: https://jmlr.org/papers/v9/vandermaaten08a.html
https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html

Presentation: cream, viridis, translucent fills, black axes, large text,
white numeric boxes with black borders. PNG 320 dpi and vector PDF; a local
HTML gallery and optional self-contained offline interactive 3D viewer.
"""


def main(argv=None):
    parser=argparse.ArgumentParser(description='TherMOO deep research atlas v4; dual-source safe intake, unlimited scientific figures.')
    parser.add_argument('--root',type=Path,default=Path('/home/tahiti/TherMOO'))
    parser.add_argument('--source-root',type=Path,action='append')
    parser.add_argument('--snapshot',nargs='?',const='latest',help='Use a completed frozen snapshot; default captures current results')
    parser.add_argument('--bootstrap',type=int,default=4000)
    parser.add_argument('--permutations',type=int,default=50000)
    parser.add_argument('--representative-seed',type=int,default=0)
    parser.add_argument('--max-embedding-points',type=int,default=1600)
    parser.add_argument('--no-tsne',action='store_true')
    parser.add_argument('--problems',nargs='+',help='Optional function-detail and trace subset; overview retains the full snapshot')
    parser.add_argument('--trace-panel',action='store_true',help='Explicitly run the separate 18-trial mechanistic trace panel')
    args=parser.parse_args(argv)
    if args.max_embedding_points<50:parser.error('--max-embedding-points must be at least 50')
    root=args.root.expanduser().resolve()
    vendor=root/'ANALYSIS_VENDOR/v4'
    if vendor.is_dir():sys.path.insert(0,str(vendor))
    forwarded=['--root',str(root),'--bootstrap',str(args.bootstrap),'--permutations',str(args.permutations)]
    for source in args.source_root or []:forwarded+=['--source-root',str(source)]
    if args.snapshot:forwarded+=['--replot',args.snapshot]
    code=snapshot_main(forwarded)
    if code:return code
    version,stats_dir,figures_dir=LAST_ALLOCATED_VERSION
    manifest=json.loads((stats_dir/'manifest.json').read_text())
    manifest.update(status='ATLAS_RUNNING',deep_atlas_version=ATLAS_VERSION,atlas_arguments=vars(args))
    atomic_json(stats_dir/'manifest.json',manifest)
    (stats_dir/'ATLAS_PROTOCOL.md').write_text(ATLAS_PROTOCOL)
    atlas=None
    try:
        atlas=DeepAtlas(root,stats_dir,figures_dir,args)
        count=atlas.run()
        if args.trace_panel:supplementary_trace_panel(atlas)
        manifest=json.loads((stats_dir/'manifest.json').read_text())
        dependencies={}
        for package in ('scikit-learn','joblib','threadpoolctl','plotly'):
            try:dependencies[package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:dependencies[package]=None
        manifest.update(status='COMPLETE',deep_atlas_figures=len(atlas.style.entries),atlas_dependencies=dependencies,
                        atlas_finished_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(stats_dir/'manifest.json',manifest)
        log(f'[ATLAS DONE] {version}: {len(atlas.style.entries)} scientific atlas figures; no fixed figure cap')
        log(f'[GALLERY] {figures_dir/"ATLAS/index.html"}')
        log(f'[INSIGHTS] {stats_dir/"INSIGHTS.md"}')
        return 0
    except BaseException as error:
        manifest.update(status='ATLAS_INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'ATLAS_FAILED',error=str(error),traceback=traceback.format_exc())
        atomic_json(stats_dir/'manifest.json',manifest)
        if atlas is not None:atlas.style.finish_atlas()
        raise


if __name__=='__main__':
    sys.exit(main())

