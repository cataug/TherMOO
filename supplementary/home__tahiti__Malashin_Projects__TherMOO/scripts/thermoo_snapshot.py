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

VERSION = "1.0.0"
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


def capture(root, fingerprint, jobs, family, stats_dir, cutoff_ns):
    """Capture only immutable, successful, planned trials. Audit every exclusion."""
    source = root / "outputs" / "full" / fingerprint
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
                if not result_file.is_file():
                    missing.append(dict(run_id=run_id, block=job.block, problem=job.problem,
                                        method=job.method, seed=job.seed, reason="not_complete_at_cutoff"))
                    continue
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
            except (OSError, ValueError, KeyError, TypeError, CutoffError) as error:
                reason = str(error)
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
                       source_result=str(result_file), result_sha256=meta["sha256"])
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
                sidecar = root / "reports" / scheduler / fingerprint / f"{run_id}.json"
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


class Figures:
    def __init__(self, folder, version, cutoff_ns):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        self.folder, self.version = folder, version
        self.cutoff = datetime.fromtimestamp(cutoff_ns / 1e9, timezone.utc).strftime("%d %b %Y, %H:%M UTC")
        self.index = []
        plt.rcParams.update({"font.size": 16, "axes.titlesize": 20, "axes.labelsize": 18,
                             "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 14,
                             "figure.titlesize": 22, "font.family": "DejaVu Sans", "axes.linewidth": 1.,
                             "lines.linewidth": 2.2, "axes.edgecolor": "black", "text.color": "black",
                             "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
                             "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 320})
        self.cmap = plt.get_cmap("viridis")

    def canvas(self, title, columns=1, height=7.8, width=None):
        fig, axes = self.plt.subplots(1, columns, figsize=(width or (12 if columns == 1 else 19), height),
                                      facecolor=BG, squeeze=False)
        fig.suptitle(title, weight="bold", y=.975)
        fig.subplots_adjust(left=.19 if columns == 1 else .12, right=.96, top=.83, bottom=.19,
                            wspace=.32)
        for ax in axes.flat:
            ax.set_facecolor(AX_BG)
            ax.grid(color=GRID, lw=.8, alpha=.7, zorder=0)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(1.)
        return fig, list(axes.flat)

    def save(self, fig, number, slug, caption, note=""):
        stem = f"{number:02d}_{slug}"
        text = f"{self.version} · {self.cutoff}"
        if note:
            text += "\n" + note
        fig.text(.02, .025, text, fontsize=11, va="bottom", linespacing=1.5)
        for extension in ("png", "pdf"):
            fig.savefig(self.folder / f"{stem}.{extension}", dpi=320, bbox_inches="tight", facecolor=BG)
        self.plt.close(fig)
        self.index.append(dict(number=number, file=stem, status="created", caption=caption, note=note))
        log(f"[FIGURE] {stem}.png / .pdf")

    def skip(self, number, slug, reason):
        self.index.append(dict(number=number, file=f"{number:02d}_{slug}", status="not_ready", caption=reason, note=""))
        log(f"[FIGURE] {number:02d}_{slug}: not ready ({reason})")

    def point_plot(self, ax, labels, centers, lows, highs, color=.6, limits=None, neutral=None):
        y = np.arange(len(labels))
        for i, (value, low, high) in enumerate(zip(centers, lows, highs)):
            col = self.cmap(color if np.isscalar(color) else color[i])
            if np.isfinite(low) and np.isfinite(high):
                ax.plot([low, high], [i, i], color=col, lw=3.5, alpha=.65, zorder=3)
                ax.plot([low, low, high, high], [i-.09, i+.09, i-.09, i+.09], color=col, ls="none", marker="|", ms=8)
            ax.scatter(value, i, color=col, edgecolors="black", s=115, linewidths=.9, zorder=4)
        ax.set_yticks(y, labels)
        ax.set_ylim(len(labels)-.45, -.55)
        if neutral is not None:
            ax.axvline(neutral, color="black", ls="--", lw=1.2, alpha=.6)
        if limits is not None:
            ax.set_xlim(*limits)
        ax.grid(axis="y", visible=False)

    def heatmap(self, ax, matrix, row_labels, column_labels, texts=None, vmin=None, vmax=None,
                cmap=None, colorbar=None, fontsize=12):
        cmap = (self.plt.get_cmap(cmap) if isinstance(cmap, str) else cmap) or self.cmap
        cmap = cmap.copy()
        cmap.set_bad("#DDDAD1")
        a = np.asarray(matrix, float)
        im = ax.imshow(np.ma.masked_invalid(a), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
        ax.grid(False)
        ax.set_xticks(range(len(column_labels)), column_labels, rotation=40, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(row_labels)), row_labels)
        if texts is not None:
            norm = im.norm
            for (i, j), value in np.ndenumerate(a):
                content = texts[i][j]
                rgb = cmap(norm(value))[:3] if np.isfinite(value) else (1., 1., 1.)
                luminance = .2126*rgb[0] + .7152*rgb[1] + .0722*rgb[2]
                color = "white" if luminance < .46 else "black"
                ax.text(j, i, content, ha="center", va="center", fontsize=fontsize, color=color, weight="bold")
        if colorbar:
            cb = ax.figure.colorbar(im, ax=ax, fraction=.04, pad=.025)
            cb.set_label(colorbar, weight="bold", fontsize=14)
        return im

    def coverage(self, table, total, complete):
        methods = [m for m in METHODS if m in set(table.method)]
        rows = list(BLOCKS)
        fractions = table.pivot(index="block", columns="method", values="fraction").reindex(index=rows, columns=methods).to_numpy()
        texts = []
        for block in rows:
            line = []
            for method in methods:
                g = table[(table.block == block) & (table.method == method)]
                line.append(f"{g.completed.iloc[0]:,}\n/{g.planned.iloc[0]:,}" if len(g) else "—")
            texts.append(line)
        fig, (ax,) = self.canvas(f"Experiment coverage · {complete:,} / {total:,} trials", height=8.8, width=18)
        self.heatmap(ax, fractions, rows, [("GDE3*" if m == "GDE3" else LABELS.get(m, m)) for m in methods], texts, 0, 1,
                     colorbar="Completion", fontsize=11)
        self.save(fig, 1, "coverage", "Completed and planned trials by experiment block and optimizer slot. Grey cells are not planned.",
                  "Cells: completed / planned. GDE3* uses SPEA2 on binary ZDT5. A complete cell need not imply a complete comparison.")

    def main_ranking(self, ranks):
        sub = ranks[ranks.study == "main"] if len(ranks) else ranks
        if not len(sub):
            return self.skip(2, "main_ranking", "No main problem has 30 matched seeds for all ten methods.")
        levels = [m for m in METHODS if m in set(sub.level)]
        order = sub[sub.metric == "hv"].set_index("level").family_balanced_rank.reindex(levels).sort_values().index
        fig, axes = self.canvas("Main benchmark · balanced method ranks", columns=2, height=9)
        for ax, metric in zip(axes, METRICS):
            g = sub[sub.metric == metric].set_index("level").loc[order]
            self.point_plot(ax, [LABELS.get(v, v) for v in order], g.family_balanced_rank, g.ci_low, g.ci_high,
                            color=[.82 if v == "FULL" else .35 for v in order], limits=(.7, len(levels)+.3))
            ax.set_title(METRIC_NAMES[metric], weight="bold", pad=12)
            ax.set_xlabel("Mean rank · lower is better", weight="bold", labelpad=8)
        n, planned = int(sub.cases.iloc[0]), int(sub.planned_cases.iloc[0])
        axes[1].tick_params(axis="y", labelleft=False)
        readiness = "INTERIM" if n < planned else "COMPLETE"
        self.save(fig, 2, "main_ranking", "Ranks of per-problem medians. Equal weight per represented family. 95% intervals resample problems within families.",
                  f"{readiness} · {n}/{planned} eligible problems · {int(sub.families.iloc[0])}/6 families · 30 matched seeds per method. ZDT5 is separate.")

    def family_ranking(self, rank_cases):
        sub = rank_cases[rank_cases.study == "main"] if len(rank_cases) else rank_cases
        if not len(sub):
            return self.skip(3, "family_ranking", "No balanced main cases yet.")
        levels = [m for m in METHODS if m in set(sub.level)]
        fig, axes = self.canvas("Main benchmark · ranks within each family", columns=2, height=8.5, width=22)
        for ax, metric in zip(axes, METRICS):
            g = sub[sub.metric == metric]
            matrix = g.pivot_table(index="family", columns="level", values="rank", aggfunc="mean").reindex(index=FAMILIES, columns=levels).to_numpy()
            counts = g.groupby("family").case.nunique()
            labels = [f"{f}  (n={int(counts.get(f, 0))})" for f in FAMILIES]
            texts = [[f"{v:.1f}" if np.isfinite(v) else "—" for v in row] for row in matrix]
            self.heatmap(ax, matrix, labels, [LABELS.get(v, v) for v in levels], texts,
                         1, len(levels), cmap="viridis_r", fontsize=11)
            ax.set_title(METRIC_NAMES[metric], weight="bold", pad=12)
        axes[1].tick_params(axis="y", labelleft=False)
        self.save(fig, 3, "family_ranking", "Mean ranks within each family, using balanced complete cases only.",
                  "Lower ranks are better. n counts complete problems, not seeds. Missing families remain grey.")

    def wins(self, comparisons):
        sub = comparisons[(comparisons.study == "main") & comparisons.eligible_inference] if len(comparisons) else comparisons
        if not len(sub):
            return self.skip(4, "wins_losses", "No main FULL/baseline pair has 30 matched seeds.")
        levels = [m for m in METHODS if m in set(sub.level)]
        fig, axes = self.canvas("FULL vs baselines · paired tests by problem", columns=2, height=8.7)
        colors = [self.cmap(.78), "#D8D3C5", self.cmap(.12)]
        for ax, metric in zip(axes, METRICS):
            labels = []
            for i, level in enumerate(levels):
                g = sub[(sub.metric == metric) & (sub.level == level)]
                counts = [int((g.decision == label).sum()) for label in ("win", "no_difference_detected", "loss")]
                labels.append(f"{('GDE3*' if level == 'GDE3' else LABELS.get(level, level))}  (n={sum(counts)})")
                left = 0
                for count, color in zip(counts, colors):
                    ax.barh(i, count, left=left, color=color, alpha=.85, edgecolor="black", linewidth=.8, height=.66, zorder=3)
                    if count:
                        ax.text(left+count/2, i, str(count), va="center", ha="center", weight="bold", fontsize=13,
                                color="white" if color == colors[2] else "black")
                    left += count
            ax.set_yticks(range(len(levels)), labels)
            ax.invert_yaxis()
            ax.set_xlim(0, max(1, sub.problem.nunique()))
            ax.set_title(METRIC_NAMES[metric], weight="bold", pad=12)
            ax.set_xlabel("Problems", weight="bold", labelpad=8)
            ax.grid(axis="y", visible=False)
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(facecolor=c, edgecolor="black", label=t) for c, t in zip(colors, ("FULL wins", "No difference detected", "FULL loses"))],
                   loc="upper center", bbox_to_anchor=(.53, .925), ncol=3, frameon=False)
        axes[1].tick_params(axis="y", labelleft=False)
        self.save(fig, 4, "wins_losses", "FULL/control paired sign-flip tests; Holm correction across both endpoints and every planned main contrast.",
                  "Only 30 matched seeds enter a test. No difference detected is not an equivalence claim. GDE3* uses SPEA2 on ZDT5.")

    def effects(self, effects, number, slug, studies, title, custom_label=None):
        sub = effects[effects.study.isin(studies)] if len(effects) else effects
        if not len(sub):
            return self.skip(number, slug, "No common problem panel with 30 matched seeds for every displayed variant.")
        ids = list(dict.fromkeys(zip(sub.study, sub.level)))
        fig, axes = self.canvas(title, columns=2, height=max(7.7, 3.8+.51*len(ids)), width=20)
        for ax, metric in zip(axes, METRICS):
            selected = [sub[(sub.study == study) & (sub.level == level) & (sub.metric == metric)].iloc[0] for study, level in ids]
            labels = [custom_label(s, v) if custom_label else LABELS.get(v, v) for s, v in ids]
            self.point_plot(ax, labels, [r.paired_win_probability for r in selected],
                            [r.ci_low for r in selected], [r.ci_high for r in selected],
                            color=np.linspace(.2, .8, len(ids)), limits=(-.03, 1.03), neutral=.5)
            ax.set_title(METRIC_NAMES[metric], weight="bold", pad=12)
            ax.set_xlabel("P(default better) + ½ P(tie)", weight="bold", labelpad=8)
            ax.set_xticks([0, .25, .5, .75, 1])
        axes[1].tick_params(axis="y", labelleft=False)
        count_text = "; ".join(f"{s.removeprefix('sensitivity_')}: {int(g.cases.iloc[0])}/{int(g.planned_cases.iloc[0])}" for s, g in sub.groupby("study", sort=False))
        self.save(fig, number, slug, "Paired seed win probability for the predefined control. Equal-family mean and 95% problem-bootstrap intervals; no pooled raw metric scales.",
                  f"Matched problems · {count_text}. Right of 0.5 favors the default. Intervals describe this problem panel; no significance stars.")

    def faceted_ranks(self, ranks, study, number, slug, title, facets, methods):
        sub = ranks[ranks.study == study] if len(ranks) else ranks
        if not len(sub):
            return self.skip(number, slug, "No complete problem with all planned methods within a facet.")
        fig, axes = self.canvas(title, columns=2, height=8.2, width=20)
        offsets = np.linspace(-.21, .21, len(facets))
        for ax, metric in zip(axes, METRICS):
            for offset, facet, color in zip(offsets, facets, np.linspace(.15, .8, len(facets))):
                g = sub[(sub.facet == facet) & (sub.metric == metric)].set_index("level")
                if not len(g):
                    continue
                label = f"{facet}  (n={int(g.cases.iloc[0])})"
                for i, method in enumerate(methods):
                    if method not in g.index:
                        continue
                    row = g.loc[method]
                    if np.isfinite(row.ci_low) and np.isfinite(row.ci_high):
                        ax.plot([row.ci_low, row.ci_high], [i+offset]*2, color=self.cmap(color), alpha=.6, lw=2.7)
                    ax.scatter(row.family_balanced_rank, i+offset, color=self.cmap(color), edgecolors="black", s=85,
                               linewidths=.8, label=label if i == 0 else None, zorder=4)
            ax.set_yticks(range(len(methods)), [LABELS.get(v, v) for v in methods])
            ax.set_ylim(len(methods)-.45, -.6)
            ax.set_xlim(.65, len(methods)+.35)
            ax.set_title(METRIC_NAMES[metric], weight="bold", pad=12)
            ax.set_xlabel("Mean rank within facet · lower is better", weight="bold", labelpad=8)
            ax.grid(axis="y", visible=False)
            ax.legend(loc="upper left", bbox_to_anchor=(0, -.14), ncol=3, fontsize=12, frameon=False)
        axes[1].tick_params(axis="y", labelleft=False)
        self.save(fig, number, slug, "Family-balanced ranks calculated separately within each facet, with 30 matched seeds for all compared methods.",
                  "n counts eligible problems in each facet. Facets may have different interim coverage; comparisons across facets are descriptive.")

    def many(self, cells):
        sub = cells[cells.study == "many"]
        if not len(sub) or not (sub.n > 0).any():
            return self.skip(10, "many_objective", "No five-objective results at the cutoff.")
        problems = list(dict.fromkeys(sub.problem))
        methods = list(METHODS)
        fig, axes = self.canvas("Five objectives · per-problem medians", columns=2, height=8.8, width=24)
        for ax, metric in zip(axes, METRICS):
            g = sub[sub.metric == metric]
            values = g.pivot(index="problem", columns="level", values="median").reindex(index=problems, columns=methods).to_numpy()
            n = g.pivot(index="problem", columns="level", values="n").reindex(index=problems, columns=methods).fillna(0).to_numpy()
            # Each row has its own metric range, so colors cannot imply a shared raw scale.
            display = np.full_like(values, np.nan)
            for i, row in enumerate(values):
                ok = np.isfinite(row)
                x = np.log10(np.maximum(row[ok], 1e-12)) if metric == "igd_plus" else row[ok]
                display[i, ok] = (x-x.min())/(x.max()-x.min()) if len(x) and x.max() > x.min() else .5
                if metric == "igd_plus":
                    display[i, ok] = 1-display[i, ok]
            texts = [[f"{v:.2g}\n{int(n[i,j])}/30" if np.isfinite(v) else "—" for j, v in enumerate(row)] for i, row in enumerate(values)]
            self.heatmap(ax, display, problems, [LABELS.get(v, v) for v in methods], texts, 0, 1, fontsize=10)
            ax.set_title(METRIC_NAMES[metric], weight="bold", pad=12)
        axes[1].tick_params(axis="y", labelleft=False)
        readiness = "COMPLETE" if (sub.n == sub.expected_n).all() else "INTERIM"
        self.save(fig, 10, "many_objective", "Median HV and IGD+ for M=5. Each cell states its available run count; missing cells are grey.",
                  f"{readiness} · Colors are scaled within each problem (brighter is better); IGD+ uses a log color transform. Values are raw medians, not ranks.")

    def convergence(self, table):
        if not len(table):
            return self.skip(11, "convergence", "No common panel with all required methods, 30 seeds and identical observed FE points.")
        levels = list(table.level.unique())
        colors = {level: self.cmap(c) for level, c in zip(levels, np.linspace(.08, .9, len(levels)))}
        fig, axes = self.canvas("Convergence · ranks at identical FE counts", columns=2, height=8.7, width=20)
        for ax, metric in zip(axes, METRICS):
            for level in levels:
                g = table[(table.metric == metric) & (table.level == level)].sort_values("fe")
                x = g.fe.to_numpy(float)
                y = g.family_balanced_rank.to_numpy(float)
                ax.plot(x, y, color=colors[level], marker="o", markersize=6,
                        markeredgecolor="black", markeredgewidth=.5,
                        lw=3.3 if level == "FULL" else 2., label=LABELS.get(level, level))
                ax.fill_between(x, g.ci_low.to_numpy(float), g.ci_high.to_numpy(float), color=colors[level], alpha=.09)
            ax.set_xscale("log")
            ax.set_xticks([400, 10000, 25200, 100000], ["400", "10k", "25.2k", "100k"])
            ax.set_ylim(len(levels)+.3, .7)
            ax.set_xlabel("Objective-vector evaluations", weight="bold", labelpad=8)
            ax.set_ylabel("Mean rank · lower is better", weight="bold", labelpad=8)
            ax.set_title(METRIC_NAMES[metric], weight="bold", pad=12)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, .925), ncol=7, fontsize=12, frameon=False)
        self.save(fig, 11, "convergence", "Family-balanced ranks of per-problem medians on the same complete problem/seed panel at every FE point. 95% problem-bootstrap bands.",
                  f"{int(table.cases.iloc[0])}/{int(table.planned_cases.iloc[0])} eligible problems. Actual FE: 400, 10,000, 25,200, 50,000, 100,000. No interpolation.")

    def overhead(self, table):
        if not len(table):
            return self.skip(12, "migration_overhead", "No balanced main problem panel for migration and cost summaries.")
        methods = [m for m in METHODS if m in set(table.method)]
        policies = [m for m in methods if m not in SINGLETONS]
        t = table.set_index("method")
        fig, axes = self.canvas("Migration activity and measured time composition", columns=2, height=8.8, width=21)
        for offset, column, color, label in ((-.16, "sent_per_10k", .35, "Sent"), (.16, "accepted_per_10k", .8, "Accepted")):
            axes[0].barh(np.arange(len(policies))+offset, t.loc[policies, column], height=.28,
                         color=self.cmap(color), alpha=.85, edgecolor="black", linewidth=.8, label=label, zorder=3)
        axes[0].set_yticks(range(len(policies)), [LABELS.get(v, v) for v in policies])
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Individuals per 10,000 FE", weight="bold", labelpad=8)
        axes[0].set_title("Logical migration", weight="bold", pad=12)
        axes[0].legend(frameon=False, loc="upper right")
        left = np.zeros(len(methods))
        columns = ("control_fraction", "integration_fraction", "diagnostic_fraction")
        fractions = np.asarray([t.loc[methods, c].to_numpy(float) for c in columns])
        fractions = np.vstack([fractions, np.maximum(0, 1-fractions.sum(axis=0))])
        for values, color, label in zip(fractions, (self.cmap(.15), self.cmap(.42), self.cmap(.78), "#D8D3C5"),
                                         ("Controller", "Integration", "Diagnostics", "Remaining time")):
            axes[1].barh(range(len(methods)), values*100, left=left*100, color=color, alpha=.85,
                         edgecolor="black", linewidth=.65, label=label, height=.65, zorder=3)
            left += values
        axes[1].set_yticks(range(len(methods)), [LABELS.get(v, v) for v in methods])
        axes[1].invert_yaxis()
        axes[1].set_xlim(0, 100)
        axes[1].set_xlabel("Share of measured trial wall time (%)", weight="bold", labelpad=8)
        axes[1].set_title("Time composition", weight="bold", pad=12)
        axes[1].legend(loc="upper left", bbox_to_anchor=(0, -.14), ncol=2, fontsize=12, frameon=False)
        self.save(fig, 12, "migration_overhead", "Per-problem seed medians, averaged with equal family weights. Sent and accepted migrants are distinct. Costs describe the shared server.",
                  f"{int(table.problems.iloc[0])} matched problems. Payload is logical array traffic. Sequential islands within each trial; no physical speedup claim.")

    def finish(self):
        write_csv(self.folder / "figure_index.csv", self.index)
        paragraphs = [f"# TherMOO figures · {self.version}", "", "PNG: 320 dpi. PDF: vector, embedded TrueType fonts.", ""]
        for row in self.index:
            paragraphs += [f"## {row['file']} · {row['status']}", "", row["caption"], "", row["note"], ""]
        (self.folder / "captions.md").write_text("\n".join(paragraphs))


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
    (stats_dir / "REPORT.md").write_text("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("THERMOO_ROOT", "/home/tahiti/Malashin_Projects/TherMOO")))
    parser.add_argument("--bootstrap", type=int, default=4000, help="Bootstrap resamples; production default 4000")
    parser.add_argument("--permutations", type=int, default=50000, help="Permutation resamples; production default 50000")
    args = parser.parse_args(argv)
    if args.bootstrap < 200 or args.permutations < 1000:
        parser.error("At least 200 bootstrap and 1000 permutation resamples are required")
    root = args.root.expanduser().resolve()
    if not (root / "thermoo" / "plan.py").is_file():
        parser.error(f"TherMOO project not found: {root}")
    os.environ["THERMOO_ROOT"] = str(root)
    sys.path.insert(0, str(root))
    version, stats_dir, figures_dir = allocate_version(root)
    manifest = dict(status="RUNNING", version=version, analyzer_version=VERSION,
                    analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    cutoff_ns=START_NS, cutoff_utc=datetime.fromtimestamp(START_NS / 1e9, timezone.utc).isoformat(),
                    root=str(root), statistics=str(stats_dir), figures=str(figures_dir),
                    protocol=dict(bootstrap=args.bootstrap, permutations=args.permutations, alpha=.05,
                                  minimum_paired_runs=30, inference="paired_sign_flip_mean", correction="Holm_planned_block_both_endpoints"),
                    python=sys.version, dependencies={})
    atomic_json(stats_dir / "manifest.json", manifest)
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
        raw, histories, issues, missing, provenance = capture(root, fingerprint, planned_jobs, family, stats_dir, START_NS)
        manifest.update(accepted_trials=len(raw), missing_trials=len(missing))
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
        fig = Figures(figures_dir, version, START_NS)
        fig.coverage(coverage, len(planned), len(raw))
        fig.main_ranking(ranks)
        fig.family_ranking(rank_cases)
        fig.wins(comparisons)
        fig.effects(effects, 5, "ablations", ["ablation"], "Ablations · FULL vs reduced controller")
        def sensitivity_label(study, level):
            factor = study.removeprefix("sensitivity_")
            return {"interval": "Interval", "rho": "Migration fraction", "temperature": "Temperature", "quality": "Quality"}[factor] + ": " + level.replace("rank_", "")
        fig.effects(effects, 6, "sensitivity", [f"sensitivity_{k}" for k in ("interval", "rho", "temperature", "quality")],
                    "One-factor sensitivity · default vs variation", sensitivity_label)
        fig.effects(effects, 7, "composition", ["composition_FULL"], "Island composition · mixed vs homogeneous",
                    lambda study, level: LABELS.get(level, level))
        fig.faceted_ranks(ranks, "islands", 8, "island_count", "Island count · fixed population and FE budget", ["K=4", "K=8", "K=16"], POLICIES)
        fig.faceted_ranks(ranks, "dimensions", 9, "dimension", "Decision-space dimension · method ranks", ["Default", "D=50", "D=100"], DIM_METHODS)
        fig.many(cells)
        fig.convergence(convergence)
        fig.overhead(timing)
        fig.finish()
        manifest.update(created_figures=sum(row["status"] == "created" for row in fig.index),
                        figure_slots=12, audit_records=len(issues),
                        eligible_paired_tests=int(comparisons.eligible_inference.sum()))
        (stats_dir / "PROTOCOL.md").write_text(PROTOCOL)
        manifest["provenance_files"] = provenance_archive(root, stats_dir, manifest)
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
