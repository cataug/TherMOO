#!/usr/bin/env python3
"""TherMOO case atlas: one problem, M, D, FE budget and setting per page.

Read-only scientific intake from both original experiment locations, or from
one previously frozen STATS/vN snapshot. Never starts an optimization run.
Writes exclusively under --root/STATS/vN and --root/FIGURES/vN.
"""
import time
CUTOFF_NS = time.time_ns()

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import html
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
from functools import lru_cache

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

BG, AXBG, GRID = "#FAF8F2", "#F4F1E8", "#D8D3C5"
METRICS = ("hv", "igd_plus")
BASE = ("main", "many", "dimensions", "islands", "homogeneous")
SETTINGS = ("problem", "m", "d", "budget", "population", "islands", "composition",
            "interval", "rho", "temperature", "quality", "archive", "hv_epsilon", "js_epsilon")
COLORS = {"FULL": "#356F32", "NO_MIG": "#414487", "ELITE": "#238A8D", "RANDOM": "#7652A2",
          "NSGA2": "#440154", "MOEAD": "#31688E", "GDE3": "#26828E", "SMS": "#35B779",
          "HV_GATE": "#6CCE59", "CROWDING": "#9FDA3A", "NSGA3": "#7AD151", "RVEA": "#B5DE2B"}
ORDER = ("FULL", "NO_MIG", "ELITE", "RANDOM", "HV_GATE", "CROWDING", "NSGA2", "MOEAD", "GDE3", "SMS", "NSGA3", "RVEA")
LABEL = {"NSGA2": "NSGA-II", "MOEAD": "MOEA/D", "GDE3": "GDE3", "SMS": "SMS-EMOA",
         "NSGA3": "NSGA-III", "NO_MIG": "No migration", "HV_GATE": "HV gate"}
SEGMENTS = (("both", "#35B779", "FULL: both"), ("hv_only", "#8DD3A6", "HV only"),
            ("igd_only", "#74A9CF", "IGD+ only"), ("opponent_both", "#482878", "Other: both"),
            ("other", "#D7D3C9", "Ties / other"))


def log(message):
    print(message, flush=True)


def freeze(path):
    """Reject a file published or changed after the command started."""
    before = path.stat()
    if max(before.st_mtime_ns, before.st_ctime_ns) > CUTOFF_NS:
        raise ValueError("after_cutoff")
    blob = path.read_bytes()
    after = path.stat()
    ident = lambda s: (s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if ident(before) != ident(after):
        raise ValueError("changed_during_read")
    return blob


def load_plan(path):
    spec = importlib.util.spec_from_file_location("thermoo_atlas_plan", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return {j.signature(): j for j in module.jobs()}


def read_live(roots):
    engine = next((r for r in roots if (r / "thermoo/plan.py").is_file()), None)
    if engine is None:
        raise RuntimeError("Original thermoo/plan.py is absent from both input roots; use --snapshot")
    planned = load_plan(engine / "thermoo/plan.py")
    found, conflicts, issues = {}, set(), Counter()
    for root in roots:
        for path in sorted((root / "outputs/full").glob("*/*/result.json")):
            rid, fp = path.parent.name, path.parents[1].name
            try:
                job = planned.get(rid)
                if job is None:
                    raise ValueError("not_in_original_plan")
                data = json.loads(freeze(path))
                if data.get("status") != "complete" or data.get("run_id") != rid or data.get("fingerprint") != fp:
                    raise ValueError("incomplete_or_identity_mismatch")
                if data.get("job") != asdict(job) or data.get("evaluations") != job.budget or data.get("metrics", {}).get("fe") != job.budget:
                    raise ValueError("plan_or_budget_mismatch")
                values = [float(data["metrics"][m]) for m in METRICS]
                if any(not np.isfinite(x) or x < 0 for x in values):
                    raise ValueError("invalid_metrics")
                archive = path.parent / "population.npz"
                stamp = archive.stat()
                if not stamp.st_size or max(stamp.st_mtime_ns, stamp.st_ctime_ns) > CUTOFF_NS:
                    raise ValueError("missing_or_unfrozen_archive")
                if not re.fullmatch(r"[0-9a-f]{64}", str(data.get("initial_hash", ""))):
                    raise ValueError("invalid_initial_hash")
                algs = data.get("algorithms", [])
                expected_k = 1 if job.method in ("NSGA2", "MOEAD", "GDE3", "SMS", "NSGA3", "RVEA") else job.islands
                if len(algs) != expected_k:
                    raise ValueError("algorithm_count_mismatch")
                row = asdict(job)
                row.update(run_id=rid, m=job.objectives, d=job.variables,
                           hv=values[0], igd_plus=values[1], fingerprint=fp,
                           initial_hash=data["initial_hash"], actual_algorithms="|".join(algs),
                           source_result=str(path), seconds=float(data["seconds"]),
                           sent=int(data.get("sent", 0)), accepted=int(data.get("accepted", 0)),
                           control_seconds=float(data.get("control_seconds", 0)),
                           integration_seconds=float(data.get("integration_seconds", 0)))
                if not np.isfinite(row["seconds"]) or row["seconds"] <= 0 or row["accepted"] > row["sent"]:
                    raise ValueError("invalid_runtime_or_counters")
                key = (fp, rid)
                science = (row["hv"], row["igd_plus"], row["initial_hash"], row["sent"], row["accepted"], row["actual_algorithms"])
                if key in conflicts:
                    continue
                if key in found:
                    if science != found[key][1]:
                        found.pop(key)
                        conflicts.add(key)
                        issues["conflicting_duplicate"] += 1
                    elif str(root) == str(roots[0]):
                        found[key] = (row, science)
                else:
                    found[key] = (row, science)
            except (OSError, ValueError, TypeError, KeyError) as error:
                issues[str(error)] += 1
        log(f"[READ] {root}: valid copies so far={len(found):,}")
    counts = Counter(k[0] for k in found)
    if not counts:
        raise RuntimeError(f"No completed original trials in {', '.join(map(str, roots))}")
    fp = counts.most_common(1)[0][0]
    if len(counts) > 1:
        log(f"[COHORT] selected fingerprint {fp} ({counts[fp]:,} results); ignored other fingerprints: {dict(counts)}")
    records = [v[0] for k, v in found.items() if k[0] == fp]
    return pd.DataFrame(records), {"mode": "live", "fingerprint": fp, "planned": len(planned),
                                   "source_roots": list(map(str, roots)), "excluded": dict(issues)}


def read_snapshot(folder):
    folder = folder.resolve()
    raw = pd.read_csv(folder / "per_run.csv", low_memory=False)
    info = json.loads((folder / "manifest.json").read_text())
    required = set(SETTINGS) | {"seed", "run_id", "block", "method", "initial_hash", "hv", "igd_plus", "seconds"}
    if not required.issubset(raw.columns):
        raise ValueError("The given snapshot lacks original experiment per_run columns")
    if raw.empty:
        raise ValueError("The selected snapshot has no completed runs")
    fp = info.get("fingerprint")
    if not fp:
        raise ValueError("The selected snapshot has no experiment fingerprint")
    raw["fingerprint"] = fp
    for col in ("sent", "accepted", "control_seconds", "integration_seconds"):
        if col not in raw:
            raw[col] = 0
    return raw, {"mode": "snapshot", "fingerprint": fp, "planned": int(info.get("planned_trials", -1)),
                 "source_snapshot": str(folder), "cutoff_utc": info.get("cutoff_utc")}


def reserve(root):
    stats, figures = root / "STATS", root / "FIGURES"
    stats.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    with (stats / ".versions.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        numbers = [int(p.name[1:]) for base in (stats, figures) for p in base.iterdir()
                   if p.is_dir() and re.fullmatch(r"v[0-9]+", p.name)]
        n = max(numbers, default=0) + 1
        while (stats / f"v{n}").exists() or (figures / f"v{n}").exists():
            n += 1
        sd, fd = stats / f"v{n}", figures / f"v{n}"
        sd.mkdir(); fd.mkdir()
    return sd, fd


def settings_key(row, omit=()):
    return {k: row[k] for k in SETTINGS if k not in omit}


def make_views(raw):
    values = []
    by_problem = set(raw.loc[raw.block.isin(("ablation", "sensitivity")), "problem"])
    for row in raw.to_dict("records"):
        study, block = row["block"], row["block"]
        if block in BASE:
            values.append(dict(row, study=study, level=row["method"], config=settings_key(row)))
            if block == "main" and row["method"] == "FULL" and row["problem"] in by_problem:
                values.append(dict(row, study="ablation", level="FULL", config=settings_key(row)))
                for field in ("interval", "rho", "temperature", "quality"):
                    values.append(dict(row, study=f"sensitivity_{field}", level="default",
                                       config=settings_key(row, (field,))))
        elif block == "ablation":
            values.append(dict(row, study="ablation", level=row["ablation"], config=settings_key(row)))
        elif block == "sensitivity":
            defaults = {"interval": 10, "rho": .1, "temperature": 1., "quality": "rank_crowding"}
            changed = [k for k, v in defaults.items() if row[k] != v]
            if len(changed) == 1:
                field = changed[0]
                values.append(dict(row, study=f"sensitivity_{field}", level=f"{field}={row[field]}",
                                   config=settings_key(row, (field,))))
    for row in values:
        cfg = json.dumps(row.pop("config"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        row["case_key"] = hashlib.sha256((row["study"] + cfg).encode()).hexdigest()[:16]
        row["case_config"] = cfg
        if row["problem"] == "ZDT5" and row["level"] == "GDE3" and "SPEA2" in str(row.get("actual_algorithms", "")):
            row["level"] = "SPEA2 (binary)"
    frame = pd.DataFrame(values)
    if frame.duplicated(["study", "case_key", "level", "seed"]).any():
        raise ValueError("Duplicate seed within one fixed scientific configuration")
    return frame


def oriented(a, b):
    dh = a["hv"] - b["hv"]
    di = b["igd_plus"] - a["igd_plus"]
    tolerance_h = 1e-12 * max(1.0, abs(a["hv"]), abs(b["hv"]))
    tolerance_i = 1e-12 * max(1.0, abs(a["igd_plus"]), abs(b["igd_plus"]))
    return dh, di, tolerance_h, tolerance_i


def paired(panel, control="FULL"):
    if control not in set(panel.level):
        return pd.DataFrame(), pd.DataFrame()
    base = panel[panel.level == control].set_index("seed")
    data, summary = [], []
    for level, other in panel[panel.level != control].groupby("level", sort=False):
        variants = other.set_index("seed")
        for seed in base.index.intersection(variants.index):
            a, b = base.loc[seed], variants.loc[seed]
            if a.initial_hash != b.initial_hash:
                continue
            dh, di, th, ti = oriented(a, b)
            both = dh > th and di > ti
            which = ("both" if both else "hv_only" if dh > th else "igd_only" if di > ti
                     else "opponent_both" if dh < -th and di < -ti else "other")
            data.append(dict(level=level, seed=int(seed), hv_gain=dh, igd_gain=di, result=which,
                             time_ratio=float(a.seconds)/float(b.seconds),
                             full_time=float(a.seconds), other_time=float(b.seconds),
                             sent=int(a.sent), accepted=int(a.accepted),
                             control_fraction=float(a.control_seconds)/float(a.seconds)))
    detail = pd.DataFrame(data)
    if detail.empty:
        return detail, pd.DataFrame()
    for level, group in detail.groupby("level", sort=False):
        counts = group.result.value_counts()
        summary.append(dict(level=level, n_paired=len(group), both=int(counts.get("both", 0)),
                            hv_only=int(counts.get("hv_only", 0)), igd_only=int(counts.get("igd_only", 0)),
                            opponent_both=int(counts.get("opponent_both", 0)), other=int(counts.get("other", 0)),
                            median_hv_gain=float(group.hv_gain.median()), median_igd_gain=float(group.igd_gain.median()),
                            median_time_ratio=float(group.time_ratio.median()),
                            median_sent=float(group.sent.median()), median_accepted=float(group.accepted.median())))
    return detail, pd.DataFrame(summary)


def name(level):
    return LABEL.get(level, level)


def color(level, levels=None):
    if level in COLORS:
        return COLORS[level]
    levels = sorted(levels or [level])
    return plt.get_cmap("viridis")(.13 + .75 * (levels.index(level) / max(1, len(levels)-1)))


def ordered(levels):
    return sorted(set(levels), key=lambda s: (ORDER.index(s) if s in ORDER else len(ORDER), str(s)))


def style(axes):
    for ax in np.ravel(axes):
        ax.set_facecolor(AXBG)
        ax.grid(alpha=.7, lw=.8, color=GRID, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("black"); spine.set_linewidth(1.1)
        ax.tick_params(labelsize=14, colors="black")
        ax.xaxis.label.set_fontweight("bold"); ax.yaxis.label.set_fontweight("bold")


def title(panel):
    r = panel.iloc[0]
    suffix = f" · K={int(r.islands)} · {r.composition}" if r.study == "homogeneous" or int(r.islands) != 4 else ""
    if r.study.startswith("sensitivity_"):
        suffix += " · " + r.study.split("_", 1)[1]
    return f"{r.problem} · M={int(r.m)} · D={int(r.d)} · FE={int(r.budget):,}{suffix}"


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + extension), dpi=200, bbox_inches="tight",
                    facecolor=BG, pad_inches=.18)
    plt.close(fig)


def quality_and_outcomes(panel, detail, summary, target):
    levels = ordered(panel.level)
    if len(levels) < 2:
        return False
    fig, (left, right) = plt.subplots(1, 2, figsize=(20, max(8, 4.7 + .37*len(summary))),
                                      facecolor=BG, gridspec_kw=dict(width_ratios=(1.02, 1.18)))
    style((left, right))
    for level in levels:
        sub = panel[panel.level == level]
        col = color(level, levels)
        left.scatter(sub.hv, sub.igd_plus, s=48, color=col, alpha=.4,
                     edgecolor="black", linewidth=.35, zorder=3)
        left.scatter(sub.hv.median(), sub.igd_plus.median(), s=260 if level == "FULL" else 140,
                     marker="*" if level == "FULL" else "o", color=col,
                     edgecolor="black", linewidth=1.4, zorder=6, label=f"{name(level)}  n={len(sub)}")
    left.invert_yaxis()
    left.set_xlabel("Final HV  ↑", fontsize=17); left.set_ylabel("Final IGD+  ↓", fontsize=17)
    left.set_title("Both quality metrics, each dot = one seed", weight="bold", fontsize=19, pad=16)
    right.set_title("Same-seed FULL versus each alternative", weight="bold", fontsize=19, pad=16)
    if len(summary):
        summary = summary.set_index("level").reindex([s for s in levels if s in set(summary.level)])
        y = np.arange(len(summary))
        lefts = np.zeros(len(summary))
        for key, col, _ in SEGMENTS:
            fractions = summary[key].to_numpy(float) / summary.n_paired.to_numpy(float)
            right.barh(y, fractions, left=lefts, color=col, edgecolor="black", linewidth=.35,
                       height=.7, label=key, zorder=3)
            lefts += fractions
        right.set_yticks(y, [name(s) for s in summary.index], fontsize=15)
        right.invert_yaxis(); right.set_xlim(0, 1.12)
        right.set_xticks([0, .25, .5, .75, 1], ["0", "25", "50", "75", "100%"])
        right.set_xlabel("Fraction of matched seeds", fontsize=17)
        for ypos, item in enumerate(summary.itertuples()):
            right.text(1.015, ypos, f"n={item.n_paired}", va="center", fontsize=13, weight="bold",
                       bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=.15"))
    else:
        right.text(.5, .5, "No same-seed pairs with matching starts", transform=right.transAxes,
                   ha="center", va="center", fontsize=17)
        right.set_yticks([]); right.set_xticks([])
    h, lab = left.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", bbox_to_anchor=(.5, .01), ncol=4, fontsize=14,
               frameon=False, columnspacing=1.4)
    fig.legend(handles=[Patch(facecolor=col, edgecolor="black", label=label) for _, col, label in SEGMENTS],
               loc="upper center", bbox_to_anchor=(.5, .915), ncol=5, fontsize=13, frameon=False)
    fig.suptitle(title(panel), fontsize=25, weight="bold", y=.985)
    fig.subplots_adjust(left=.095, right=.96, bottom=.2, top=.80, wspace=.34)
    save(fig, target)
    return True


def bootstrap_interval(values, seed):
    x = np.asarray(values, dtype=float)
    if len(x) < 3:
        return None
    rng = np.random.default_rng(seed)
    samples = np.median(x[rng.integers(0, len(x), (500, len(x)))], axis=1)
    return tuple(np.quantile(samples, [.025, .975]))


def cost_effect(panel, detail, target):
    if detail.empty:
        return False
    levels = ordered(detail.level)
    fig, axes = plt.subplots(1, 3, figsize=(20, max(7.5, 3.5 + .66 * len(levels))), facecolor=BG)
    style(axes)
    unit = (("hv_gain", "HV gain  ↑"), ("igd_gain", "IGD+ reduction  ↑"),
            ("time_ratio", "Time ratio  ↓"))
    for ax, (field, xlabel) in zip(axes, unit):
        ax.axvline(1 if field == "time_ratio" else 0, lw=1.5, c="black", ls="--", zorder=2)
        for i, level in enumerate(levels):
            group = detail[detail.level == level]
            values = group[field].to_numpy(float)
            center = float(np.median(values))
            if not np.isfinite(center):
                continue
            c = color(level, levels)
            ci = bootstrap_interval(values, 1001+i)
            if ci:
                ax.plot(ci, [i, i], color=c, lw=2.5, zorder=4)
                ax.scatter(ci, [i, i], s=15, color=c, zorder=5)
            ax.scatter(center, i, s=190, color=c, edgecolor="black", lw=1.1, zorder=6)
            ax.axhline(i, color=GRID, lw=.6, zorder=1)
        ax.set_yticks(range(len(levels)), [name(level) + f"  n={len(detail[detail.level==level])}"
                                                for level in levels] if ax is axes[0] else [""] * len(levels))
        ax.invert_yaxis()
        ax.set_xlabel(xlabel, fontsize=17)
        ax.set_title("Paired HV difference" if field == "hv_gain" else
                     "Paired IGD+ difference" if field == "igd_gain" else
                     "FULL / alternative runtime", fontsize=18, weight="bold", pad=16)
        if field == "time_ratio":
            ax.set_xscale("log", base=2)
            ax.text(.98, .03, "< 1: FULL faster", transform=ax.transAxes, ha="right", fontsize=12,
                    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=.18"))
    fig.suptitle(title(panel) + " · paired benefit and cost", fontsize=23, weight="bold", y=.985)
    fig.text(.5, .024, "Positive quality values favor FULL. Dots are matched-seed medians; whiskers: bootstrap 95% intervals (n ≥ 3)."
             " Runtime is affected by concurrent server load.", ha="center", fontsize=12)
    fig.subplots_adjust(left=.20, right=.975, top=.82, bottom=.17, wspace=.22)
    save(fig, target)
    return True


def migration_diagnostic(panel, detail, target):
    """Within-setting association only. Migration timing is endogenous."""
    if detail.empty:
        return False
    pairs = detail[detail.level == "NO_MIG"].copy()
    if len(pairs) < 8 or pairs.accepted.nunique() < 3 or pairs.sent.max() <= 0:
        return False
    denominator = float(panel.iloc[0].budget) / 10000
    x = pairs.accepted.to_numpy(float) / denominator
    acceptance = np.divide(pairs.accepted, pairs.sent, out=np.zeros(len(pairs), float),
                           where=pairs.sent.to_numpy() > 0)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8.5), facecolor=BG)
    style(axes)
    points = None
    for ax, metric, label in zip(axes, ("hv_gain", "igd_gain"), ("HV gain  ↑", "IGD+ reduction  ↑")):
        y = pairs[metric].to_numpy(float)
        ax.axhline(0, color="black", lw=1.4, ls="--")
        points = ax.scatter(x, y, c=acceptance, vmin=0, vmax=1, cmap="viridis", s=120,
                            edgecolors="black", linewidth=.7, zorder=4)
        rho = pd.Series(x).corr(pd.Series(y), method="spearman")
        text = f"n={len(pairs)}  ·  rank correlation={rho:.2f}" if np.isfinite(rho) else f"n={len(pairs)}"
        ax.text(.035, .965, text, transform=ax.transAxes, va="top", fontsize=15, weight="bold",
                bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=.25"))
        ax.set_xlabel("FULL: accepted migrants per 10,000 FE", fontsize=17)
        ax.set_ylabel(label, fontsize=17)
        ax.set_title(label + " against no migration", fontsize=19, weight="bold", pad=15)
    fig.colorbar(points, ax=axes, fraction=.025, pad=.015, label="Accepted / sent migrants")
    median_share = float(pairs.control_fraction.median()) * 100
    fig.suptitle(title(panel) + " · migration diagnostic", fontsize=24, weight="bold", y=.985)
    fig.text(.5, .027, f"Same-seed FULL − no migration; median controller time: {median_share:.2f}% of run. "
             "Correlation is descriptive, not causal.", ha="center", fontsize=13)
    fig.subplots_adjust(left=.10, right=.88, top=.83, bottom=.16, wspace=.28)
    save(fig, target)
    return True


class Histories:
    def __init__(self, raw, snapshot=None):
        self.paths = dict(zip(raw.run_id.astype(str), raw.source_result.astype(str))) if "source_result" in raw else {}
        self.snapshot = snapshot
        self.frozen = {}

    def prepare(self, selected_ids):
        if not self.snapshot:
            return
        src = self.snapshot / "snapshot/histories.jsonl.gz"
        if not src.is_file():
            log("[HISTORY] Frozen history file missing; convergence will be skipped")
            return
        with gzip.open(src, "rt", encoding="utf-8") as stream:
            for i, line in enumerate(stream, 1):
                item = json.loads(line)
                rid = str(item["run_id"])
                if rid in selected_ids:
                    history = json.loads(item["data_json"])
                    self.frozen[rid] = {int(r["fe"]): (float(r["hv"]), float(r["igd_plus"]))
                                        for r in history if int(r.get("fe", -1)) in (400, 10000, 25200, 50000, 100000)
                                        and "hv" in r and "igd_plus" in r}
                if i % 5000 == 0:
                    log(f"[HISTORY] scanned={i:,}; retained={len(self.frozen):,}")

    @lru_cache(maxsize=4096)
    def get(self, rid):
        if self.snapshot:
            return self.frozen.get(str(rid), {})
        else:
            path = Path(self.paths[str(rid)]).parent / "history.json"
            try:
                values = json.loads(freeze(path))
            except (OSError, ValueError, TypeError):
                return {}
        out = {}
        for item in values:
            try:
                fe = int(item["fe"])
                hv, igd = float(item["hv"]), float(item["igd_plus"])
                if np.isfinite(hv) and np.isfinite(igd):
                    out[fe] = (hv, igd)
            except (KeyError, TypeError, ValueError):
                pass
        return out


def choose_lines(panel):
    levels = ordered(panel.level)
    preferred = ["FULL", "NO_MIG", "ELITE", "SMS", "NSGA2", "default"]
    selected = [m for m in preferred if m in levels]
    return (selected + [m for m in levels if m not in selected])[:5]


def convergence(panel, detail, histories, target):
    levels = choose_lines(panel)
    selected = panel[panel.level.isin(levels)]
    grids = set()
    for r in selected.itertuples():
        grids.update(histories.get(str(r.run_id)).keys())
    standard = [fe for fe in (400, 10000, 25200, 50000, 100000) if fe in grids]
    # Actual checkpoints only; no interpolation, especially across differing FE budgets.
    grid = standard if len(standard) >= 3 else sorted(grids)[:20]
    if len(grid) < 3:
        return False
    values = {}
    for level in levels:
        for fe in grid:
            xy = []
            for r in selected[selected.level == level].itertuples():
                item = histories.get(str(r.run_id)).get(fe)
                if item is not None:
                    xy.append(item)
            if xy:
                values[level, fe] = np.array(xy)
    if not any(sum((level, fe) in values for fe in grid) >= 3 for level in levels):
        return False
    fig, axes = plt.subplots(1, 3, figsize=(21, 8), facecolor=BG,
                             gridspec_kw=dict(width_ratios=(1, 1, 1.12)))
    style(axes)
    for level in levels:
        col = color(level, levels)
        for j in (0, 1):
            xx, median, low, high = [], [], [], []
            for fe in grid:
                if (level, fe) not in values:
                    continue
                arr = values[level, fe][:, j]
                xx.append(fe); median.append(np.median(arr))
                low.append(np.quantile(arr, .25)); high.append(np.quantile(arr, .75))
            if len(xx) >= 2:
                axes[j].plot(xx, median, marker="o", ms=6, lw=3 if level == "FULL" else 2,
                             color=col, label=name(level))
                if all(len(values[level, fe]) >= 3 for fe in xx):
                    axes[j].fill_between(xx, low, high, color=col, alpha=.10)
    axes[0].set_ylabel("HV  ↑", fontsize=17)
    axes[1].set_ylabel("IGD+  ↓", fontsize=17)
    axes[0].set_title("Quality over evaluations", fontsize=18, weight="bold")
    axes[1].set_title("Distance over evaluations", fontsize=18, weight="bold")
    ax = axes[2]
    ax.set_title("Paired lead on BOTH metrics", fontsize=18, weight="bold")
    reference = ("FULL" if "FULL" in levels else "default" if "default" in levels else None)
    if reference:
        controls = [x for x in ("NO_MIG", "ELITE") if x in levels and x != reference]
        if not controls:
            controls = [x for x in levels if x != reference][:2]
        for level in controls:
            xs, fractions, ns = [], [], []
            a = selected[selected.level == reference].set_index("seed")
            b = selected[selected.level == level].set_index("seed")
            common = [s for s in a.index.intersection(b.index) if a.loc[s, "initial_hash"] == b.loc[s, "initial_hash"]]
            for fe in grid:
                matched = [(histories.get(str(a.loc[s, "run_id"])).get(fe),
                            histories.get(str(b.loc[s, "run_id"])).get(fe)) for s in common]
                matched = [(u, v) for u, v in matched if u is not None and v is not None]
                if matched:
                    xs.append(fe); ns.append(len(matched))
                    fractions.append(np.mean([u[0] > v[0] and u[1] < v[1] for u, v in matched]))
            if len(xs) >= 2:
                ax.plot(xs, fractions, marker="o", lw=2.5, color=color(level, levels),
                        label=f"vs {name(level)} · n={min(ns)}–{max(ns)}")
    ax.axhline(.5, color="black", lw=1, ls="--")
    ax.set_ylim(-.06, 1.06)
    ax.set_ylabel("Fraction of matched seeds  ↑", fontsize=16)
    for axis in axes:
        axis.set_xlabel("Function evaluations (FE)", fontsize=16)
        axis.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(.5, .01), loc="lower center", ncol=5,
               frameon=False, fontsize=14)
    ax.legend(loc="best", fontsize=13, framealpha=.92)
    fig.suptitle(title(panel) + " · convergence", fontsize=23, weight="bold", y=.985)
    fig.subplots_adjust(left=.075, right=.97, top=.84, bottom=.18, wspace=.30)
    save(fig, target)
    return True


def gallery(figures, items, manifest):
    cards = []
    for item in items:
        links = []
        for entry in item["figures"]:
            src = html.escape(entry + ".png", quote=True)
            links.append(f'<a href="{src}"><img loading="lazy" src="{src}" alt="{html.escape(entry)}"></a>'
                         f'<a class="pdf" href="{html.escape(entry)}.pdf">PDF ↗</a>')
        problem = html.escape(item["problem"])
        study = html.escape(item["study"])
        note = html.escape(item["note"])
        cards.append(f'<article data-search="{problem.lower()} {study.lower()}"><h2>{html.escape(item["title"])} · {study}</h2>'
                     f'<p>{note}</p><div class="gallery">{"".join(links)}</div></article>')
    body = "\n".join(cards)
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>TherMOO · fixed-case atlas</title><style>
body{{background:{BG};color:#161616;font:18px system-ui,sans-serif;margin:0 auto;max-width:1700px;padding:28px}}
h1{{font-size:38px;margin:8px 0}}.sub{{color:#474747}}input{{font:22px system-ui;padding:12px;width:min(680px,90%);background:white;border:2px solid #282828;border-radius:8px}}
article{{margin:32px 0;padding:24px;border:1px solid #bdb9ad;border-radius:12px;background:{AXBG}}}
h2{{font-size:26px;margin:0 0 8px}}.gallery{{display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start}}.gallery a:has(img){{max-width:31%;min-width:290px}}
img{{width:100%;height:auto;background:white;border:1px solid #ccc}}.pdf{{font-size:15px;padding-top:8px}}
@media(max-width:900px){{.gallery a:has(img){{max-width:100%}}}}</style>
<h1>TherMOO · fixed-case atlas</h1><p class="sub">One problem, M, D, FE budget and experimental setting per card. {len(items)} cases · fingerprint {html.escape(manifest['fingerprint'])}. FULL is the original heuristic, not the neural controller.</p>
<label>Find problem or block<br><input id="find" type="search" placeholder="DTLZ2, ablation, many…"></label>
{body}<script>document.getElementById('find').oninput=e=>{{let q=e.target.value.toLowerCase();for(let x of document.querySelectorAll('article'))x.hidden=!x.dataset.search.includes(q)}}</script>
</html>'''
    (figures / "index.html").write_text(page, encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/home/tahiti/TherMOO"), help="Canonical WRITE root")
    p.add_argument("--input-root", type=Path, action="append", help="Read roots in priority order; defaults to canonical and legacy")
    p.add_argument("--snapshot", type=Path, help="Read frozen STATS/vN instead of live original outputs")
    p.add_argument("--problems", help="Optional comma-separated problem names")
    p.add_argument("--max-cases", type=int, default=0, help="QA only: cap plotted cases")
    args = p.parse_args(argv)
    root = args.root.expanduser().resolve()
    if args.snapshot:
        raw, source = read_snapshot(args.snapshot)
    else:
        roots = args.input_root or [root, Path("/home/tahiti/Malashin_Projects/TherMOO")]
        roots = list(dict.fromkeys(x.expanduser().resolve() for x in roots if x.expanduser().is_dir()))
        raw, source = read_live(roots)
    if args.problems:
        keep = {s.strip().upper() for s in args.problems.split(",")}
        raw = raw[raw.problem.str.upper().isin(keep)].copy()
    if raw.empty:
        raise RuntimeError("No completed results match the requested problem filter")
    views = make_views(raw)
    stats_dir, figures_dir = reserve(root)
    manifest = dict(version=stats_dir.name, root=str(root), status="RUNNING",
                    started_utc=datetime.fromtimestamp(CUTOFF_NS/1e9, timezone.utc).isoformat(),
                    **source)
    (stats_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    raw.to_csv(stats_dir / "per_run.csv", index=False)
    histories = Histories(raw, args.snapshot.resolve() if args.snapshot else None)
    cases = list(views.groupby(["study", "case_key"], sort=True))
    cases = [(key, g.copy()) for key, g in cases if g.level.nunique() >= 2]
    priority = {name: i for i, name in enumerate(("main", "many", "ablation", "dimensions", "islands", "homogeneous"))}
    cases.sort(key=lambda item: (priority.get(item[0][0], 6), item[1].problem.iloc[0],
                                 int(item[1].m.iloc[0]), int(item[1].d.iloc[0]), item[0][0], item[0][1]))
    if args.max_cases:
        cases = cases[:args.max_cases]
    if args.snapshot:
        ids = {str(rid) for _, g in cases for rid in g[g.level.isin(choose_lines(g))].run_id}
        histories.prepare(ids)
    summaries, items = [], []
    log(f"[INPUT] completed={len(raw):,}; fixed cases={len(cases)}; output={figures_dir}")
    for i, ((study, case_id), group) in enumerate(cases, 1):
        anchor = "default" if study.startswith("sensitivity_") else "FULL"
        pair_rows, pairs = paired(group, anchor)
        r = group.iloc[0]
        folder = f"cases/{study}/{r.problem}_M{int(r.m)}_D{int(r.d)}_{case_id}"
        saved = []
        for suffix, renderer in (("01_quality_outcome", lambda path: quality_and_outcomes(group, pair_rows, pairs, path)),
                                 ("02_convergence", lambda path: convergence(group, pair_rows, histories, path)),
                                 ("03_quality_cost", lambda path: cost_effect(group, pair_rows, path)),
                                 ("04_migration_diagnostic", lambda path: migration_diagnostic(group, pair_rows, path))):
            namepath = f"{folder}/{suffix}"
            try:
                if renderer(figures_dir / namepath):
                    saved.append(namepath)
            except Exception as error:
                log(f"[FIGURE ERROR] {r.problem}/{study}/{suffix}: {type(error).__name__}: {error}")
                raise
        if len(pairs):
            for entry in pairs.to_dict("records"):
                summaries.append(dict(study=study, problem=r.problem, m=int(r.m), d=int(r.d),
                                      budget=int(r.budget), case_key=case_id, control=anchor,
                                      comparison=entry.pop("level"), **entry))
        current = pairs[pairs.level == "NO_MIG"] if len(pairs) else pd.DataFrame()
        if current.empty and len(pairs):
            current = pairs.iloc[:1]
        note = (f"{anchor} vs {current.iloc[0].level}: both metrics better in {int(current.iloc[0].both)}/{int(current.iloc[0].n_paired)} paired seeds. "
                if len(current) else "No matched seeds are available. ")
        note += "Descriptive until 30 paired runs; no cross-problem or cross-dimension pooling."
        items.append(dict(title=title(group), study=study, problem=r.problem, note=note, figures=saved))
        log(f"[{i}/{len(cases)}] {study} {r.problem} M={int(r.m)} D={int(r.d)}: paired={len(pair_rows)}; figures={len(saved)}")
    pd.DataFrame(summaries).to_csv(stats_dir / "case_pairs.csv", index=False)
    gallery(figures_dir, items, manifest)
    lines = ["# TherMOO fixed-case atlas", "", f"Source: `{source.get('source_snapshot', ', '.join(source.get('source_roots', [])))}`.",
             f"Cohort fingerprint: `{source['fingerprint']}`. Completed original runs: {len(raw):,}; cases: {len(items)}.",
             "", "Each card compares methods within an identical function, objective count M, decision dimension D, FE budget and experimental setting.",
             "Ablations and parameter changes use the main FULL runs at exactly the same settings as controls.",
             "Only identical seed numbers with the same initial_hash enter paired comparisons.",
             "Joint win requires FULL to have higher HV **and** lower IGD+ on the same seed.",
             "The intermediate labels HV only and IGD+ only include trade-offs and ties on the other metric.",
             "Whiskers are descriptive bootstrap intervals of paired medians, not Holm-adjusted significance tests.",
             "Observed wall time is affected by concurrent server load. Each history point is an actual recorded FE, never interpolated.",
             "The optional migration panel uses only matched FULL/NO_MIG seeds with variation in accepted counts; its correlation is not causal.",
             "ZDT5 uses the actual binary algorithm label when the GDE3 slot runs SPEA2.",
             "FULL here is the original heuristic migration policy, not a trained neural controller.",
             "", f"Open `{figures_dir / 'index.html'}` for the gallery.",
             f"Paired numerical table: `{stats_dir / 'case_pairs.csv'}`.", ""]
    (stats_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    manifest.update(status="COMPLETE", completed_runs=len(raw), fixed_cases=len(items),
                    figures=sum(len(x["figures"]) for x in items), finished_utc=datetime.now(timezone.utc).isoformat())
    (stats_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"[DONE] {stats_dir.name}: cases={len(items)}, figures={manifest['figures']}; gallery={figures_dir / 'index.html'}")


if __name__ == "__main__":
    main()

