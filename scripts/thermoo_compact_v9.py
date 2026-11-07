#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""TherMOO compact scientific renderer V9: at most 18 figures, original experiment."""
import time
START_NS = time.time_ns()
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import sys
import warnings
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

"""The complete 33,240-trial matrix, without a Cartesian parameter sweep."""
from dataclasses import asdict, dataclass, replace
import hashlib
import json

FAMILIES = {
    "ZDT": [f"ZDT{i}" for i in range(1, 7)],
    "DTLZ": [f"DTLZ{i}" for i in range(1, 8)],
    "WFG": [f"WFG{i}" for i in range(1, 10)],
    "LZ09": [f"LZ09_F{i}" for i in range(1, 10)],
    "UF": [f"UF{i}" for i in range(1, 11)],
    "ZCAT": [f"ZCAT{i}" for i in range(1, 21)],
}
PROBLEMS = sum(FAMILIES.values(), [])
PANEL = ["ZDT1", "ZDT3", "DTLZ2", "DTLZ7", "WFG1", "WFG9", "LZ09_F1", "UF8", "ZCAT14", "ZCAT20"]
METHODS = ["NSGA2", "MOEAD", "GDE3", "SMS", "NO_MIG", "RANDOM", "ELITE", "HV_GATE", "CROWDING", "FULL"]
ABLATIONS = ["conventional", "no_entropy", "no_free_energy", "no_js", "fixed_rate", "random_source", "boltzmann_only"]
BLOCKS = ["main", "ablation", "sensitivity", "homogeneous", "islands", "many", "dimensions"]


def family(name):
    return next(f for f, names in FAMILIES.items() if name in names)


@dataclass(frozen=True)
class Job:
    problem: str
    method: str
    seed: int
    block: str = "main"
    m: int = 0
    d: int = 0
    budget: int = 100000
    population: int = 400
    islands: int = 4
    composition: str = "mixed"
    interval: int = 10
    rho: float = 0.10
    temperature: float = 1.0
    quality: str = "rank_crowding"
    ablation: str = "none"
    archive: int = 400
    hv_epsilon: float = 1e-4
    js_epsilon: float = 0.02

    @property
    def objectives(self):
        if self.m:
            return self.m
        if self.problem.startswith("ZDT"):
            return 2
        if self.problem.startswith("LZ09"):
            return 3 if self.problem == "LZ09_F6" else 2
        if self.problem.startswith("UF"):
            return 2 if int(self.problem[2:]) <= 7 else 3
        return 3

    @property
    def variables(self):
        if self.d:
            return self.d
        if self.problem == "ZDT5":
            return 80
        if self.problem in ("ZDT4", "ZDT6"):
            return 10
        if self.problem.startswith("DTLZ"):
            k = 5 if self.problem == "DTLZ1" else 20 if self.problem == "DTLZ7" else 10
            return self.objectives + k - 1
        if self.problem.startswith("WFG"):
            return 2 * (self.objectives - 1) + 20
        if self.problem.startswith("ZCAT"):
            return 10 * self.objectives
        return 30

    @property
    def label(self):
        if self.ablation != "none":
            return self.ablation
        return self.method

    def signature(self):
        data = asdict(self)
        data.pop("block")
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:20]


def jobs(blocks=None, seeds=range(30)):
    blocks = set(BLOCKS if blocks is None else blocks)
    for seed in seeds:
        if "main" in blocks:
            for problem in PROBLEMS:
                for method in METHODS:
                    yield Job(problem, method, seed)
        if "ablation" in blocks:
            for problem in PANEL:
                for ablation in ABLATIONS:
                    yield Job(problem, "FULL", seed, block="ablation", ablation=ablation)
        if "sensitivity" in blocks:
            changes = [{"interval": x} for x in (5, 20)] + [{"rho": x} for x in (0.05, 0.20)]
            changes += [{"temperature": x} for x in (0.5, 2.0)]
            changes += [{"quality": x} for x in ("rank_hv", "rank_novelty")]
            for problem in PANEL:
                for change in changes:
                    yield replace(Job(problem, "FULL", seed, block="sensitivity"), **change)
        if "homogeneous" in blocks:
            for problem in PANEL:
                for composition in ("NSGA2", "MOEAD", "GDE3", "SMS"):
                    for method in ("NO_MIG", "ELITE", "FULL"):
                        yield Job(problem, method, seed, block="homogeneous", composition=composition)
        if "islands" in blocks:
            for problem in PANEL:
                for k in (8, 16):
                    for method in ("NO_MIG", "ELITE", "FULL"):
                        yield Job(problem, method, seed, block="islands", islands=k)
        if "many" in blocks:
            for problem in ("DTLZ2", "DTLZ7", "WFG4", "WFG9", "ZCAT14", "ZCAT20"):
                for method in METHODS + ["NSGA3", "RVEA"]:
                    yield Job(problem, method, seed, block="many", m=5)
        if "dimensions" in blocks:
            for problem in ("ZDT4", "DTLZ2", "DTLZ7", "WFG9", "UF1", "ZCAT20"):
                for d in (50, 100):
                    for method in ["NSGA2", "MOEAD", "GDE3", "SMS", "NO_MIG", "ELITE", "HV_GATE", "FULL"]:
                        yield Job(problem, method, seed, block="dimensions", d=d)


VERSION="9.0.0"
METRICS=("hv","igd_plus")
METRIC_NAMES={"hv":"HV ↑","igd_plus":"IGD+ ↓"}
POLICIES=("NO_MIG","ELITE","FULL")
DIM_PROBLEMS=("ZDT4","DTLZ2","DTLZ7","WFG9","UF1","ZCAT20")
DIM_METHODS=("NSGA2","MOEAD","GDE3","SMS","NO_MIG","ELITE","HV_GATE","FULL")
SINGLETONS={"NSGA2","MOEAD","GDE3","SMS","NSGA3","RVEA"}
CORE=("NSGA2","MOEAD","GDE3","NO_MIG","RANDOM","ELITE","HV_GATE","CROWDING","FULL")
CURVE_METHODS=("NSGA2","MOEAD","NO_MIG","RANDOM","ELITE","FULL")
EXAMPLES=("ZDT3","DTLZ7","WFG1","LZ09_F1","UF8","ZCAT20")
EXAMPLE_POOL=EXAMPLES+("ZDT1","DTLZ2","WFG9","UF1","ZCAT14")
MANY_PROBLEMS=("DTLZ2","DTLZ7","WFG4","WFG9","ZCAT14","ZCAT20")
BG,AX_BG,GRID="#FAF8F2","#F4F1E8","#D8D3C5"
C=plt.get_cmap("viridis")
COLORS={m:C(x) for m,x in zip((*METHODS,"NSGA3","RVEA"),np.linspace(.06,.94,12))}
COLORS.update(FULL=C(.84),NO_MIG=C(.16),ELITE=C(.59),RANDOM=C(.38))
LABELS={"NSGA2":"NSGA-II","MOEAD":"MOEA/D","SMS":"SMS-EMOA","NSGA3":"NSGA-III","NO_MIG":"No migration","HV_GATE":"HV gate","CROWDING":"Crowding","RANDOM":"Random","ELITE":"Elite","FULL":"FULL", "conventional":"Conventional state","no_entropy":"Without entropy","no_free_energy":"Without free energy","no_js":"Without JS drift","fixed_rate":"Fixed rate","random_source":"Random source","boltzmann_only":"Boltzmann only"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":20,"axes.titlesize":24,"axes.labelsize":21,"xtick.labelsize":18,"ytick.labelsize":18,"legend.fontsize":17,"figure.titlesize":27,"axes.linewidth":1.1,"lines.linewidth":2.5,"axes.edgecolor":"black","text.color":"black","axes.labelcolor":"black","xtick.color":"black","ytick.color":"black","axes.labelweight":"bold","axes.titleweight":"bold","pdf.fonttype":42,"ps.fonttype":42})
def log(message):print(message,flush=True)

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


def collect_live(roots, fingerprint=None):
    plan = {j.signature(): j for j in jobs()}
    accepted, conflicts, issues = {}, set(), []
    checked = 0
    for root in roots:
        for path in sorted((root/'outputs/full').glob('*/*/result.json')):
            checked += 1
            if checked % 3000 == 0:
                log(f'[READ] checked={checked:,}; accepted={len(accepted):,}')
            fp, rid = path.parents[1].name, path.parent.name
            if fingerprint and fp != fingerprint:
                continue
            try:
                blob, stamp = stable_read(path, START_NS)
                data = json.loads(blob);job = plan.get(rid)
                if job is None or data.get('job') != asdict(job):
                    raise ValueError('not_in_original_plan')
                if data.get('status') != 'complete' or data.get('fingerprint') != fp or data.get('run_id') != rid:
                    raise ValueError('commit_or_identity_mismatch')
                if data.get('evaluations') != job.budget or data.get('metrics',{}).get('fe') != job.budget:
                    raise ValueError('incomplete_FE_budget')
                metrics = {m:float(data['metrics'][m]) for m in METRICS}
                if any(not math.isfinite(v) or v < 0 for v in metrics.values()):
                    raise ValueError('invalid_metric')
                if not re.fullmatch('[0-9a-f]{64}',str(data.get('initial_hash',''))):
                    raise ValueError('missing_initial_hash')
                expected_k=1 if job.method in SINGLETONS else job.islands
                if len(data.get('algorithms',[]))!=expected_k:
                    raise ValueError('actual_algorithm_count_mismatch')
                pop=(path.parent/'population.npz').stat()
                if not pop.st_size or max(pop.st_ctime_ns,pop.st_mtime_ns)>START_NS:
                    raise ValueError('uncommitted_population')
                row=normalize_job(job,family)
                row.update(metrics, fingerprint=fp, initial_hash=data['initial_hash'], source_result=str(path),
                           result_sha256=stamp['sha256'], actual_algorithms='|'.join(data.get('algorithms',[])))
                for field in ('seconds','sent','accepted','payload_bytes','control_seconds','integration_seconds','diagnostic_seconds'):
                    row[field]=float(data.get(field,0))
                if row['seconds']<=0 or not math.isfinite(row['seconds']) or row['accepted']>row['sent']:
                    raise ValueError('invalid_runtime_or_counters')
                row['peak_rss_gib']=np.nan
                for kind in ('dynamic','scheduler'):
                    side=root/'reports'/kind/fp/f'{rid}.json'
                    if side.is_file():
                        try:
                            payload=json.loads(stable_read(side,START_NS)[0])
                            if payload.get('status') in ('complete','cached'):
                                row['peak_rss_gib']=float(payload.get('peak_rss_bytes',np.nan))/2**30;break
                        except (OSError,ValueError,TypeError):pass
                identity=tuple(row[k] for k in ('hv','igd_plus','initial_hash','actual_algorithms','sent','accepted'))
                key=(fp,rid)
                if key in conflicts:continue
                if key in accepted and accepted[key][1] != identity:
                    accepted.pop(key);conflicts.add(key)
                    raise ValueError('conflicting_duplicate_excluded')
                if key not in accepted:accepted[key]=(row,identity)
            except (OSError,ValueError,KeyError,TypeError,CutoffError) as error:
                issues.append(dict(path=str(path),reason=str(error)))
    counts=Counter(k[0] for k in accepted)
    if not counts:
        raise RuntimeError('No completed original trials found. Use --snapshot /path/to/STATS/vN if results are archived.')
    fp=fingerprint or counts.most_common(1)[0][0]
    raw=pd.DataFrame(v[0] for k,v in accepted.items() if k[0]==fp)
    return raw,dict(mode='live',fingerprint=fp,source_roots=list(map(str,roots)),cohorts=dict(counts),cutoff_ns=START_NS),issues


def collect_snapshot(folder):
    info=json.loads((folder/'manifest.json').read_text())
    raw=pd.read_csv(folder/'per_run.csv',low_memory=False)
    if raw.empty or not info.get('fingerprint'):
        raise ValueError('Snapshot has no completed original cohort')
    plan={j.signature():j for j in jobs()};keep=[]
    for r in raw.to_dict('records'):
        job=plan.get(str(r.get('run_id','')))
        if job is None:continue
        if int(r.get('budget',-1))!=job.budget or int(r.get('m',-1))!=job.objectives or int(r.get('d',-1))!=job.variables:continue
        r.update(normalize_job(job,family));keep.append(r)
    raw=pd.DataFrame(keep)
    if raw.empty:raise ValueError('Snapshot does not match the original protocol')
    if raw.duplicated('run_id').any():raise ValueError('Snapshot contains duplicate run IDs')
    return raw,dict(mode='snapshot',fingerprint=info['fingerprint'],source_snapshot=str(folder.resolve()),
                    cutoff_ns=info.get('cutoff_ns',START_NS)),[]


def observed_views(raw,planned):
    fields=['run_id','hv','igd_plus','initial_hash','seconds','sent','accepted','control_seconds',
            'integration_seconds','diagnostic_seconds','payload_bytes','peak_rss_gib']
    data=raw.copy()
    for col in fields:
        if col not in data:data[col]=np.nan if col=='peak_rss_gib' else 0
    result=planned.merge(data[fields],on='run_id',how='inner',validate='many_to_one')
    return result[result.problem!='ZDT5'].copy()


def paired_analysis(planned,obs):
    details,rows=[],[]
    actual={k:g for k,g in obs.groupby(['study','facet','case'],sort=False)}
    expected=planned[planned.problem!='ZDT5'];hypotheses=Counter()
    for _,g in expected.groupby(['study','facet','case'],sort=False):
        hypotheses[g.comparison_family.iloc[0]] += (g.level.nunique()-1)*len(METRICS)
    for key,g in actual.items():
        first=g.iloc[0];control=str(first.control)
        a=g[g.level==control].set_index('seed')
        if a.empty:continue
        for level,other in g[g.level!=control].groupby('level',sort=False):
            b=other.set_index('seed');seeds=a.index.intersection(b.index)
            seeds=[s for s in seeds if a.loc[s,'initial_hash']==b.loc[s,'initial_hash']]
            if not seeds:continue
            inferred=set(seeds)==set(range(30))
            for metric in METRICS:
                x=a.loc[seeds,metric].to_numpy(float);y=b.loc[seeds,metric].to_numpy(float)
                delta=(x-y) if metric=='hv' else (y-x)
                tolerance=1e-12*np.maximum(1,np.maximum(np.abs(x),np.abs(y)))
                points=np.where(delta>tolerance,1.,np.where(delta < -tolerance,0.,.5))
                p=normal=np.nan;test='descriptive'
                if inferred:
                    if np.all(np.abs(delta)<=tolerance):p=1.;test='constant_tie'
                    elif np.ptp(delta)<=max(1e-300,np.max(np.abs(delta))*1e-13):
                        p=2.0**(1-len(delta));test='constant_sign_exact'
                    else:
                        normal=float(stats.shapiro(delta).pvalue)
                        if normal>.05:p=float(stats.ttest_1samp(delta,0).pvalue);test='paired_t'
                        else:
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore');p=float(stats.wilcoxon(delta,zero_method='pratt',method='auto').pvalue)
                            test='wilcoxon_pratt'
                rec=dict(study=key[0],facet=key[1],case=key[2],problem=first.problem,family=first.family,
                         m=int(first.m),d=int(first.d),control=control,level=str(level),metric=metric,
                         comparison_family=first.comparison_family,n=len(seeds),probability=float(points.mean()),
                         mean_delta=float(delta.mean()),median_delta=float(np.median(delta)),p_raw=p,
                         normality_p=normal,test=test,eligible=inferred)
                rows.append(rec)
                for s,dv,win in zip(seeds,delta,points):
                    details.append(dict(study=key[0],facet=key[1],case=key[2],problem=first.problem,family=first.family,
                                        level=str(level),metric=metric,seed=int(s),win=float(win),delta=float(dv),
                                        time_ratio=float(a.loc[s,'seconds'])/float(b.loc[s,'seconds'])))
    result=pd.DataFrame(rows)
    if result.empty:return result,pd.DataFrame(details)
    result['p_holm']=np.nan;result['decision']='descriptive'
    for fam,g in result.groupby('comparison_family',sort=False):
        idx=g.index[g.eligible]
        if len(idx):
            result.loc[idx,'p_holm']=holm(result.loc[idx,'p_raw'],hypotheses[fam])
            for i in idx:
                r=result.loc[i]
                result.loc[i,'decision']=('win' if r.mean_delta>0 else 'loss') if r.p_holm<.05 else 'not_detected'
    return result,pd.DataFrame(details)


def probability_summary(details,bootstrap=600):
    groups=[g.win.to_numpy(float) for _,g in details.groupby('case',sort=False)]
    if not groups:return np.nan,np.nan,np.nan,0
    rng=np.random.default_rng(stable_seed('superiority',tuple(details.level.unique()),tuple(details.metric.unique())))
    inside=np.column_stack([a[rng.integers(0,len(a),(bootstrap,len(a)))].mean(axis=1) for a in groups])
    picks=rng.integers(0,len(groups),(bootstrap,len(groups)))
    sample=inside[np.arange(bootstrap)[:,None],picks].mean(axis=1)
    lo,hi=np.quantile(sample,[.025,.975])
    return float(np.mean([a.mean() for a in groups])),float(lo),float(hi),len(groups)


def balanced_main(obs,min_seeds=3):
    selected=[]
    for _,g in obs[(obs.study=='main') & obs.level.isin(CORE)].groupby('case',sort=False):
        if set(g.level)!=set(CORE):continue
        seeds=set.intersection(*[set(g[g.level==m].seed) for m in CORE])
        seeds=[s for s in seeds if g[g.seed==s].initial_hash.nunique()==1]
        if len(seeds)>=min_seeds:selected.append(g[g.seed.isin(seeds)].copy())
    if not selected:return pd.DataFrame(),False
    strict=[g for g in selected if set(g.seed)==set(range(30))];infer=len(strict)>=3
    return pd.concat(strict if infer else selected,ignore_index=True),infer


def balanced_scores(main):
    rows=[]
    if main.empty:return pd.DataFrame()
    for case,g in main.groupby('case',sort=False):
        first=g.iloc[0]
        for metric in METRICS:
            med=g.groupby('level')[metric].median().reindex(CORE)
            ranking=stats.rankdata(-med if metric=='hv' else med,method='average')
            best=float(med.max() if metric=='hv' else med.min())
            for method,rank,value in zip(CORE,ranking,med):
                ratio=(best/value if value>0 else (1. if best==0 else np.inf)) if metric=='hv' else (value/best if best>0 else (1. if value==0 else np.inf))
                rows.append(dict(case=case,problem=first.problem,family=first.family,metric=metric,method=method,
                                 rank=float(rank),median=float(value),ratio=float(ratio),n=int(g[g.level==method].seed.nunique())))
    return pd.DataFrame(rows)


def selected_histories(raw,source,sd):
    rows=raw[(raw.block=='main') & raw.problem.isin(EXAMPLE_POOL) & raw.method.isin(CURVE_METHODS)].copy()
    wanted=set(rows.run_id.astype(str));histories={}
    path=Path(source.get('source_snapshot',''))/'snapshot/histories.jsonl.gz'
    reduced=Path(source.get('source_snapshot',''))/'snapshot/selected_histories.jsonl.gz'
    def add(rid,values):
        histories[rid]={int(v['fe']):{m:float(v[m]) for m in METRICS} for v in values
                         if int(v.get('fe',-1)) in (400,10000,25200,50000,100000)
                         and all(m in v and np.isfinite(float(v[m])) for m in METRICS)}
    if source['mode']=='snapshot' and reduced.is_file():
        with gzip.open(reduced,'rt',encoding='utf-8') as stream:
            for line in stream:
                item=json.loads(line);rid=str(item['run_id'])
                if rid in wanted:histories[rid]={int(fe):v for fe,v in item['points'].items()}
    elif source['mode']=='snapshot' and path.is_file():
        with gzip.open(path,'rt',encoding='utf-8') as stream:
            for line in stream:
                item=json.loads(line);rid=str(item['run_id'])
                if rid in wanted:
                    blob=item['data_json'];digest=item.get('meta',{}).get('sha256')
                    if digest and hashlib.sha256(blob.encode()).hexdigest()!=digest:continue
                    add(rid,json.loads(blob))
    else:
        for r in rows.itertuples():
            try:
                result=Path(r.source_result);data=json.loads(stable_read(result,int(source['cutoff_ns']))[0])
                if data.get('initial_hash')!=r.initial_hash or data.get('run_id')!=r.run_id:continue
                if any(not np.isclose(float(data['metrics'][m]),float(getattr(r,m)),rtol=1e-10,atol=1e-14) for m in METRICS):continue
                add(str(r.run_id),json.loads(stable_read(result.parent/'history.json',int(source['cutoff_ns']))[0]))
            except (AttributeError,OSError,ValueError,KeyError,TypeError,CutoffError):pass
    (sd/'snapshot').mkdir(exist_ok=True)
    with gzip.open(sd/'snapshot/selected_histories.jsonl.gz','wt',encoding='utf-8') as stream:
        for rid,data in histories.items():stream.write(json.dumps(dict(run_id=rid,points=data))+'\n')
    return histories


def nm(value):return LABELS.get(str(value),str(value))


def style_ax(ax,title=None,xlabel=None,ylabel=None):
    ax.set_facecolor(AX_BG);ax.grid(True,color=GRID,linewidth=.8,alpha=.7,zorder=0);ax.set_axisbelow(True)
    for spine in ax.spines.values():spine.set_color('black');spine.set_linewidth(1.1)
    if title:ax.set_title(title,pad=14,weight='bold')
    if xlabel:ax.set_xlabel(xlabel,labelpad=10)
    if ylabel:ax.set_ylabel(ylabel,labelpad=10)


def box(ax,x,y,text,fontsize=18,**kwargs):
    return ax.text(x,y,text,fontsize=fontsize,weight='bold',ha=kwargs.pop('ha','center'),
                   va=kwargs.pop('va','center'),bbox=dict(boxstyle='round,pad=.2',facecolor='white',
                   edgecolor='black',linewidth=.8,alpha=.96),zorder=10,**kwargs)


def gradient(ax,y,start,end,height=.55):
    if start==end:return
    lo,hi=sorted((start,end))
    ax.imshow(np.linspace(.15,.9,256)[None,:],extent=(lo,hi,y-height/2,y+height/2),
              cmap='viridis',vmin=0,vmax=1,aspect='auto',alpha=.8,zorder=2)
    ax.add_patch(Rectangle((lo,y-height/2),hi-lo,height,facecolor='none',edgecolor='black',lw=.9,zorder=3))


def heat(ax,matrix,rows,cols,title,labels=None,vmin=0,vmax=1):
    a=np.asarray(matrix,float);style_ax(ax,title)
    ax.grid(False);masked=np.ma.masked_invalid(a)
    ax.imshow(masked,cmap='viridis',vmin=vmin,vmax=vmax,aspect='auto',alpha=.83)
    ax.set_xticks(range(len(cols)),[nm(v) for v in cols],rotation=35,ha='right')
    ax.set_yticks(range(len(rows)),rows)
    ax.set_xticks(np.arange(-.5,len(cols),1),minor=True);ax.set_yticks(np.arange(-.5,len(rows),1),minor=True)
    ax.grid(which='minor',color='black',lw=.65);ax.tick_params(which='minor',bottom=False,left=False)
    for i in range(len(rows)):
        for j in range(len(cols)):
            value=a[i,j]
            text=(labels[i][j] if labels is not None else f'{100*value:.0f}%') if np.isfinite(value) else '—'
            box(ax,j,i,text,fontsize=17)


class CompactFigures:
    def __init__(self,raw,obs,pairs,detail,scores,rank_infer,histories,sd,fd,dpi=320):
        self.raw,self.obs,self.pairs,self.detail=raw,obs,pairs,detail
        self.scores,self.rank_infer,self.histories=scores,rank_infer,histories
        self.sd,self.fd,self.dpi=sd,fd,dpi;self.entries=[]
        (sd/'tables').mkdir(exist_ok=True)

    def subplots(self,title,rows=1,cols=2,width=21,height=9):
        fig,axes=plt.subplots(rows,cols,figsize=(width,height),facecolor=BG,squeeze=False,layout='constrained')
        fig.suptitle(title,weight='bold',fontsize=27)
        for ax in axes.flat:style_ax(ax)
        return fig,axes

    def emit(self,fig,number,slug,caption,data=None):
        stem=f'{number:02d}_{slug}'
        for suffix in ('png','pdf'):
            fig.savefig(self.fd/f'{stem}.{suffix}',dpi=self.dpi,bbox_inches='tight',facecolor=BG,pad_inches=.2)
        plt.close(fig)
        if data is not None:write_csv(self.sd/'tables'/f'{stem}.csv',data)
        self.entries.append(dict(number=number,stem=stem,status='created',caption=caption))
        log(f'[FIGURE {number:02d}/18] {stem}')

    def skip(self,number,slug,reason):
        self.entries.append(dict(number=number,stem=f'{number:02d}_{slug}',status='unavailable',caption=reason))
        log(f'[FIGURE {number:02d}/18] skipped: {reason}')

    def ranks(self):
        data=self.scores
        if data.empty:return self.skip(1,'ranks','No common seed panel for the nine predeclared core methods.')
        fig,axes=self.subplots('Overall comparison · within-problem ranks',height=10)
        tables=[]
        for ax,metric in zip(axes.flat,METRICS):
            sub=data[data.metric==metric];matrix=sub.pivot(index='case',columns='method',values='rank').reindex(columns=CORE)
            avg=matrix.mean().sort_values();k=len(avg);n=len(matrix)
            ax.clear();ax.set_facecolor(AX_BG);ax.set_xlim(-1.7,k+2.6);ax.set_ylim(6.5,-.65);ax.axis('off')
            ax.set_title(METRIC_NAMES[metric],fontsize=25,pad=20)
            ax.plot([1,k],[0,0],color='black',lw=1.5)
            for tick in range(1,k+1):
                ax.plot([tick,tick],[-.08,.08],color='black',lw=1);ax.text(tick,-.24,str(tick),ha='center',fontsize=18)
            cd=np.nan;omnibus=np.nan
            if self.rank_infer and n>=3:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore');omnibus=float(stats.friedmanchisquare(*[matrix[m] for m in CORE]).pvalue)
                cd=float(stats.studentized_range.ppf(.975,k,np.inf)/np.sqrt(2)*np.sqrt(k*(k+1)/(6*n)))
                ax.plot([1,1+cd],[-.5,-.5],color='black',lw=2.7)
                ax.text(1+cd/2,-.59,f'CD={cd:.2f}',fontsize=17,ha='center',va='bottom')
                if np.isfinite(omnibus) and omnibus<.025:
                    values=avg.to_numpy();groups=[]
                    for i in range(k):
                        j=max(j for j in range(i,k) if values[j]-values[i]<=cd)
                        if j>i and not any(i>=a and j<=b for a,b in groups):groups.append((i,j))
                    groups=[g for g in groups if not any(g!=h and g[0]>=h[0] and g[1]<=h[1] for h in groups)]
                    for j,(a,b) in enumerate(groups):ax.plot([values[a],values[b]],[.30+.16*j]*2,color='black',lw=4.2)
            split=(k+1)//2
            for i,(method,value) in enumerate(avg.items()):
                left=i<split;y=2.1+.73*(i if left else i-split)
                edge=.55 if left else k+.45;textx=.4 if left else k+.6
                ax.plot([value,value,edge],[.12,y,y],color=COLORS[method],lw=1.7)
                ax.scatter(value,.12,color=COLORS[method],s=105,edgecolor='black',zorder=5)
                ax.text(textx,y,nm(method),ha='right' if left else 'left',va='center',fontsize=19,
                        weight='bold' if method=='FULL' else 'normal',bbox=dict(facecolor=BG,edgecolor='none',pad=1.5))
                tables.append(dict(metric=metric,method=method,average_rank=float(value),problems=n,
                                   CD=cd,friedman_p=omnibus,inferential=self.rank_infer))
            box(ax,(k+1)/2,5.95,f'{n} matched problems · lower rank is better',fontsize=18)
        self.emit(fig,1,'ranks','Each problem contributes one seed-median rank. The fixed core excludes SMS-EMOA, which is included in all available pairwise comparisons and many-objective matrices. ZDT5 is kept in the raw table as a separate binary cohort. CD uses Nemenyi, alpha=.025 per metric, and is drawn only after the Friedman gate and 30 matched seeds per problem. Bars mean no detected difference, not equivalence.',tables)

    def family_map(self):
        d=self.pairs[self.pairs.study=='main']
        if d.empty:return self.skip(2,'families','No paired main comparisons.')
        fig,axes=self.subplots('Where FULL improves · benchmark families',width=24,height=10)
        cols=[m for m in METHODS if m!='FULL'];families=list(FAMILIES)
        out=[]
        for ax,metric in zip(axes.flat,METRICS):
            matrix=np.full((len(families),len(cols)),np.nan)
            for i,f in enumerate(families):
                for j,level in enumerate(cols):
                    sub=d[(d.family==f)&(d.level==level)&(d.metric==metric)]
                    if len(sub):
                        matrix[i,j]=sub.probability.mean();out.append(dict(family=f,method=level,metric=metric,probability=matrix[i,j],problems=len(sub),min_pairs=int(sub.n.min())))
            heat(ax,matrix,families,cols,METRIC_NAMES[metric])
        self.emit(fig,2,'families','Cells are P(FULL better)+0.5 P(tie), averaged after within-problem seed pairing. Each problem has equal weight within its family. 50% is neutral. Available pair counts can differ; exact n and problem counts are in the figure CSV. No family-level significance claim is made.',out)

    def significance(self):
        d=self.pairs[(self.pairs.study=='main')&self.pairs.eligible]
        if d.empty:return self.skip(3,'corrected_comparisons','No complete 30-seed comparisons; partial data are shown in descriptive figures.')
        fig,axes=self.subplots('Statistical evidence · FULL versus alternatives',height=11)
        cols=[m for m in METHODS if m!='FULL'];out=[]
        for ax,metric in zip(axes.flat,METRICS):
            for y,level in enumerate(cols):
                sub=d[(d.metric==metric)&(d.level==level)];left=0
                for decision,c in (('win',C(.75)),('not_detected','#D8D3C5'),('loss',C(.12))):
                    count=int((sub.decision==decision).sum())
                    ax.barh(y,count,left=left,color=c,height=.65,ec='black',lw=.7)
                    if count:box(ax,left+count/2,y,str(count),fontsize=17)
                    left+=count
                if not len(sub):ax.text(0,y,'<30 matched seeds',fontsize=17,va='center')
                out.append(dict(metric=metric,method=level,problems=len(sub),wins=int((sub.decision=='win').sum()),losses=int((sub.decision=='loss').sum()),not_detected=int((sub.decision=='not_detected').sum())))
            ax.set_yticks(range(len(cols)),[nm(m) for m in cols]);ax.invert_yaxis()
            ax.set_title(METRIC_NAMES[metric]);ax.set_xlabel('Number of benchmark problems')
        fig.legend(handles=[Patch(fc=c,ec='black',label=n) for n,c in [('FULL better',C(.75)),('Not detected','#D8D3C5'),('FULL worse',C(.12))]],loc='outside lower center',ncol=3)
        self.emit(fig,3,'corrected_comparisons','Only the exact seeds 0–29 with equal initial hashes enter inference. Shapiro is applied to paired differences; paired t or Wilcoxon-Pratt is used accordingly. Holm covers both metrics and all planned hypotheses of the block, including unfinished comparisons. Not detected is not evidence of equality.',out)

    def profiles(self):
        if self.scores.empty:return self.skip(4,'profiles','No balanced core-method panel.')
        fig,axes=self.subplots('Performance profiles · proximity to the best observed method',height=9)
        points=[];grid=np.geomspace(1,10,150)
        for ax,metric in zip(axes.flat,METRICS):
            for method in CORE:
                values=self.scores[(self.scores.metric==metric)&(self.scores.method==method)].ratio.to_numpy(float)
                y=np.array([(values<=tau+1e-12).mean() for tau in grid])
                ax.plot(grid,y,color=COLORS[method],lw=3.5 if method=='FULL' else 2,label=nm(method))
                points.extend(dict(metric=metric,method=method,tau=float(tau),fraction=float(v)) for tau,v in zip(grid,y))
            ax.set_xscale('log');ax.set_xlim(1,10);ax.set_ylim(0,1.04)
            ax.set_xticks([1,1.25,2,5,10],['1','1.25','2','5','10'])
            ax.set_title(METRIC_NAMES[metric]);ax.set_xlabel('Allowed factor from best observed value, τ')
            ax.set_ylabel('Fraction of benchmark problems')
        h,l=axes.flat[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=5)
        self.emit(fig,4,'profiles','Same balanced core problem and seed panel as figure 01. Ratios are best/HV and IGD+/best, computed within each problem from seed medians. 0/0 is a tie (ratio 1); a nonzero loss against a zero best value is infinite. The reference is the best observed competitor, not a certified optimum.',points)

    def effects(self):
        data=self.detail[self.detail.study=='main']
        if data.empty:return self.skip(5,'practical_effect','No matched main seeds.')
        levels=[m for m in METHODS if m!='FULL'];fig,axes=self.subplots('How consistently FULL wins · paired practical effects',height=11)
        out=[]
        for ax,metric in zip(axes.flat,METRICS):
            ax.axvline(.5,color='black',ls='--',lw=1.4)
            for y,m in enumerate(levels):
                g=data[(data.metric==metric)&(data.level==m)]
                if g.empty:continue
                value,lo,hi,n=probability_summary(g)
                gradient(ax,y,.5,value);ax.plot([lo,hi],[y,y],color='black',lw=2.2)
                ax.scatter(value,y,s=125,c=[COLORS[m]],ec='black',zorder=5)
                box(ax,1.09,y,f'{100*value:.0f}%',fontsize=18)
                out.append(dict(metric=metric,method=m,value=value,ci_low=lo,ci_high=hi,problems=n,min_pairs=int(g.groupby('case').seed.nunique().min())))
            ax.set_xlim(-.02,1.17);ax.set_yticks(range(len(levels)),[nm(m) for m in levels]);ax.invert_yaxis()
            ax.set_title(METRIC_NAMES[metric]);ax.set_xticks([0,.25,.5,.75,1],['0','25','50','75','100%'])
            ax.set_xlabel('Probability FULL wins · ties count as ½')
        self.emit(fig,5,'practical_effect','Pairing is within an identical function, M, D, budget and setting. Problem-level win probabilities receive equal weight. Whiskers resample problems and their paired seeds; these descriptive intervals are not Holm tests. The exact panel for every alternative is exported.',out)

    def joint(self):
        data=self.detail[self.detail.study=='main']
        if data.empty:return self.skip(6,'joint_metrics','No paired two-metric outcomes.')
        pivot=data.pivot(index=['case','problem','level','seed'],columns='metric',values='win').reset_index()
        outcomes=[]
        for (case,level),g in pivot.groupby(['case','level']):
            both=(g.hv==1)&(g.igd_plus==1);loss=(g.hv==0)&(g.igd_plus==0);tie=(g.hv==.5)&(g.igd_plus==.5)
            outcomes.append(dict(case=case,level=level,both=both.mean(),loss=loss.mean(),tie=tie.mean(),mixed=(~(both|loss|tie)).mean(),n=len(g)))
        table=pd.DataFrame(outcomes);levels=[m for m in METHODS if m!='FULL']
        fig,axes=self.subplots('Agreement of HV and IGD+ · outcomes on the same seed',cols=1,width=15,height=11);ax=axes[0,0]
        for y,level in enumerate(levels):
            g=table[table.level==level]
            if g.empty:continue
            start=0
            for key,c in [('both',C(.8)),('mixed',C(.48)),('tie','#D8D3C5'),('loss',C(.1))]:
                value=float(g[key].mean());ax.barh(y,value,left=start,height=.66,color=c,ec='black',lw=.6)
                if value>.08:box(ax,start+value/2,y,f'{100*value:.0f}%',fontsize=18)
                start+=value
        ax.set_yticks(range(len(levels)),[nm(m) for m in levels]);ax.invert_yaxis();ax.set_xlim(0,1)
        ax.set_xlabel('Fraction of paired outcomes · equal weight per problem')
        fig.legend(handles=[Patch(fc=c,ec='black',label=label) for label,c in [('FULL better on both',C(.8)),('Trade-off / one metric',C(.48)),('Both tied','#D8D3C5'),('FULL worse on both',C(.1))]],loc='outside lower center',ncol=2)
        self.emit(fig,6,'joint_metrics','Both metrics use the exact same paired seed and initial population. Each problem first yields category proportions; these proportions are then averaged equally over problems. Mixed includes improvement in one metric and deterioration or numerical tie in the other. This is descriptive, not a Pareto-front attainment probability.',table)

    def example_panels(self,require_history=False):
        chosen=[];main=self.raw[(self.raw.block=='main')&self.raw.method.isin(CURVE_METHODS)]
        grid={400,10000,25200,50000,100000}
        for f in FAMILIES:
            for problem in EXAMPLE_POOL:
                if family(problem)!=f:continue
                g=main[main.problem==problem]
                if set(g.method)!=set(CURVE_METHODS):continue
                seeds=set.intersection(*[set(g[g.method==m].seed) for m in CURVE_METHODS])
                good=[]
                for seed in seeds:
                    panel=g[g.seed==seed]
                    if panel.initial_hash.nunique()!=1:continue
                    if require_history and not all(grid.issubset(self.histories.get(str(rid),{})) for rid in panel.run_id):continue
                    good.append(seed)
                if len(good)>=3:
                    chosen.append(g[g.seed.isin(good)].copy());break
        return chosen

    def convergence(self,metric,number):
        panels=self.example_panels(True)
        if not panels:return self.skip(number,'convergence_'+metric,'No complete common history/seed panel on the prespecified example list.')
        nrows=math.ceil(len(panels)/2);fig,axes=self.subplots('Convergence · '+METRIC_NAMES[metric],rows=nrows,cols=2,width=23,height=6.8*nrows)
        table=[];grid=[400,10000,25200,50000,100000]
        for ax,g in zip(axes.flat,panels):
            r=g.iloc[0];n=g.seed.nunique()
            for method in CURVE_METHODS:
                sub=g[g.method==method]
                values=np.array([[self.histories[str(rid)][fe][metric] for fe in grid] for rid in sub.run_id])
                median=np.median(values,axis=0);low,high=np.quantile(values,[.25,.75],axis=0)
                ax.plot(grid,median,color=COLORS[method],lw=3.7 if method=='FULL' else 2.5,marker='o',ms=5,label=nm(method))
                ax.fill_between(grid,low,high,color=COLORS[method],alpha=.12)
                table.extend(dict(problem=r.problem,m=int(r.m),d=int(r.d),method=method,metric=metric,fe=fe,n=int(n),median=float(v),q25=float(lo),q75=float(hi)) for fe,v,lo,hi in zip(grid,median,low,high))
            ax.set_title(f'{r.problem} · M={int(r.m)}, D={int(r.d)}, n={n}')
            ax.set_xlabel('Function evaluations');ax.set_ylabel(METRIC_NAMES[metric])
            ax.ticklabel_format(axis='x',style='sci',scilimits=(4,4))
            if metric=='igd_plus':
                vals=np.array([v['median'] for v in table if v['problem']==r.problem]);positive=vals[vals>0]
                if len(positive)==len(vals) and positive.max()/positive.min()>100:ax.set_yscale('log')
        for ax in list(axes.flat)[len(panels):]:ax.remove()
        h,l=axes.flat[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=3)
        self.emit(fig,number,'convergence_'+metric,'Each axis is one fixed problem/M/D; raw metrics are never pooled across axes. Medians and IQR use the same seed set for all six methods and all five actual recorded FE points. One example per family is selected by a fixed name priority and availability, never by outcome. Primary choices are ZDT3, DTLZ7, WFG1, LZ09_F1, UF8, ZCAT20; fallback names are documented in the source. Full numerical membership is exported.',table)

    def distributions(self):
        panels=self.example_panels(False)
        if not panels:return self.skip(9,'seed_distributions','No common seed panel for the example functions.')
        nrows=math.ceil(len(panels)/2);fig,axes=self.subplots('Run-to-run variability · final hypervolume',rows=nrows,cols=2,width=23,height=6.4*nrows)
        out=[];rng=np.random.default_rng(19)
        for ax,g in zip(axes.flat,panels):
            r=g.iloc[0]
            for y,method in enumerate(CURVE_METHODS):
                vals=g[g.method==method].hv.to_numpy(float)
                bp=ax.boxplot([vals],positions=[y],vert=False,widths=.5,patch_artist=True,showfliers=False,
                              medianprops=dict(color='black',lw=2),boxprops=dict(facecolor=COLORS[method],alpha=.45),
                              whiskerprops=dict(color='black'),capprops=dict(color='black'))
                ax.scatter(vals,y+rng.uniform(-.13,.13,len(vals)),color=COLORS[method],s=28,alpha=.6,ec='black',lw=.3,zorder=4)
                out.extend(dict(problem=r.problem,m=int(r.m),d=int(r.d),method=method,seed=int(s),hv=float(v)) for s,v in zip(g[g.method==method].seed,vals))
            ax.set_yticks(range(len(CURVE_METHODS)),[nm(m) for m in CURVE_METHODS]);ax.invert_yaxis()
            ax.set_title(f'{r.problem} · M={int(r.m)}, D={int(r.d)}, n={g.seed.nunique()}')
            ax.set_xlabel('Final HV ↑')
        for ax in list(axes.flat)[len(panels):]:ax.remove()
        self.emit(fig,9,'seed_distributions','Each axis preserves the raw HV scale of one fixed task. The same seeds and methods are used within an axis. Boxes show median, quartiles and 1.5 IQR whiskers; every measured seed is also plotted. Box overlap is not used as a significance test.',out)

    def ablations(self):
        d=self.pairs[self.pairs.study=='ablation']
        if d.empty:return self.skip(10,'ablations','No paired ablation outcomes.')
        problems=[p for p in PANEL if p in set(d.problem)];cols=list(ABLATIONS)
        fig,axes=self.subplots('Component contributions · FULL versus each ablation',width=24,height=max(8,3+.8*len(problems)))
        for ax,metric in zip(axes.flat,METRICS):
            matrix=np.full((len(problems),len(cols)),np.nan);labels=[['']*len(cols) for _ in problems]
            for i,p in enumerate(problems):
                for j,c in enumerate(cols):
                    g=d[(d.problem==p)&(d.level==c)&(d.metric==metric)]
                    if len(g):
                        r=g.iloc[0];matrix[i,j]=r.probability
                        labels[i][j]=f'{100*r.probability:.0f}%'+('*' if r.eligible and r.p_holm<.05 else '')
            heat(ax,matrix,problems,cols,METRIC_NAMES[metric],labels)
        self.emit(fig,10,'ablations','Each row is a fixed problem/M/D. Cells are paired seed-win percentages for FULL against the indicated ablation, with half-weight numerical ties. Stars require all 30 planned pairs and a Holm-adjusted p<.05 across the entire ablation block and both metrics. High win percentage alone is not a significant component contribution.',d)

    def sensitivities(self,metric,number):
        defaults={'interval':10,'rho':.1,'temperature':1.,'quality':'rank_crowding'}
        choices={'interval':['5','10','20'],'rho':['0.05','0.1','0.2'],'temperature':['0.5','1.0','2.0'],
                 'quality':['rank_hv','rank_crowding','rank_novelty']}
        d=self.pairs[(self.pairs.study.str.startswith('sensitivity_'))&(self.pairs.metric==metric)]
        if d.empty:return self.skip(number,'sensitivity_'+metric,'No paired sensitivity data.')
        problems=[p for p in PANEL if p in set(d.problem)]
        fig,axes=self.subplots('Parameter sensitivity · '+METRIC_NAMES[metric],rows=2,cols=2,width=24,height=max(12,5+1.15*len(problems)))
        for ax,(parameter,default) in zip(axes.flat,defaults.items()):
            cols=choices[parameter];matrix=np.full((len(problems),len(cols)),np.nan);labels=[['']*3 for _ in problems]
            for i,p in enumerate(problems):
                sub=d[(d.problem==p)&(d.study=='sensitivity_'+parameter)]
                if sub.empty:continue
                for j,c in enumerate(cols):
                    if c==str(default):matrix[i,j]=.5;labels[i][j]='ref.';continue
                    row=sub[sub.level==c]
                    if len(row):
                        r=row.iloc[0];matrix[i,j]=r.probability
                        labels[i][j]=f'{100*r.probability:.0f}%'+('*' if r.eligible and r.p_holm<.05 else '')
            heat(ax,matrix,problems,cols,parameter.replace('_',' ').title(),labels)
        self.emit(fig,number,'sensitivity_'+metric,'Default FULL versus one changed parameter at a time, paired within each row/task. Higher percentage means the default wins more often. The reference column is neutral by definition. Stars use 30 complete seeds and Holm over all four sensitivity studies and both indicators. This is the existing one-factor experiment, not a joint factorial response surface.',d)

    def composition(self):
        obs=self.obs[self.obs.study=='homogeneous'];records=[]
        for (problem,policy),g in obs.groupby(['problem','level'],sort=False):
            base=g[g.composition=='mixed'].set_index('seed')
            if base.empty:continue
            for comp in ('NSGA2','MOEAD','GDE3','SMS'):
                alt=g[g.composition==comp].set_index('seed');common=base.index.intersection(alt.index)
                for seed in common:
                    a,b=base.loc[seed],alt.loc[seed]
                    if a.initial_hash!=b.initial_hash:continue
                    if any(a[k]!=b[k] for k in ('m','d','budget','population','islands','interval','rho','temperature','quality')):continue
                    for metric in METRICS:
                        delta=(a[metric]-b[metric])*(1 if metric=='hv' else -1);tol=1e-12*max(1,abs(a[metric]),abs(b[metric]))
                        records.append(dict(problem=problem,policy=policy,composition=comp,metric=metric,seed=int(seed),win=1. if delta>tol else 0. if delta < -tol else .5))
        table=pd.DataFrame(records)
        if table.empty:return self.skip(13,'composition','No mixed/homogeneous comparisons at identical settings.')
        fig,axes=self.subplots('Island composition · mixed versus homogeneous optimizers',width=22,height=8)
        cols=['NSGA2','MOEAD','GDE3','SMS'];policies=list(POLICIES)
        for ax,metric in zip(axes.flat,METRICS):
            matrix=np.full((3,4),np.nan)
            for i,policy in enumerate(policies):
                for j,comp in enumerate(cols):
                    g=table[(table.policy==policy)&(table.composition==comp)&(table.metric==metric)]
                    if len(g):matrix[i,j]=g.groupby('problem').win.mean().mean()
            heat(ax,matrix,[nm(p) for p in policies],cols,METRIC_NAMES[metric])
            ax.set_xlabel('Homogeneous optimizer being compared')
        self.emit(fig,13,'composition','Every cell compares mixed and homogeneous islands under the SAME migration policy, problem/M/D, K, total population and FE budget. Percentages average within-problem paired wins, then weight problems equally. This displays the full composition-by-policy experiment; it does not attribute composition gains to the migration controller.',table)

    def island_count(self):
        data=self.detail[(self.detail.study=='islands')&self.detail.level.isin(('NO_MIG','ELITE'))]
        if data.empty:return self.skip(14,'island_count','No within-K migration comparisons.')
        fig,axes=self.subplots('Island count · does migration still help?',height=9);out=[]
        for ax,metric in zip(axes.flat,METRICS):
            for method in ('NO_MIG','ELITE'):
                g=data[(data.metric==metric)&(data.level==method)]
                byk={k:g[g.facet==f'K={k}'] for k in (4,8,16)}
                available={k:v for k,v in byk.items() if len(v)}
                if not available:continue
                common=set.intersection(*[set(v.problem) for v in available.values()])
                xx=[];yy=[];lo=[];hi=[]
                for k,sub in available.items():
                    sub=sub[sub.problem.isin(common)];v,l,h,n=probability_summary(sub)
                    xx.append(k);yy.append(v);lo.append(l);hi.append(h)
                    out.append(dict(metric=metric,alternative=method,k=k,probability=v,ci_low=l,ci_high=h,problems=n))
                    box(ax,k,v,f'{100*v:.0f}%',fontsize=17,va='bottom')
                ax.plot(xx,yy,marker='o',color=COLORS[method],lw=3,label='FULL vs '+nm(method))
                ax.fill_between(xx,lo,hi,color=COLORS[method],alpha=.14)
            ax.axhline(.5,c='black',lw=1.2,ls='--');ax.set_ylim(-.04,1.1);ax.set_xticks([4,8,16])
            ax.set_title(METRIC_NAMES[metric]);ax.set_xlabel('Number of islands K');ax.set_ylabel('Paired probability FULL wins')
        h,l=axes.flat[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=2)
        self.emit(fig,14,'island_count','FULL is compared with no migration and elite separately within each K; there is no cross-K seed pairing. Each alternative uses the common set of available problems over the shown K values. Total population and FE budget follow the original fixed protocol. Bands use a hierarchical problem/seed bootstrap. K=4 is reused from the main experiment.',out)

    def metric_matrix(self,ax,data,problems,methods,metric,title):
        matrix=np.full((len(problems),len(methods)),np.nan);labels=[['']*len(methods) for _ in problems];rows=[];out=[]
        for i,p in enumerate(problems):
            g=data[data.problem==p];first=g.iloc[0] if len(g) else None
            rows.append(f'{p} · D={int(first.d)}' if first is not None else p)
            vals=[];indices=[]
            for j,method in enumerate(methods):
                sub=g[g.method==method]
                if len(sub):
                    value=float(sub[metric].median());vals.append(value);indices.append(j)
                    labels[i][j]=f'{value:.3g}'+('†' if sub.seed.nunique()<30 else '')
                    out.append(dict(problem=p,m=int(first.m),d=int(first.d),method=method,metric=metric,median=value,n=int(sub.seed.nunique())))
            if vals:
                rank=stats.rankdata(-np.array(vals) if metric=='hv' else vals)
                for j,r in zip(indices,rank):matrix[i,j]=1-(r-1)/max(1,len(vals)-1)
        heat(ax,matrix,rows,methods,title,labels)
        return out

    def dimensions(self):
        data=self.raw[self.raw.block=='dimensions']
        if data.empty:return self.skip(15,'dimensions','No completed D=50/100 trials.')
        problems=[p for p in DIM_PROBLEMS if p in set(data.problem)]
        fig,axes=self.subplots('Decision-space dimension · fixed D in every panel',rows=2,cols=2,width=26,height=max(12,5+1.3*len(problems)))
        out=[]
        for i,d in enumerate((50,100)):
            for j,metric in enumerate(METRICS):out+=self.metric_matrix(axes[i,j],data[data.d==d],problems,DIM_METHODS,metric,f'D={d} · '+METRIC_NAMES[metric])
        self.emit(fig,15,'dimensions','D=50 and D=100 are kept in separate panels. Numbers are actual within-task seed medians. Color is a within-row quality rank, not a shared raw scale. A dagger marks n<30; missing cells are empty. This descriptive matrix retains all eight planned methods and does not pool scores or pair genotypes across dimensions.',out)

    def many_objectives(self):
        data=self.raw[self.raw.block=='many']
        if data.empty:return self.skip(16,'many_objectives','No completed M=5 trials.')
        problems=[p for p in MANY_PROBLEMS if p in set(data.problem)];methods=(*METHODS,'NSGA3','RVEA')
        fig,axes=self.subplots('Five-objective optimization · all twelve methods',rows=2,cols=1,width=25,height=max(11,5+1.8*len(problems)))
        out=[]
        for ax,metric in zip(axes.flat,METRICS):out+=self.metric_matrix(ax,data,problems,methods,metric,'M=5 · '+METRIC_NAMES[metric])
        self.emit(fig,16,'many_objectives','All twelve planned methods, including SMS-EMOA, NSGA-III and RVEA. Every row has one fixed problem/M/D. Numbers are raw seed medians, color is a within-row rank, and † indicates fewer than 30 repeats. No raw scores are averaged across functions or objective counts.',out)

    def quality_cost(self):
        main=self.raw[(self.raw.block=='main')&(self.raw.problem!='ZDT5')];rows=[]
        for problem,g in main.groupby('problem',sort=False):
            base=g[g.method=='NO_MIG'].set_index('seed')
            if base.empty:continue
            for method in ('RANDOM','ELITE','HV_GATE','CROWDING','FULL'):
                alt=g[g.method==method].set_index('seed');seeds=base.index.intersection(alt.index)
                for seed in seeds:
                    a,b=alt.loc[seed],base.loc[seed]
                    if a.initial_hash!=b.initial_hash:continue
                    for metric in METRICS:
                        delta=(a[metric]-b[metric])*(1 if metric=='hv' else -1);tol=1e-12*max(1,abs(a[metric]),abs(b[metric]))
                        rows.append(dict(problem=problem,method=method,metric=metric,seed=int(seed),
                                         win=1. if delta>tol else 0. if delta < -tol else .5,
                                         logtime=math.log(float(a.seconds)/float(b.seconds))))
        table=pd.DataFrame(rows)
        if table.empty:return self.skip(17,'quality_cost','No policy/no-migration matched runs.')
        fig,axes=self.subplots('Quality gain and measured runtime · migration policies',height=10)
        rng=np.random.default_rng(1027)
        for ax,metric in zip(axes.flat,METRICS):
            for method in ('RANDOM','ELITE','HV_GATE','CROWDING','FULL'):
                g=table[(table.method==method)&(table.metric==metric)]
                if g.empty:continue
                cells=g.groupby('problem').agg(win=('win','mean'),logtime=('logtime','median'))
                center=cells.mean();picks=rng.integers(0,len(cells),(600,len(cells)))
                boot=cells.to_numpy()[picks].mean(axis=1);lo,hi=np.quantile(boot,[.025,.975],axis=0)
                x,y=np.exp(center.logtime),center.win
                ax.plot(np.exp([lo[1],hi[1]]),[y,y],c=COLORS[method],lw=2)
                ax.plot([x,x],[lo[0],hi[0]],c=COLORS[method],lw=2)
                ax.scatter(x,y,s=220 if method=='FULL' else 145,c=[COLORS[method]],marker='*' if method=='FULL' else 'o',
                           ec='black',lw=1.2,zorder=5,label=f'{nm(method)} · {len(cells)} problems')
            ax.axvline(1,color='black',ls='--',lw=1.3);ax.axhline(.5,color='black',ls='--',lw=1.3)
            ax.set_xscale('log');ax.set_ylim(-.03,1.04);ax.set_title(METRIC_NAMES[metric])
            ax.set_xlabel('Observed runtime / no-migration runtime')
            ax.set_ylabel('Paired probability of quality improvement')
        h,l=axes.flat[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=3)
        self.emit(fig,17,'quality_cost','Every policy is compared with no migration at identical problem/M/D, population and budget with matched seeds/initial hashes. Quality is first averaged per problem. Runtime is the geometric mean of per-problem median paired ratios. Whiskers bootstrap problems. Shared-server load can affect observed wall time; these are not physical speedup or CPU-efficiency claims.',table)

    def migration_cost(self):
        methods=('RANDOM','ELITE','HV_GATE','CROWDING','FULL')
        data=self.raw[(self.raw.block=='main')&(self.raw.problem!='ZDT5')&self.raw.method.isin(methods)].copy()
        if data.empty:return self.skip(18,'migration_cost','No migration-policy cost counters.')
        for col in ('sent','accepted','control_seconds','integration_seconds','payload_bytes'):
            if col not in data:data[col]=np.nan
        data['sent_rate']=data.sent/data.budget*10000
        data['accepted_rate']=data.accepted/data.budget*10000
        data['control_share']=100*data.control_seconds/data.seconds
        data['integration_share']=100*data.integration_seconds/data.seconds
        data['payload_mib']=data.payload_bytes/2**20/data.budget*10000
        cells=data.groupby(['problem','method'])[['sent_rate','accepted_rate','control_share','integration_share','payload_mib']].median().reset_index()
        fig,axes=self.subplots('Migration efficiency and implementation cost',cols=3,width=25,height=10)
        for y,method in enumerate(methods):
            g=cells[cells.method==method]
            if g.empty:continue
            a=float(g.sent_rate.median());b=float(g.accepted_rate.median())
            axes[0,0].plot([b,a],[y,y],c='black',lw=2)
            axes[0,0].scatter(a,y,s=150,c='white',ec='black',zorder=4)
            axes[0,0].scatter(b,y,s=150,c=[COLORS[method]],ec='black',zorder=5)
            cvals=g.control_share.dropna();ivals=g.integration_share.dropna()
            ctrl=float(cvals.median()) if len(cvals) else np.nan
            integration=float(ivals.median()) if len(ivals) else np.nan
            if np.isfinite(ctrl) and np.isfinite(integration):
                axes[0,1].barh(y,ctrl,height=.55,color=C(.28),ec='black',lw=.7)
                axes[0,1].barh(y,integration,left=ctrl,height=.55,color=C(.76),ec='black',lw=.7)
                box(axes[0,1],ctrl+integration,y,f'{ctrl+integration:.2g}%',fontsize=17,ha='left')
            else:box(axes[0,1],0,y,'—',fontsize=17)
            pvals=g.payload_mib.dropna();payload=float(pvals.median()) if len(pvals) else np.nan
            if np.isfinite(payload):
                gradient(axes[0,2],y,0,payload);box(axes[0,2],payload,y,f'{payload:.2g}',fontsize=17,ha='left')
            else:box(axes[0,2],0,y,'—',fontsize=17)
        for j,ax in enumerate(axes.flat):
            ax.set_yticks(range(len(methods)),[nm(m) for m in methods] if j==0 else ['']*len(methods));ax.invert_yaxis()
            if j in (1,2):
                low,high=ax.get_xlim();ax.set_xlim(0,max(high*1.3,1e-6))
        axes[0,0].set_title('Sent versus accepted');axes[0,0].set_xlabel('Migrants per 10,000 FE')
        axes[0,1].set_title('Measured coordination overhead');axes[0,1].set_xlabel('Share of observed wall time (%)')
        axes[0,2].set_title('Logical migration payload');axes[0,2].set_xlabel('MiB per 10,000 FE')
        fig.legend(handles=[Line2D([0],[0],marker='o',color='none',mfc='white',mec='black',label='Sent'),
                            Line2D([0],[0],marker='o',color='none',mfc=C(.7),mec='black',label='Accepted'),
                            Patch(fc=C(.28),ec='black',label='Controller'),Patch(fc=C(.76),ec='black',label='Integration')],
                   loc='outside lower center',ncol=4)
        self.emit(fig,18,'migration_cost','Runs are reduced to a per-problem seed median, then to a median across problems; no raw quality scores are pooled. Sent, accepted and logical payload counters are normalized by the same FE budget. Controller and integration timers are measured wall-time shares. Logical bytes are not measured network traffic. Exact per-problem values and available panels are exported.',cells)

    def finish(self,manifest):
        write_csv(self.fd/'figure_index.csv',self.entries)
        lines=['# TherMOO compact figures V9','',f"Cohort: `{manifest['fingerprint']}`. Original heuristic FULL protocol.",'']
        cards=[]
        for r in self.entries:
            lines.extend([f"## {r['stem']} · {r['status']}",'',r['caption'],''])
            if r['status']=='created':
                stem=html.escape(r['stem'])
                cards.append(f'<article><h2>{stem.replace("_"," ")}</h2><a href="{stem}.png"><img loading="lazy" src="{stem}.png"></a><p>{html.escape(r["caption"])}</p><a href="{stem}.pdf">Vector PDF</a></article>')
        (self.fd/'captions.md').write_text('\n'.join(lines),encoding='utf-8')
        page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TherMOO compact figures</title><style>body{background:#FAF8F2;color:#161616;font:19px system-ui;margin:24px auto;max-width:1500px;padding:20px}h1{font-size:38px}h2{font-size:26px}article{background:#F4F1E8;padding:25px;margin:30px 0;border:1px solid #bbb;border-radius:12px}img{width:100%;height:auto}p{line-height:1.55}</style><h1>TherMOO · compact experimental figures</h1><p>18 fixed scientific figure slots. No population trajectories or experiment progress charts.</p>'''+''.join(cards)
        (self.fd/'index.html').write_text(page,encoding='utf-8')


PROTOCOL='''# Compact figure protocol V9

This renderer summarises the ORIGINAL 33,240-run heuristic experiment. Neural V5/V6
results are different protocols and are never merged into these comparisons.

All new writes use the canonical root /home/tahiti/TherMOO. Both original input
locations can be read. Fingerprints are kept separate; conflicting duplicated
scientific outcomes are excluded. A run must have a complete, committed FE budget.
The freeze cutoff is the beginning of this script invocation.

Raw HV and IGD+ are NOT averaged across different functions, M, D or budgets.
Aggregate figures use within-task ranks or paired seed-win probabilities, then
weight benchmark problems equally. Controlled D and M variations retain separate
panels; all variant/control pairing is within identical settings.

Main ranks/performance profiles use the predeclared nine-method core. SMS-EMOA is
reported in pairwise summaries, family matrices, D and M=5 matrices. It cannot be
silently included in a rank panel with different or missing problems. The core
uses complete 30-seed task panels when at least three such tasks exist; otherwise
it is explicitly descriptive and requires at least three shared seeds per task.
CD uses a Friedman gate and Nemenyi at .025 per indicator (two indicators).

Paired inference requires exactly seeds 0..29 and equal initial-population hashes.
Shapiro tests paired differences; a paired t test is used for normal differences,
and Wilcoxon-Pratt otherwise. Constant ties have p=1. Holm is applied jointly to
all planned contrasts of a block and both indicators. The exploratory rendering
of partial n does not grant it a significance claim. No detected difference is
not statistical equivalence. Bootstrap intervals are descriptive.

ZDT5 is retained in the full raw output but excluded from aggregate scientific
inference, because the original binary implementation used a SPEA2 substitute
in the GDE3 slot and requires a separate binary-method audit.

History examples are chosen by fixed priorities and availability, not performance:
ZDT3 > ZDT1; DTLZ7 > DTLZ2; WFG1 > WFG9; LZ09_F1; UF8 > UF1; ZCAT20 > ZCAT14.
Each convergence axis uses the same seeds across six methods and five actual
recorded FE checkpoints. Bands show IQR, not standard errors or tests.

Wall times reflect server contention. No plot establishes physical parallel
speedup, causal migration benefit from correlational counters, or neural efficacy.
Color ranks in raw-value matrices are local to each row. A dagger marks n<30.

Figure design was informed by:
* Li et al. (2023), Swarm and Evolutionary Computation 77, 101253:
  https://doi.org/10.1016/j.swevo.2023.101253 (rank distributions, overall ranks,
  selected seed boxplots, controlled runtime comparisons; figures 3,4,13,14).
* Benitez-Hidalgo et al. (2019), Swarm and Evolutionary Computation 51, 100598:
  https://doi.org/10.1016/j.swevo.2019.100598 (boxplots, CD and statistical summaries).
* Osaba et al. (2021), Swarm and Evolutionary Computation 64, 100888:
  https://doi.org/10.1016/j.swevo.2021.100888 (experimental reporting context).

Visual template: cataug/fractional-lora-unlearning, FIGURES/build_figures.py and
FIGURES/build_figures_extended.py. Cream backgrounds, transparent viridis fills,
black axes/ticks, large bold labels, white rounded numeric boxes, 320 dpi PNG and
vector PDF with embedded fonts. Style is adapted to the fixed compact panel count.
'''


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/home/tahiti/TherMOO'))
    parser.add_argument('--input-root',type=Path,action='append')
    parser.add_argument('--snapshot',type=Path,help='Existing original STATS/vN snapshot; default reads current committed results')
    parser.add_argument('--fingerprint',help='Optional original cohort identifier')
    parser.add_argument('--dpi',type=int,default=320)
    parser.add_argument('--figures',help='Optional comma-separated figure numbers, 1..18')
    args=parser.parse_args(argv)
    root=args.root.expanduser().resolve()
    if args.snapshot:raw,source,issues=collect_snapshot(args.snapshot.expanduser().resolve())
    else:
        roots=args.input_root or [root,Path('/home/tahiti/Malashin_Projects/TherMOO')]
        roots=list(dict.fromkeys(p.expanduser().resolve() for p in roots if p.expanduser().is_dir()))
        raw,source,issues=collect_live(roots,args.fingerprint)
    version,sd,fd=allocate_version(root)
    manifest=dict(source,version=version,root=str(root),renderer_version=VERSION,status='RUNNING',
                  cutoff_utc=datetime.fromtimestamp(int(source['cutoff_ns'])/1e9,timezone.utc).isoformat(),
                  planned_trials=33240,accepted_trials=len(raw),figure_limit=18)
    atomic_json(sd/'manifest.json',manifest)
    write_csv(sd/'per_run.csv',raw);write_csv(sd/'issues.csv',issues)
    log(f'[SOURCE] {len(raw):,} committed trials; fingerprint={source["fingerprint"]}')
    log(f'[OUTPUT] {fd} · at most 18 figures')
    normalized=[normalize_job(j,family) for j in jobs()]
    planned=make_views(normalized);obs=observed_views(raw,planned)
    write_csv(sd/'planned_runs.csv',normalized)
    log('[STATS] paired differences, normality, tests and Holm correction')
    pairs,detail=paired_analysis(planned,obs)
    if pairs.empty:raise RuntimeError('No matched original control/variant pairs are available.')
    write_csv(sd/'comparisons.csv',pairs);write_csv(sd/'paired_seeds.csv',detail)
    main_panel,infer=balanced_main(obs);scores=balanced_scores(main_panel)
    write_csv(sd/'rank_cases.csv',scores)
    histories=selected_histories(raw,source,sd)
    renderer=CompactFigures(raw,obs,pairs,detail,scores,infer,histories,sd,fd,args.dpi)
    operations=[renderer.ranks,renderer.family_map,renderer.significance,renderer.profiles,renderer.effects,
                renderer.joint,lambda:renderer.convergence('hv',7),lambda:renderer.convergence('igd_plus',8),
                renderer.distributions,renderer.ablations,lambda:renderer.sensitivities('hv',11),
                lambda:renderer.sensitivities('igd_plus',12),renderer.composition,renderer.island_count,
                renderer.dimensions,renderer.many_objectives,renderer.quality_cost,renderer.migration_cost]
    selected={int(x) for x in args.figures.split(',')} if args.figures else set(range(1,19))
    for number,operation in enumerate(operations,1):
        if number in selected:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',category=FutureWarning);operation()
    renderer.finish(manifest)
    (sd/'PROTOCOL.md').write_text(PROTOCOL,encoding='utf-8')
    manifest.update(status='COMPLETE',created_figures=sum(r['status']=='created' for r in renderer.entries),
                    figures=str(fd),finished_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(sd/'manifest.json',manifest)
    log(f'[DONE] {version}: {manifest["created_figures"]}/18 figures · {fd/"index.html"}')


if __name__=='__main__':
    main()
