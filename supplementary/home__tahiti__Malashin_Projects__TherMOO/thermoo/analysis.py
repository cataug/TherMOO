"""Seed-paired inference, family summaries, complete-case ranks and cream plots."""
from .runtime import ROOT, stable_seed
from .plan import Job, family
from .engine import experiment_fingerprint, atomic_json
from dataclasses import replace
from pathlib import Path
from collections import defaultdict
import json

import numpy as np
import pandas as pd
from scipy import stats


def holm(values):
    values = np.asarray(values, float)
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    running = 0.
    for position, index in enumerate(order):
        running = max(running, (len(values)-position)*values[index])
        adjusted[index] = min(1., running)
    return adjusted


def paired_stats(a, b, seed, resamples=50000):
    d = np.asarray(a)-np.asarray(b)
    n = len(d)
    rng = np.random.default_rng(seed)
    if np.all(d == 0):
        return dict(n=n, difference=0., ci_low=0., ci_high=0., p_primary=1., p_shapiro=None, p_ttest=None, p_wilcoxon=None, rank_biserial=0.)
    observed = abs(float(np.mean(d)))
    greater = 0
    for start in range(0,resamples,5000):
        size = min(5000,resamples-start)
        signs = 2*rng.integers(0,2,(size,n))-1
        greater += int(np.count_nonzero(np.abs((signs*d).mean(axis=1)) >= observed-1e-15))
    p = (greater+1)/(resamples+1)
    sample = d[rng.integers(0,n,(4000,n))].mean(axis=1)
    low, high = np.quantile(sample,[.025,.975])
    if np.ptp(d) <= 1e-14:
        shapiro, ttest = None, 0.
    else:
        shapiro = float(stats.shapiro(d).pvalue) if n >= 3 else None
        ttest = float(stats.ttest_rel(a,b).pvalue) if shapiro is not None and shapiro >= .05 else None
    nz = d[d != 0]
    ranks = stats.rankdata(np.abs(nz))
    effect = float(np.sum(ranks*np.sign(nz))/np.sum(ranks))
    # Signed-rank is reported only as a sensitivity check when symmetry is plausible.
    symmetric = len(nz)>=8 and (np.ptp(nz)<1e-14 or stats.skewtest(nz).pvalue >= .05)
    w = float(stats.wilcoxon(d,zero_method="pratt",method="approx").pvalue) if symmetric else None
    return dict(n=n,difference=float(d.mean()),ci_low=float(low),ci_high=float(high),p_primary=float(p),p_shapiro=shapiro,p_ttest=ttest,p_wilcoxon=w,rank_biserial=effect)


def case_key(job):
    # Source energy/rate parameters are *not* hidden in a generic method label.
    return f"{job.problem}|M{job.objectives}|D{job.variables}|B{job.budget}|P{job.population}|K{job.islands}|{job.composition}|I{job.interval}|R{job.rho}|T{job.temperature}|Q{job.quality}"


def read_results(output):
    folder=ROOT/"outputs"/output/experiment_fingerprint()
    results=[]
    for path in sorted(folder.glob("*/result.json")):
        data=json.loads(path.read_text())
        if data.get("status")!="complete":
            continue
        job=Job(**data["job"])
        row=dict(run_id=job.signature(),block=job.block,problem=job.problem,family=family(job.problem),
                 m=job.objectives,d=job.variables,seed=job.seed,label=("SPEA2" if job.problem=="ZDT5" and job.method=="GDE3" else job.label),case=case_key(job),
                 method=job.method,composition=job.composition,k=job.islands,interval=job.interval,
                 rho=job.rho,temperature=job.temperature,quality=job.quality,ablation=job.ablation,
                 budget=job.budget,actual_evaluations=data["evaluations"],hv=data["metrics"]["hv"],igd_plus=data["metrics"]["igd_plus"],
                 seconds=data["seconds"],sent=data["sent"],accepted=data["accepted"],payload_bytes=data["payload_bytes"],
                 control_seconds=data["control_seconds"],integration_seconds=data["integration_seconds"],
                 diagnostic_seconds=data["diagnostic_seconds"],initial_hash=data["initial_hash"],actual_algorithms="|".join(data["algorithms"]),folder=str(path.parent))
        results.append((row,job))
    return results


def analyze(output="full", plots=False):
    pairs=read_results(output)
    if not pairs:
        raise RuntimeError(f"No completed results for {output}/{experiment_fingerprint()}")
    out=ROOT/"reports"/output/experiment_fingerprint()
    out.mkdir(parents=True,exist_ok=True)
    raw=pd.DataFrame([r for r,j in pairs])
    raw.to_csv(out/"per_run.csv",index=False)
    summary=raw.groupby(["block","case","label"],dropna=False).agg(
        runs=("seed","nunique"),hv_mean=("hv","mean"),hv_std=("hv","std"),hv_median=("hv","median"),
        igd_mean=("igd_plus","mean"),igd_std=("igd_plus","std"),igd_median=("igd_plus","median"),
        seconds_mean=("seconds","mean"),migrants_mean=("sent","mean"),bytes_mean=("payload_bytes","mean")).reset_index()
    intervals=[]
    for (block,case,label),group in raw.groupby(["block","case","label"],dropna=False):
        record=dict(block=block,case=case,label=label)
        for metric in ("hv","igd_plus"):
            values=group[metric].to_numpy()
            if len(values)<30:
                lo,hi=np.nan,np.nan
            else:
                rng=np.random.default_rng(stable_seed(block,case,label,metric,"mean_ci"))
                means=values[rng.integers(0,len(values),(4000,len(values)))].mean(axis=1)
                lo,hi=np.quantile(means,[.025,.975])
            record[metric+"_ci_low"]=lo
            record[metric+"_ci_high"]=hi
        intervals.append(record)
    summary=summary.merge(pd.DataFrame(intervals),on=["block","case","label"],how="left")
    summary.to_csv(out/"summary.csv",index=False)
    lookup={j.signature():r for r,j in pairs}
    tests=[]
    # Compare all relevant variants with the exact matching FULL control.
    buckets=defaultdict(list)
    for row,job in pairs:
        if job.method=="FULL" and job.ablation=="none" and job.block not in ("sensitivity",):
            continue
        if job.block in ("ablation","sensitivity"):
            control=replace(job,method="FULL",block="main",ablation="none",interval=10,rho=.1,temperature=1.,quality="rank_crowding")
            contrast=f"{job.label}|I{job.interval}|R{job.rho}|T{job.temperature}|Q{job.quality}"
            test_case=f"{job.problem}|M{job.objectives}|D{job.variables}"
        else:
            control=replace(job,method="FULL",ablation="none")
            contrast=row["label"]
            test_case=case_key(job)
        matched=lookup.get(control.signature())
        if matched is not None:
            if row["initial_hash"]!=matched["initial_hash"]:
                raise AssertionError("Paired comparison uses different initial samples")
            buckets[(job.block,test_case,contrast)].append((row,matched))
    for (block,case,contrast),values in buckets.items():
        values=sorted(values,key=lambda pair:pair[0]["seed"])
        if len({x[0]["seed"] for x in values})<30 or output!="full":
            continue
        for metric,orientation in (("hv",1),("igd_plus",-1)):
            # Positive differences always mean FULL is better.
            baseline=np.array([r[metric] for r,c in values])*orientation
            full=np.array([c[metric] for r,c in values])*orientation
            result=paired_stats(full,baseline,stable_seed(block,case,contrast,metric))
            tests.append(dict(block=block,case=case,contrast=contrast,metric=metric,**result))
    inference=pd.DataFrame(tests)
    if not inference.empty:
        inference["p_holm"]=np.nan
        # Both endpoints and every per-problem contrast share one family within each block.
        for block,group in inference.groupby("block"):
            inference.loc[group.index,"p_holm"]=holm(group.p_primary)
        inference["outcome"]=np.where(inference.p_holm>=.05,"tie",np.where(inference.difference>0,"FULL_wins","FULL_loses"))
        inference.to_csv(out/"paired_tests.csv",index=False)
        inference.groupby(["block","contrast","metric","outcome"]).size().unstack(fill_value=0).to_csv(out/"wins_ties_losses.csv")
    else:
        pd.DataFrame(columns=["block","case","contrast","metric","n","difference","ci_low","ci_high","p_primary","p_holm","outcome"]).to_csv(out/"paired_tests.csv",index=False)
    rankings=[]
    omnibus=[]
    for block in ("main","many","dimensions"):
        part=raw[(raw.block==block)&(raw.problem!="ZDT5")]
        if part.empty:
            continue
        eligible=part.groupby(["case","label"]).seed.nunique().unstack()
        eligible=eligible.dropna()
        eligible=eligible[(eligible>=30).all(axis=1)]
        if len(eligible)<3 or len(eligible.columns)<3:
            continue
        for metric,ascending in (("hv",False),("igd_plus",True)):
            table=part.groupby(["case","label"])[metric].median().unstack().loc[eligible.index,eligible.columns]
            ranks=table.rank(axis=1,ascending=ascending,method="average")
            family_names=[family(case.split('|')[0]) for case in ranks.index]
            balanced=ranks.assign(family=family_names).groupby('family').mean().mean()
            for label,value in ranks.mean().items():
                rankings.append(dict(block=block,metric=metric,label=label,mean_rank=float(value),family_balanced_mean_rank=float(balanced[label])))
            if (table.nunique(axis=1)==1).all():
                statistic,p=0.,1.
            else:
                statistic,p=stats.friedmanchisquare(*[table[c].to_numpy() for c in table])
            comparisons=[]
            if "FULL" in table:
                for label in table.columns:
                    if label=="FULL":continue
                    d=table["FULL"]-table[label]
                    p_pair=1. if np.all(d==0) else float(stats.wilcoxon(d,zero_method="pratt",method="auto").pvalue)
                    comparisons.append(dict(label=label,p=p_pair))
                corrected=holm([c["p"] for c in comparisons])
                for item,adjusted in zip(comparisons,corrected):item["p_holm"]=float(adjusted)
            omnibus.append(dict(block=block,metric=metric,n_cases=len(table),friedman_statistic=float(statistic),p=float(p),posthoc_full=comparisons))
    pd.DataFrame(rankings,columns=['block','metric','label','mean_rank','family_balanced_mean_rank']).to_csv(out/"mean_ranks.csv",index=False)
    atomic_json(out/"friedman_posthoc.json",omnibus)
    raw.groupby(["block","family","label"])[["hv","igd_plus","seconds","sent"]].mean().to_csv(out/"family_summaries.csv")
    # A timing estimate is useful after a pilot; actual objective counts are retained.
    timing=raw.groupby(["problem","m","method"]).agg(seconds=("seconds","median"),budget=("budget","median"),trials=("seed","size")).reset_index()
    timing["seconds_per_100k_linear_estimate"]=timing.seconds*100000/timing.budget
    timing.to_csv(out/"timing_estimates.csv",index=False)
    expected=33240 if output=="full" else None
    atomic_json(out/"coverage.json",dict(completed=len(raw),expected=expected,complete=(len(raw)==expected),minimum_runs_per_cell=int(summary.runs.min()),inference_requires=30,
        primary="paired sign-flip randomization of mean differences; 50,000 resamples; Holm across both endpoints within block",
        ci="paired percentile bootstrap of mean differences; 4,000 resamples",pairing="same problem/dimension/objective/seed initial population",
        conditional_tests="Shapiro diagnostics; paired t sensitivity when Shapiro p>=.05; signed-rank sensitivity when symmetry is plausible"))
    if plots:
        draw(raw,out)
    print(f"[ANALYSIS] {len(raw)} complete trials; reports: {out}",flush=True)


def draw(raw,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor":"#FAF8F2","axes.facecolor":"#F4F1E8","savefig.facecolor":"#FAF8F2",
        "font.size":16,"font.family":"DejaVu Sans","xtick.labelsize":15,"ytick.labelsize":15,"legend.fontsize":14,
        "axes.labelsize":18,"axes.labelweight":"bold","axes.titlesize":20,"axes.titleweight":"bold",
        "axes.edgecolor":"black","axes.labelcolor":"black","xtick.color":"black","ytick.color":"black",
        "axes.spines.top":True,"axes.spines.right":True,"axes.linewidth":1.0,"pdf.fonttype":42})
    target=out/"figures"
    target.mkdir(exist_ok=True)
    methods=["NO_MIG","ELITE","HV_GATE","CROWDING","FULL"]
    colors=plt.get_cmap("viridis")(np.linspace(.12,.87,len(methods)))
    for problem in raw[raw.block=="main"].problem.unique():
        part=raw[(raw.block=="main")&(raw.problem==problem)]
        for metric,title in (("hv","Normalized hypervolume"),("igd_plus","IGD+")):
            fig,ax=plt.subplots(figsize=(8.5,6))
            for method,color in zip(methods,colors):
                histories=[]
                for _,row in part[part.label==method].iterrows():
                    values=json.loads((Path(row.folder)/"history.json").read_text())
                    histories.append((np.array([r["fe"] for r in values]),np.array([r[metric] for r in values])))
                if not histories:continue
                grid=np.unique(np.concatenate([h[0] for h in histories]))
                curves=[]
                for x,y in histories:
                    idx=np.searchsorted(x,grid,side="right")-1
                    curves.append(y[np.maximum(idx,0)])
                values=np.array(curves)
                center=np.median(values,axis=0)
                low,high=np.quantile(values,[.25,.75],axis=0)
                ax.plot(grid,center,color=color,lw=2.2,label=method)
                ax.fill_between(grid,low,high,color=color,alpha=.16,linewidth=0)
            ax.set(xlabel="Objective-vector evaluations",ylabel=title,title=problem)
            ax.set_ylim(bottom=0.)
            ax.grid(color="#D8D3C5",linewidth=.8,alpha=.7,zorder=0)
            ax.legend(frameon=False,fontsize=14)
            fig.tight_layout()
            fig.savefig(target/f"{problem}_{metric}.pdf",bbox_inches="tight")
            fig.savefig(target/f"{problem}_{metric}.png",dpi=320,bbox_inches="tight")
            plt.close(fig)
