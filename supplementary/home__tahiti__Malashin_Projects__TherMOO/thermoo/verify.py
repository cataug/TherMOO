"""Independent benchmark checks and behavioral invariants before long trials."""
from .runtime import ROOT
from .plan import Job, jobs, PROBLEMS, BLOCKS
from .problems import Benchmark, initial_population, load_reference
from .control import histogram, js
from . import native
from dataclasses import replace
from collections import Counter
import json
import numpy as np
from moocore import hypervolume


def verify():
    results = []
    matrix = list(jobs())
    counts = dict(Counter(j.block for j in matrix))
    expected = dict(main=18300, ablation=2100, sensitivity=2400, homogeneous=3600, islands=1800, many=2160, dimensions=2880)
    assert counts == expected, counts
    assert len({j.signature() for j in matrix}) == 33240
    print("[PASS] Plan: 33,240 unique trials", flush=True)
    # Analytic staircase HV and permutation-invariant probability cells.
    assert abs(hypervolume(np.array([[.2,.8],[.5,.4]]), ref=np.ones(2))-.36) < 1e-12
    F = np.array([[0., 1.], [.5,.5], [1.,0.]])
    w = np.array([.1,.3,.6])
    assert np.allclose(histogram(F,w), histogram(F[::-1],w[::-1]))
    assert abs(js(histogram(F,w),histogram(F,w))) < 1e-12
    print("[PASS] Hypervolume and JS alignment invariants", flush=True)
    for name in PROBLEMS:
        job = Job(name,"FULL",0)
        problem = Benchmark(job)
        X = initial_population(problem, 8, 9182)
        Y = problem.evaluate(X)
        assert Y.shape == (8,job.objectives) and np.isfinite(Y).all()
        assert np.allclose(Y, problem.evaluate(X), rtol=1e-12, atol=1e-12)
        pf = load_reference(job)
        assert pf.shape[1] == job.objectives
        results.append(dict(problem=name,m=job.objectives,d=job.variables,reference_points=len(pf)))
    print("[PASS] 61 deterministic benchmark evaluators and reference fronts", flush=True)
    # This perturbation detects an off-by-one omission of decision variable x_2.
    n, t, eps = 30, .37, .1
    j = np.arange(2,n+1)
    x = np.r_[t,np.sin(6*np.pi*t+j*np.pi/n)]
    y = native.uf(1,x[None])[0]
    assert np.allclose(y,[t,1-np.sqrt(t)],atol=1e-12)
    x[1] += eps
    y = native.uf(1,x[None])[0]
    assert np.allclose(y,[t,1-np.sqrt(t)+2*eps**2/15],atol=1e-12)
    # Every UF function is checked on its analytically known Pareto set.
    for p in range(1,11):
        t = .35 if p == 6 else .3 if p == 5 else .1 if p == 9 else .37
        if p < 8:
            j = np.arange(2,n+1)
            values = np.sin(6*np.pi*t+j*np.pi/n)
            if p == 2:
                amplitude=.3*t*(t*np.cos(24*np.pi*t+4*j*np.pi/n)+2)
                values=amplitude*np.where(j%2==0,np.sin(6*np.pi*t+j*np.pi/n),np.cos(6*np.pi*t+j*np.pi/n))
            if p == 3:
                values=t**(.5*(1+3*(j-2)/(n-2)))
            x=np.r_[t,values]
            expected=[t,1-np.sqrt(t)] if p<=3 else [t,1-t*t] if p==4 else [t,1-t] if p<=6 else [t**.2,1-t**.2]
        else:
            s=.63
            j=np.arange(3,n+1)
            x=np.r_[t,s,2*s*np.sin(2*np.pi*t+j*np.pi/n)]
            expected=[t*s,(1-t)*s,1-s] if p==9 else [np.cos(np.pi*t/2)*np.cos(np.pi*s/2),np.cos(np.pi*t/2)*np.sin(np.pi*s/2),np.sin(np.pi*t/2)]
        assert np.allclose(native.uf(p,x[None])[0],expected,atol=1e-10), (p,native.uf(p,x[None])[0],expected)
    print("[PASS] UF1-UF10 analytical Pareto solutions, including x_2 perturbation", flush=True)
    # Compare the native author's ZCAT evaluator with an independent Python port.
    from pymoo.problems.many.zcat import ZCAT
    for m in (2,3,5):
        for p in range(1,21):
            job=Job(f"ZCAT{p}","FULL",0,m=m)
            problem=Benchmark(job)
            X=initial_population(problem,4,812)
            independent=ZCAT(p,n_var=job.variables,n_obj=m,complicated_pareto_set=True,level=1,bias=False,imbalance=False)
            assert np.allclose(problem.evaluate(X), independent.evaluate(X),rtol=1e-9,atol=1e-9), (p,m)
    print("[PASS] Native/Python ZCAT cross-check: 20 problems x 3 objective counts", flush=True)
    # References needed by the five-objective block are generated before any search.
    for name in ("DTLZ2","DTLZ7","WFG4","WFG9","ZCAT14","ZCAT20"):
        pf=load_reference(Job(name,"FULL",0,m=5))
        assert pf.shape[1] == 5 and len(pf)>100
    report=ROOT/"reports/verification.json"
    report.write_text(json.dumps(dict(passed=True,counts=counts,benchmarks=results),indent=2))
    print(f"[PASS] Verification report: {report}",flush=True)


def verify_trial_result(folder):
    """Saved genotypes must reproduce their objective vectors (outside trial accounting)."""
    data=json.loads((folder/"result.json").read_text())
    job=Job(**data["job"])
    pop=np.load(folder/"population.npz")
    problem=Benchmark(job)
    assert np.allclose(problem.evaluate(pop["X"]),pop["F"],rtol=1e-9,atol=1e-9)
    assert np.allclose(problem.evaluate(pop["population_X"]),pop["population_F"],rtol=1e-9,atol=1e-9)
    assert data["evaluations"]==job.budget
