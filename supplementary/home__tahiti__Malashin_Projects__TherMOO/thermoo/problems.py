"""Benchmark adapters and evaluation-only reference data."""
from .runtime import ROOT, stable_seed
from .plan import Job
from . import native
import importlib
import json
import sys
import types

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtri
from scipy.stats import qmc
from pymoo.core.problem import Problem
from pymoo.problems import get_problem


def _lz_module():
    # Load only the benchmark/core modules, not jMetalPy's eager framework imports.
    base = ROOT / "sources/jMetalPy/src/jmetal"
    if not base.exists():
        base = ROOT / "sources/jMetalPy/jmetal"
    for suffix in ("", ".core", ".util", ".problem", ".problem.multiobjective"):
        name = "jmetal" + suffix
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(base.joinpath(*suffix.strip(".").split("."))) if suffix else str(base)]
            sys.modules[name] = module
    return importlib.import_module("jmetal.problem.multiobjective.lz09")


class Benchmark(Problem):
    def __init__(self, job):
        self.name, self.job = job.problem, job
        m, d = job.objectives, job.variables
        self.binary = job.problem == "ZDT5"
        self.counter = 0
        self.raw = None
        self.lz = None
        if self.name.startswith("UF"):
            p = int(self.name[2:])
            lb, ub = np.full(d, -1.0), np.ones(d)
            if p == 3:
                lb[:] = 0
            if p == 4 or p >= 8:
                lb[:], ub[:] = -2, 2
            lb[0], ub[0] = 0, 1
            if p >= 8:
                lb[1], ub[1] = 0, 1
        elif self.name.startswith("ZCAT"):
            lb = -0.5 * np.arange(1, d + 1)
            ub = -lb
        elif self.name.startswith("LZ09"):
            self.lz = getattr(_lz_module(), self.name)(number_of_variables=d)
            lb, ub = self.lz.lower_bound, self.lz.upper_bound
            from jmetal.core.solution import FloatSolution
            self.solution = FloatSolution(lb, ub, m)
        else:
            args = dict(n_var=d)
            if self.name.startswith(("DTLZ", "WFG")):
                args["n_obj"] = m
            if self.name.startswith("WFG"):
                args["k"] = 2 * (m - 1)
            if self.binary:
                args = dict(normalize=False)
            self.raw = get_problem(self.name.lower(), **args)
            lb, ub = self.raw.xl, self.raw.xu
        super().__init__(n_var=d, n_obj=m, xl=np.broadcast_to(lb, d).copy(), xu=np.broadcast_to(ub, d).copy(), vtype=bool if self.binary else float)

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.atleast_2d(X)
        if self.name.startswith("UF"):
            F = native.uf(int(self.name[2:]), X)
        elif self.name.startswith("ZCAT"):
            F = native.zcat(int(self.name[4:]), X, self.n_obj)
        elif self.lz is not None:
            F = []
            for x in X:
                self.solution.variables = x.tolist()
                self.lz.evaluate(self.solution)
                F.append(list(self.solution.objectives))
            F = np.asarray(F)
        else:
            F = self.raw.evaluate(X)
        if F.shape != (len(X), self.n_obj) or not np.isfinite(F).all():
            raise ArithmeticError(f"Invalid {self.name} objective values")
        self.counter += len(X)
        out["F"] = F


def initial_population(problem, size, seed):
    rng = np.random.default_rng(stable_seed(problem.name, problem.n_obj, problem.n_var, seed, "initial"))
    if problem.binary:
        return rng.integers(0, 2, (size, problem.n_var)).astype(bool)
    return problem.xl + rng.random((size, problem.n_var)) * (problem.xu - problem.xl)


def directions(m, size, seed=20261104):
    if size < m:
        raise ValueError("Not enough reference directions")
    if m == 2:
        x = np.linspace(0, 1, size)
        return np.column_stack([x, 1-x])
    sampler = qmc.Sobol(m, scramble=True, seed=seed)
    u = sampler.random_base2(int(np.ceil(np.log2(max(size-m, 1)))))[:size-m]
    x = -np.log(np.clip(u, 1e-12, 1-1e-12))
    return np.vstack([np.eye(m), x / x.sum(axis=1, keepdims=True)])


def _analytical_reference(name, m, n=20000):
    sampler = qmc.Sobol(m, scramble=True, seed=20261104)
    u = sampler.random_base2(int(np.ceil(np.log2(n))))[:n]
    if name == "DTLZ2" or name in ("WFG4", "WFG9"):
        x = np.abs(ndtri(np.clip(u, 1e-12, 1-1e-12)))
        pf = np.vstack([x / np.linalg.norm(x, axis=1, keepdims=True), np.eye(m)])
        return pf if name == "DTLZ2" else pf * (2*np.arange(1, m+1))
    if name == "DTLZ7":
        g = lambda x: x * (1 + np.sin(3*np.pi*x))
        dg = lambda x: 1 + np.sin(3*np.pi*x) + 3*np.pi*x*np.cos(3*np.pi*x)
        a, c = brentq(dg, .2, .3), brentq(dg, .8, .9)
        b = brentq(lambda x: g(x)-g(a), .6, .7)
        width = a + c-b
        y = u[:, :m-1] * width
        y = np.where(y <= a, y, b+y-a)
        import itertools
        y = np.vstack([y, np.asarray(list(itertools.product([0., a, b, c], repeat=m-1)))])
        return np.column_stack([y, 2*m-np.sum(g(y), axis=1)])
    raise ValueError((name, m))


def load_reference(job):
    folder = ROOT / "data/thermoo_references"
    path = folder / f"{job.problem}.{job.objectives}D.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path}; run 'python -m thermoo prepare' first")
    return np.load(path)


def prepare_references():
    from .plan import jobs
    source = json.loads((ROOT / "configs/reference_fronts_main.json").read_text())
    folder = ROOT / "data/thermoo_references"
    folder.mkdir(parents=True, exist_ok=True)
    seen = set()
    for job in jobs(seeds=[0]):
        key = (job.problem, job.objectives)
        if key in seen:
            continue
        seen.add(key)
        if job.objectives == 5:
            if job.problem.startswith("ZCAT"):
                pf = np.loadtxt(ROOT / f"data/reference_fronts/generated/{job.problem}.5D.pf")
            else:
                pf = _analytical_reference(job.problem, 5)
        else:
            entry = source[job.problem]
            file = ROOT / entry["path"]
            pf = np.loadtxt(file, delimiter="," if file.suffix == ".csv" else None)
        pf = np.unique(np.atleast_2d(pf), axis=0)
        if pf.shape[1] != job.objectives or not np.isfinite(pf).all():
            raise ValueError(f"Invalid reference front {key}")
        # No algorithm output is used to construct these shared evaluation sets.
        from moocore import is_nondominated
        pf = pf[is_nondominated(pf)]
        np.save(folder / f"{job.problem}.{job.objectives}D.npy", pf)
    print(f"[REFERENCES] {len(seen)} problem/objective combinations", flush=True)
