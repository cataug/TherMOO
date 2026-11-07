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
