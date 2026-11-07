"""Native pymoo algorithms; explicit variation and migration integration."""
from .runtime import stable_seed
from .problems import directions
import copy
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from pymoo.core.population import Population
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.pntx import SinglePointCrossover
from pymoo.operators.mutation.pm import PM
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.decomposition.tchebicheff import Tchebicheff
from pymoo.config import Config
Config.warnings["not_compiled"] = False


def create(kind, problem, X, seed, local_budget):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.moead import MOEAD
    from pymoo.algorithms.moo.gde3 import GDE3
    from pymoo.algorithms.moo.sms import SMSEMOA, LeastHypervolumeContributionSurvival
    from pymoo.algorithms.moo.spea2 import SPEA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.algorithms.moo.rvea import RVEA
    size = len(X)
    if problem.binary and kind == "GDE3":
        kind = "SPEA2"
    crossover = SinglePointCrossover(prob=.9) if problem.binary else SBX(prob=.9, eta=20)
    mutation = BitflipMutation(prob=1.0, prob_var=1/problem.n_var) if problem.binary else PM(prob=1.0, prob_var=1/problem.n_var, eta=20)
    args = dict(sampling=X.copy(), eliminate_duplicates=False, crossover=crossover, mutation=mutation)
    if kind == "MOEAD":
        args.pop("eliminate_duplicates")
        alg = MOEAD(ref_dirs=directions(problem.n_obj, size), n_neighbors=min(20, size), decomposition=Tchebicheff(), prob_neighbor_mating=.9, **args)
    elif kind == "GDE3":
        alg = GDE3(pop_size=size, sampling=X.copy(), F=.5, CR=.9, gamma=0.)
    elif kind == "SMS":
        alg = SMSEMOA(pop_size=size, n_offsprings=1, survival=LeastHypervolumeContributionSurvival(eps=.1), **args)
    elif kind == "NSGA2":
        alg = NSGA2(pop_size=size, **args)
    elif kind == "SPEA2":
        alg = SPEA2(pop_size=size, **args)
    elif kind == "NSGA3":
        alg = NSGA3(ref_dirs=directions(problem.n_obj, size), pop_size=size, **args)
    elif kind == "RVEA":
        alg = RVEA(ref_dirs=directions(problem.n_obj, size), pop_size=size, alpha=2., adapt_freq=.1, **args)
    else:
        raise ValueError(kind)
    # RVEA needs a fixed generation horizon; every generation evaluates `size` offspring.
    termination = ("n_gen", max(2, local_budget//size)) if kind == "RVEA" else ("n_eval", local_budget+size)
    alg.setup(problem, seed=seed, termination=termination, verbose=False)
    alg.next()
    alg.thermoo_kind = kind
    return alg


def evolve(alg, evaluations):
    before = alg.evaluator.n_eval
    target = before + evaluations
    while alg.evaluator.n_eval < target:
        alg.next()
        if alg.evaluator.n_eval > target:
            raise RuntimeError(f"Offspring batch exceeds exact local budget: {alg.thermoo_kind}")
    # Migration is allowed only when MOEA/D's generator has finished its sweep.
    if alg.thermoo_kind == "MOEAD" and alg.generator is not None:
        raise RuntimeError("MOEA/D coordination must coincide with a complete sweep")


def inject(alg, migrants, problem, crowding_replacement=False):
    migrants = copy.deepcopy(migrants)
    for item in migrants:
        item.evaluated.update({"F", "G", "H"})
    before = alg.pop.get("X").copy()
    rng = alg.random_state
    if alg.thermoo_kind == "MOEAD":
        alg.ideal = np.minimum(alg.ideal, migrants.get("F").min(axis=0))
        if crowding_replacement:
            pop = RankAndCrowding().do(problem, Population.merge(alg.pop, migrants), n_survive=alg.pop_size, random_state=rng)
            f = np.maximum(pop.get("F")-alg.ideal, 0.)
            cost = cdist(f/(np.linalg.norm(f, axis=1, keepdims=True)+1e-14), alg.ref_dirs/(np.linalg.norm(alg.ref_dirs, axis=1, keepdims=True)+1e-14))
            rows, cols = linear_sum_assignment(cost)
            order = np.empty(alg.pop_size, int)
            order[cols] = rows
            alg.pop = pop[order]
        else:
            for item in migrants:
                f = np.maximum(item.F-alg.ideal, 0.)
                k = np.argmin(cdist(f[None]/(np.linalg.norm(f)+1e-14), alg.ref_dirs/(np.linalg.norm(alg.ref_dirs, axis=1, keepdims=True)+1e-14)))
                alg._replace(int(k), item)
    else:
        merged = Population.merge(alg.pop, migrants)
        survival = RankAndCrowding() if crowding_replacement else alg.survival
        alg.pop = survival.do(problem, merged, n_survive=alg.pop_size, random_state=rng, algorithm=alg)
        if crowding_replacement and alg.thermoo_kind == "SPEA2":
            # SPEA2's tournament requires strength/density fitness on every survivor.
            # Refresh those attributes on the already selected set, without changing
            # the crowding-based migration replacement or evaluating objectives.
            alg.pop = alg.survival.do(problem, alg.pop, n_survive=len(alg.pop), random_state=rng, algorithm=alg)
    alg._set_optimum()
    # Count accepted distinct migrant genotypes, not the number of changed population slots.
    now = {row.tobytes() for row in alg.pop.get("X")}
    old = {row.tobytes() for row in before}
    return len({row.tobytes() for row in migrants.get("X")} & (now-old))
