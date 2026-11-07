"""Explicit heuristic controller: no untrained network or surrogate training labels."""
from dataclasses import dataclass
import math
import numpy as np
from scipy.special import softmax
from scipy.spatial.distance import cdist
from moocore import hypervolume, hv_contributions, pareto_rank


def crowding(F):
    n, m = F.shape
    result = np.zeros(n)
    if n < 3:
        return np.ones(n)
    for j in range(m):
        order = np.argsort(F[:, j], kind="stable")
        span = F[order[-1], j]-F[order[0], j]
        if span <= 1e-14:
            continue
        result[order[0]] = result[order[-1]] = np.inf
        result[order[1:-1]] += (F[order[2:], j]-F[order[:-2], j])/span
    finite = result[np.isfinite(result)]
    cap = max(float(finite.max()) if len(finite) else 0., 1.)
    result = np.where(np.isfinite(result), result, cap)
    return result / max(float(result.max()), 1e-14)


def js(p, q):
    p, q = np.asarray(p), np.asarray(q)
    mid = .5*(p+q)
    a = np.sum(p*np.log(np.divide(p, mid, out=np.ones_like(p), where=p>0)), where=p>0)
    b = np.sum(q*np.log(np.divide(q, mid, out=np.ones_like(q), where=q>0)), where=q>0)
    return float(.5*(a+b))


def histogram(F, weights, bins=6):
    # The caller uses a fixed shared coordinate frame for every island and time.
    u = (np.arctan(F)+np.pi/2)/np.pi
    cells = np.minimum((u*bins).astype(int), bins-1)
    code = np.ravel_multi_index(cells.T, (bins,)*F.shape[1])
    hist = np.bincount(code, weights=weights, minlength=bins**F.shape[1])
    return hist/hist.sum()


@dataclass
class Profile:
    H: float
    E: float
    free: float
    temperature: float
    hv: float
    progress: float
    drift: float
    diversity: float
    nd_fraction: float
    histogram: np.ndarray
    weights: np.ndarray
    novelty: np.ndarray
    ranks: np.ndarray
    distances: np.ndarray
    dH: float = 0.
    dF: float = 0.
    dDiv: float = 0.

    def record(self):
        return {k: float(getattr(self, k)) for k in ("H", "E", "free", "temperature", "hv", "progress", "drift", "diversity", "nd_fraction", "dH", "dF", "dDiv")}


def profile(pop, problem, lower, scale, job, previous=None):
    F = (pop.get("F")-lower)/scale
    X = (pop.get("X")-problem.xl)/(problem.xu-problem.xl)
    n = len(F)
    ranks = np.asarray(pareto_rank(F))
    cd = crowding(F)
    distances = cdist(X, X, metric="hamming" if problem.binary else "euclidean")
    if not problem.binary:
        distances /= np.sqrt(problem.n_var)
    diversity = float(distances.sum()/max(n*(n-1), 1))
    np.fill_diagonal(distances, np.inf)
    nearest = np.partition(distances, min(4, n-2), axis=1)[:, :min(5, n-1)]
    novelty = nearest.mean(axis=1) if n > 1 else np.ones(n)
    novelty = novelty/(novelty.sum()+1e-14)
    if job.quality == "rank_hv":
        metric = hv_contributions(F, ref=np.full(problem.n_obj, 1.1))
        metric = metric/max(float(metric.max()), 1e-14)
    elif job.quality == "rank_novelty":
        metric = novelty/max(float(novelty.max()), 1e-14)
    else:
        metric = cd
    q = -ranks.astype(float) + .49*metric
    T = float(job.temperature*np.std(q)+1e-8)
    weights = softmax(q/T)
    H = float(-np.sum(weights*np.log(weights+1e-300)))
    E = float(np.sum(weights*(-q)))
    free = E-T*H
    # A finite fixed online reference point is independent of the true Pareto front.
    hv = float(hypervolume(F, ref=np.full(problem.n_obj, 1.1)))
    bins = 4 if problem.n_obj > 3 else 6
    hist_weights = np.full(n, 1/n) if job.ablation == "conventional" else weights
    hist = histogram(F, hist_weights, bins)
    progress = 0. if previous is None else (hv-previous.hv)/(1+abs(previous.hv))
    drift = 0. if previous is None else js(hist, previous.histogram)
    return Profile(H/max(math.log(n), 1.), E, free, T, hv, progress, drift,
                   diversity, float(np.mean(ranks == 0)), hist, weights, novelty, ranks,
                   np.linalg.norm(F, axis=1),
                   0. if previous is None else H/max(math.log(n), 1.)-previous.H,
                   0. if previous is None else free-previous.free,
                   0. if previous is None else diversity-previous.diversity)


def select_actions(profiles, job, rng):
    """Return one source per target, bounded rate, and exploitation mixture."""
    k = len(profiles)
    if job.method == "NO_MIG":
        return []
    fixed = job.method in ("RANDOM", "ELITE", "CROWDING")
    hvs = np.array([p.hv for p in profiles])
    f = np.array([p.free for p in profiles])
    f_score = (f.max()-f)/(np.ptp(f)+1e-12)
    quality = (hvs-hvs.min())/(np.ptp(hvs)+1e-12)
    actions = []
    for target, p in enumerate(profiles):
        sources = [i for i in range(k) if i != target]
        if fixed:
            actions.append(((target-1) % k, target, job.rho, 1., 1.))
            continue
        if p.progress >= job.hv_epsilon:
            continue
        if job.method == "HV_GATE":
            source = max(sources, key=lambda i: profiles[i].hv)
            actions.append((source, target, job.rho, 1., float(quality[source])))
            continue
        conventional = job.ablation == "conventional"
        no_entropy = conventional or job.ablation == "no_entropy"
        no_free = no_entropy or job.ablation == "no_free_energy"
        no_js = job.ablation == "no_js"
        # JS and geometric diversity jointly discriminate stalled states.
        if not no_js and p.drift >= job.js_epsilon and p.diversity >= np.median([s.diversity for s in profiles]):
            continue
        complement = np.array([math.sqrt(max(js(profiles[s].histogram, p.histogram), 0.)) for s in sources])
        if no_js:
            complement[:] = 0.
        scores = []
        for i, s in enumerate(sources):
            a = profiles[s]
            score = .50*quality[s] + .20*a.diversity + .20*complement[i]
            if conventional:
                score += .10*a.nd_fraction + .10*max(a.dDiv, 0.)
            else:
                if not no_entropy:
                    score += .10*a.H + .05*max(a.dH, 0.)
                if not no_free:
                    score += .10*f_score[s] + .05*max(-a.dF, 0.)/(1+abs(a.free))
            scores.append(score)
        source = int(rng.choice(sources)) if job.ablation == "random_source" else sources[int(np.argmax(scores))]
        need = .5*(1-p.diversity)
        if not no_entropy:
            need += .3*(1-p.H)
        if not no_js:
            need += .2*(1-min(p.drift/max(job.js_epsilon, 1e-14), 1.))
        ratio = job.rho if job.ablation == "fixed_rate" else job.rho*np.clip(.25+.75*need, .25, 1.)
        mixing = 1. if job.ablation == "boltzmann_only" else float(np.clip(.25+.5*quality[source], .25, .75))
        actions.append((source, target, float(ratio), mixing, float(max(scores))))
    return actions


def choose_migrants(pop, p, count, job, mixing, rng):
    count = min(count, len(pop))
    if job.method == "RANDOM":
        indices = rng.choice(len(pop), count, replace=False)
    elif job.method in ("ELITE", "HV_GATE"):
        indices = np.lexsort((p.distances, p.ranks))[:count]
    elif job.method == "CROWDING":
        indices = np.lexsort((-crowding(pop.get("F")), p.ranks))[:count]
    else:
        weights = mixing*p.weights+(1-mixing)*p.novelty + 1e-14
        weights /= weights.sum()
        indices = rng.choice(len(pop), count, replace=False, p=weights)
    return pop[indices]
