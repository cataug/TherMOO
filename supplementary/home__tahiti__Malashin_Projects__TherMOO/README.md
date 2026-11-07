# TherMOO 1.0

Executable population-aware migration evaluation: 61 benchmark functions,
10 main comparison methods, 7 ablations, parameter sensitivity, homogeneous
islands, K=4/8/16, five objectives and D=50/100. The plan has 33,240 unique trials.
No scientific results are bundled.

## Run from the existing server environment

    /home/tahiti/Malashin_Projects/.venv_a100/bin/python scripts/install_thermoo.py
    /home/tahiti/Malashin_Projects/.venv_a100/bin/python -m thermoo prepare
    /home/tahiti/Malashin_Projects/.venv_a100/bin/python -m thermoo verify
    /home/tahiti/Malashin_Projects/.venv_a100/bin/python -m thermoo plan
    /home/tahiti/Malashin_Projects/.venv_a100/bin/python -m thermoo smoke --workers 4
    /home/tahiti/Malashin_Projects/.venv_a100/bin/python -m thermoo pilot --workers 4
    /home/tahiti/Malashin_Projects/.venv_a100/bin/python -m thermoo analyze --output pilot
    bash run_all.sh --workers 4

Run a timing pilot before the long full command. Independent trials use separate
processes with one numerical-library thread each. The GPU is not required.
Small missing helper dependencies are installed under vendor_deps using --no-deps.
The existing NumPy, SciPy and Torch installation is preserved. The downloaded
pymoo source is used directly; jMetalPy's large dependency set is not installed.

Useful subsets:

    python -m thermoo run --blocks main ablation --workers 4
    python -m thermoo run --blocks main --problems ZDT1 ZDT5 --seeds 0 1 --workers 2
    python -m thermoo run --blocks many --max-jobs 12 --workers 2
    python -m thermoo analyze --output full --plots

Completed trials are skipped on re-run. An interrupted trial restarts from its
seed; this is job-level restart, not generation-level continuation. An
implementation failure stops submission of new trials. Ctrl-C terminates active
workers and retains completed results.

## Scope of FULL

FULL is an explicit heuristic state-aware controller, not a trained neural
network. It corresponds to the heuristic version in the original Discussion.
Publishing these experiments requires describing the actual implemented
controller. A learned controller needs a separate leakage-controlled training
study. No neural training, weights, positive results or claims are invented.

Exact formulas and coefficients are in thermoo/control.py. Negative results are
retained. No test-result-based hyperparameter selection is performed.

## Matrix

| Block | Trials |
|---|---:|
| 61 functions x 10 methods x 30 seeds | 18,300 |
| 10-function panel x 7 ablations x 30 | 2,100 |
| 10-function panel x 8 one-factor changes x 30 | 2,400 |
| 10 functions x 4 homogeneous compositions x 3 policies x 30 | 3,600 |
| 10 functions x 2 additional K values x 3 policies x 30 | 1,800 |
| 6 five-objective problems x 12 methods x 30 | 2,160 |
| 6 problems x 2 larger dimensions x 8 methods x 30 | 2,880 |
| Total | 33,240 |

Families: ZDT1-6, DTLZ1-7, WFG1-9, LZ09_F1-F9, UF1-10, ZCAT1-20.
Panel: ZDT1, ZDT3, DTLZ2, DTLZ7, WFG1, WFG9, LZ09_F1, UF8, ZCAT14, ZCAT20.
Five-objective panel: DTLZ2, DTLZ7, WFG4, WFG9, ZCAT14, ZCAT20.
Dimension panel: ZDT4, DTLZ2, DTLZ7, WFG9, UF1, ZCAT20; D=50 and 100.

## Parameters and budgets

- 100,000 objective-vector evaluations, including initialization, asserted at
  the end of every trial. One vector evaluation is one FE, not M scalar FEs.
- Four islands of 100 individuals; standalone solvers have 400 individuals.
  The shared bounded nondominated output archive has capacity 400.
- All methods share the initial genotypes for a given problem/M/D/population/seed.
  Algorithm and migration RNGs are separate.
- SBX p=0.9, eta=20. Polynomial mutation applied to every offspring with
  per-variable probability 1/D, eta=20. Duplicate elimination disabled.
- MOEA/D: Tchebycheff, 20 neighbors (clipped for smoke tests), neighborhood mating
  probability 0.9. Exact-cardinality reference directions consist of axes and
  fixed-seed Sobol-derived simplex points.
- GDE3: native DE/rand/1/bin, F=0.5, CR=0.9, gamma=0.
- SMS-EMOA: steady state, one offspring per step; native exact hypervolume
  contribution survival with normalized reference offset epsilon=0.1.
- RVEA: alpha=2, adaptation frequency=0.1, horizon derived from the FE budget.
  Native APD survival may retain fewer occupied niches than the maximum
  population. Offspring count, FE budget and output archive capacity stay fixed.
- ZDT5: 80 bits, raw unnormalized objectives, GDE3 replaced by SPEA2 both
  standalone and on its island; one-point crossover p=.9, bit mutation 1/80.
- Coordination every 10*N_k offspring evaluations per island, excluding
  initialization. Never inject into an unfinished MOEA/D sweep.
- rho=.10, one source per recipient, bounded incoming count. Migrants retain
  cached objectives and are not re-evaluated.
- K=8/16 use the same four optimizer types equally, with 50/25 individuals per
  island; total population and FE budget remain fixed.

D=30 for ZDT1-3, LZ09 and UF; D=10 for ZDT4/6. DTLZ uses k=5 for DTLZ1,
k=20 for DTLZ7 and k=10 otherwise, D=M+k-1. WFG uses k=2*(M-1), l=20.
ZCAT: D=10*M, level=1, bias=0, complicated Pareto set=1, imbalance=0.
All ZDT, LZ09 except F6, and UF1-7 have two objectives; LZ09_F6, UF8-10 and
the remaining main families have three objectives.

## Descriptors and policies

q=-rank+0.49*score; the rank gap preserves rank priority. Default score is
normalized crowding; alternatives are normalized HV contribution or genotype
novelty. E=-q; T=c*std(q)+1e-8; weights use stable softmax.
H measures concentration of quality weights, not geometric diversity.
Genotype diversity is measured separately, with Hamming distance for ZDT5.
F=Ebar-T*H is explicitly a derived feature.

JS uses probability mass accumulated in shared fixed objective-coordinate bins,
not a correspondence between individual indices. The frame is initialized from
the shared initial population. Arctangent maps coordinates into a fixed grid.
The true evaluation reference front never enters this online coordinate frame.

FULL gates on HV progress, JS drift and geometric diversity, scores sources using
quality, complementarity and state descriptors, adapts incoming count and mixes
Boltzmann/novelty selection. See control.py for every coefficient and threshold.

Baselines: NO_MIG; periodic RANDOM ring exchange; periodic ELITE ring exchange;
HV_GATE using only relative HV progress for triggering; CROWDING using the
Märtens-Izzo rank/crowding selection and replacement operator. CROWDING is a
synchronous adaptation in this heterogeneous harness, not a reproduction of the
original asynchronous homogeneous system. MOEA/D survivors are reassociated to
weight directions; binary SPEA2 strength/density metadata are refreshed.
Other policies use the recipient's native environmental update.

Ablations: conventional HV/diversity state; no entropy block (also removes
F/deltaF); no F/deltaF alone; no JS; fixed count; random source; Boltzmann-only
migrant selection. Sensitivity changes one factor: interval=5/20, rho=.05/.20,
temperature multiplier=.5/2, quality=rank-HV/rank-novelty.

## Evaluation and statistical reports

Reference fronts are prepared once before optimization. Main fronts come from
the downloaded libraries and author's ZCAT generator. Additional five-objective
DTLZ2/WFG4/WFG9 fronts use the sphere/ellipsoid equations, fixed-seed spherical
samples and axis extremes. DTLZ7 uses nondominated intervals derived from
g(t)=t*(1+sin(3*pi*t)); their endpoints are solved numerically and boundary
combinations are included. References do not depend on solver outputs.

HV and IGD+ use common reference-set ideal/nadir normalization. HV reference is
(1.1,...,1.1); reported HV is divided by 1.1^M. IGD+ is mean reference-to-set
distance with dominated-coordinate clipping. Out-of-box values are not clipped.

Snapshots retain actual FE counts. A nominal 25,000 threshold may first be
observed at 25,200 with 400-offspring generations. Plots use actual counts and
previous-observation interpolation; they do not relabel the point as 25,000.

Reports include per-run CSV, aggregates, family summaries, timing estimates,
coverage, paired tests, Holm corrections, wins/ties/losses, mean ranks and
Friedman/post-hoc results. Inferential tests require 30 paired completed seeds
and run only for full results. Smoke and pilot results are diagnostics.

Primary paired sign-flip tests: 50,000 resamples, fixed seed. Both HV and IGD+
contrasts share one Holm family within each block. Mean-difference intervals use
4,000 paired bootstrap resamples. Shapiro diagnostics, conditional paired t-tests
and conditional signed-rank checks are sensitivity analyses.
Friedman uses case medians, not pooled seed observations. Related dimension
variants must not be presented as new independent problem families.
The binary ZDT5 supplement is excluded from the cross-problem Friedman table
because its DE slot is deliberately replaced by SPEA2. Per-run exports identify
the actual algorithms, and its standalone comparison is labeled SPEA2.
Mean-rank reports include an equal-family-weighted sensitivity summary.

Plots use cream backgrounds, viridis, black axes, large fonts and single panels.
Bands show IQR, not a mislabeled confidence interval.

## Execution interpretation

Independent trials run in parallel; islands within a trial use a deterministic
sequential schedule. K experiments establish search behavior under fixed budgets,
not physical parallel speedup. Payload bytes count genotype/objective arrays,
not measured MPI traffic, packet overhead or controller-profile broadcasts.
Do not infer measured network savings from these logical byte counts.
Control, integration and metric-diagnostic time are reported separately.

Trial identity includes optimization code, core dependency versions, native
benchmark source hashes and the actual reference bytes. Replacing these inputs
does not silently reuse old scientific results.

## Third-party sources and licenses

TherMOO original code is GPL-3.0-or-later; see LICENSE. The bundled PaGMO source
retains its copyright notice and dual LGPL/GPL licensing; see vendor license
texts. ZCAT is linked from the author's already downloaded GPL source.
Existing third-party repositories retain their own licenses.

- pymoo: https://github.com/anyoptimization/pymoo
- jMetalPy, LZ09 adapter: https://github.com/jMetal/jMetalPy
- ZCAT: https://github.com/evo-mx/ZCAT
- PaGMO UF: https://github.com/esa/pagmo2/blob/master/src/problems/cec2009.cpp
- Märtens and Izzo 2013:
  https://www.esa.int/gsp/ACT/doc/AI/pub/ACT-RPR-AI-2013-nsgaii_island_model.pdf
- Osaba et al. 2021: DOI 10.1016/j.swevo.2021.100888.
