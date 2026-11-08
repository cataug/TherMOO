# Compact figure protocol V9

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
