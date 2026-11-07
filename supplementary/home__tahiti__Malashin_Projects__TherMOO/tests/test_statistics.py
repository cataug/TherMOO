"""Synthetic test data below are software fixtures, never paper experiment results."""
import tempfile
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch
import json
import numpy as np
import pandas as pd
from thermoo import analysis
from thermoo.plan import Job


class StatisticsTests(TestCase):
    def test_holm_known_values(self):
        np.testing.assert_allclose(analysis.holm([.01,.04,.03]),[.03,.06,.06])

    def test_paired_zero_and_shift(self):
        x=np.arange(30.)
        zero=analysis.paired_stats(x,x,42,resamples=2000)
        self.assertEqual(zero['p_primary'],1.)
        self.assertEqual(zero['difference'],0.)
        shift=analysis.paired_stats(x+1,x,42,resamples=2000)
        self.assertLess(shift['p_primary'],.005)
        self.assertEqual(shift['difference'],1.)
        self.assertEqual(shift['rank_biserial'],1.)
        self.assertEqual(shift['ci_low'],1.)

    def test_report_pairing_and_complete_cases(self):
        pairs=[]
        for problem in ['ZDT1','DTLZ2','WFG9']:
            for seed in range(30):
                for method,gain in [('NSGA2',0.),('MOEAD',.03),('FULL',.1)]:
                    job=Job(problem,method,seed)
                    hv=.5+gain+.0001*seed
                    row=dict(run_id=job.signature(),block='main',problem=problem,family=analysis.family(problem),m=job.objectives,d=job.variables,
                             seed=seed,label=method,case=analysis.case_key(job),method=method,composition='mixed',k=4,interval=10,rho=.1,temperature=1.,
                             quality='rank_crowding',ablation='none',budget=100000,hv=hv,igd_plus=1-hv,seconds=1.,sent=0,accepted=0,payload_bytes=0,
                             control_seconds=0.,integration_seconds=0.,diagnostic_seconds=0.,initial_hash=f'{problem}-{seed}',folder='unused')
                    pairs.append((row,job))
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(analysis,'ROOT',Path(directory)),patch.object(analysis,'read_results',return_value=pairs),patch.object(analysis,'experiment_fingerprint',return_value='synthetic_fixture'):
                analysis.analyze('full')
            folder=Path(directory)/'reports/full/synthetic_fixture'
            comparisons=pd.read_csv(folder/'paired_tests.csv')
            self.assertEqual(len(comparisons),12)
            self.assertTrue((comparisons.n==30).all())
            self.assertTrue((comparisons.difference>0).all())
            self.assertTrue((comparisons.outcome=='FULL_wins').all())
            ranks=pd.read_csv(folder/'mean_ranks.csv')
            self.assertTrue((ranks[ranks.label=='FULL'].mean_rank==1.).all())
            coverage=json.loads((folder/'coverage.json').read_text())
            self.assertFalse(coverage['complete'])
            self.assertEqual(coverage['completed'],270)


if __name__=='__main__':
    main()
