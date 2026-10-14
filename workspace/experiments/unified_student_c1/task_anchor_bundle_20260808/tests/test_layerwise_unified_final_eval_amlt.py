from __future__ import annotations
import copy,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from validate_layerwise_unified_final_eval_amlt import load,validate
class FinalEvalConfigTest(unittest.TestCase):
    def setUp(self):self.config=load(ROOT/'amlt/feeds_layerwise_unified_final_eval_20260731.yaml')
    def test_passes(self):validate(self.config)
    def test_frozen_training_runtime_is_required(self):
        x=copy.deepcopy(self.config);x['jobs'][0]['command']=x['jobs'][0]['command'].replace('scripts/frozen_layerwise_finalfit/layerwise_recurrent_student_model.py','scripts/layerwise_recurrent_student_model.py')
        with self.assertRaisesRegex(ValueError,'command drift'):validate(x)
    def test_scalar_is_rejected(self):
        x=copy.deepcopy(self.config);x['jobs'][0]['command']+='\n--state-architecture scalar\n'
        with self.assertRaisesRegex(ValueError,'scalar'):validate(x)
    def test_gpu_drift_rejected(self):
        x=copy.deepcopy(self.config);x['jobs'][0]['sku']='80G4-A100'
        with self.assertRaisesRegex(ValueError,'GPU'):validate(x)
    def test_ephemeral_disk_gate_is_required_for_every_job(self):
        x=copy.deepcopy(self.config);x['jobs'][0]['command']=x['jobs'][0]['command'].replace('test "$$(df --output=avail /tmp | tail -1)" -ge 83886080\n','')
        with self.assertRaisesRegex(ValueError,'disk gate'):validate(x)
    def test_selected_recovery_subset_passes(self):
        jobs={'layerwise-final-eval-t3-seed42','layerwise-final-eval-c4-seed42'}
        x=copy.deepcopy(self.config);x['jobs']=[j for j in x['jobs'] if j['name'] in jobs]
        validate(x,jobs)
    def test_single_formal_recovery_passes(self):
        jobs={'layerwise-final-eval-t2-seed42'}
        x=copy.deepcopy(self.config);x['jobs']=[j for j in x['jobs'] if j['name'] in jobs]
        validate(x,jobs)
    def test_single_c4_recovery_passes(self):
        jobs={'layerwise-final-eval-c4-seed42'}
        x=copy.deepcopy(self.config);x['jobs']=[j for j in x['jobs'] if j['name'] in jobs]
        validate(x,jobs)
    def test_all_c4_pair_seed_jobs_are_required_and_pass(self):
        jobs={f'layerwise-final-eval-c4-seed{seed}' for seed in (13,42,97)}
        x=copy.deepcopy(self.config);x['jobs']=[j for j in x['jobs'] if j['name'] in jobs]
        validate(x,jobs)
    def test_c4_pair_seed_env_drift_is_rejected(self):
        x=copy.deepcopy(self.config)
        job=next(j for j in x['jobs'] if j['name']=='layerwise-final-eval-c4-seed13')
        job['submit_args']['env']['C4_PAIR_SEED']='42'
        with self.assertRaisesRegex(ValueError,'pair-seed env drift'):validate(x)
    def test_unregistered_recovery_job_is_rejected(self):
        x=copy.deepcopy(self.config);x['jobs']=x['jobs'][:1]
        with self.assertRaisesRegex(ValueError,'Expected job drift'):validate(x,{'invented-job'})
if __name__=='__main__':unittest.main()
