#!/usr/bin/env python3
"""Render five frozen single-vector, same-layer C4 bridge jobs."""
from __future__ import annotations
import argparse,copy
from pathlib import Path
import yaml

SOURCE_JOB='cf1-visionpad-layer-transport-singles-n50-seed42'
SCRIPT_SHA='e0bb85e24e0f15d6811162e42ffc854d5864a022715d2f95865b628844319465'
TEST_SHA='67775bc7b2eb1bb8347980302f86e4305378dff98cbb3cb8327a2c9f2fafb4e5'
LAYERS=(32,40,48,56,63)

def load(path):
 v=yaml.safe_load(Path(path).read_text());
 if not isinstance(v,dict):raise ValueError('Invalid YAML')
 return v

def command(source):
 start=source.index('test -f "$$RUN_DIR/final_adapter/adapter_model.safetensors"');prefix=source[:start]
 return prefix+f'''test -f "$$RUN_DIR/final_adapter/adapter_model.safetensors"
test "$$(sha256sum scripts/evaluate_single_layer_state_bridge.py | cut -d' ' -f1)" = {SCRIPT_SHA}
test "$$(sha256sum tests/test_single_layer_state_bridge.py | cut -d' ' -f1)" = {TEST_SHA}
python -m unittest discover -s tests -p 'test_single_layer_state_bridge.py' -v
python -u scripts/evaluate_single_layer_state_bridge.py \\
  --model "$$LOCAL_MODEL" \\
  --adapter "$$RUN_DIR/final_adapter" \\
  --jsonl "$$C4_DATA/test_same_template.jsonl" \\
  --pair-manifest "$$PAIR_MANIFEST" \\
  --output-json "$$AMLT_OUTPUT_DIR/gate/single_layer_state_bridge.json" \\
  --layer "$$STATE_LAYER" \\
  --position-mode compact \\
  --state-token-alias '<|vision_pad|>' \\
  --max-length 1536 --num-pairs 150 --seed 42
printf '%s\n' 'diagnostic_only=true' 'optimizer_steps=0' "state_layer=$$STATE_LAYER" 'state_capacity=one_hidden_vector' > "$$AMLT_OUTPUT_DIR/gate/scope.txt"
'''

def render(base):
 source=next((j for j in base.get('jobs') or [] if j.get('name')==SOURCE_JOB),None)
 if source is None:raise ValueError('Missing source job')
 out=copy.deepcopy(base);out['description']='Zero-training frozen C4 sweep of one hidden-size vector captured and injected in the same native layer';out['jobs']=[];cmd=command(str(source['command']))
 for layer in LAYERS:
  job=copy.deepcopy(source);job['name']=f'cf1-single-vector-layer{layer}-c4-n150-seed42';job['command']=cmd;job['sku']='80G1-A100';job['mpi']=False;job['process_count_per_node']=1
  env=job.setdefault('submit_args',{}).setdefault('env',{});env.clear();env.update({'_AZUREML_SINGULARITY_JOB_UAI':'rankfun_aml','STATE_LAYER':str(layer)})
  out['jobs'].append(job)
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=render(load(a.base));a.output.write_text(yaml.safe_dump(v,sort_keys=False));print(f'rendered={a.output} jobs={len(v["jobs"])}')
if __name__=='__main__':main()
