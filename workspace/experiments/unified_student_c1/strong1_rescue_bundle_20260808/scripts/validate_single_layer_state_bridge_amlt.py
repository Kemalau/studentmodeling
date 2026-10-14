#!/usr/bin/env python3
from __future__ import annotations
import argparse,re
from pathlib import Path
import yaml
from amlt_validation_utils import command_text,target_matches,validate_shell_syntax
from render_single_layer_state_bridge_c4 import LAYERS
IMAGE='amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258';WORKSPACE='CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity'
def load(path):
 text=Path(path).read_text();starts=[i for m in ('description:','target:') if (i:=text.find(m))>=0];
 if not starts:raise ValueError('No YAML')
 v=yaml.safe_load(text[min(starts):]);
 if not isinstance(v,dict):raise ValueError('Invalid YAML')
 return v
def validate(v):
 text=yaml.safe_dump(v)
 if re.search(r'<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b',text):raise ValueError('Placeholder')
 if not target_matches(v.get('target'),WORKSPACE) or (v.get('environment') or {}).get('image')!=IMAGE:raise ValueError('Target/image drift')
 expected={f'cf1-single-vector-layer{x}-c4-n150-seed42':str(x) for x in LAYERS};jobs=v.get('jobs') or []
 if {j.get('name') for j in jobs}!=set(expected):raise ValueError('Job drift')
 for j in jobs:
  name=j['name'];env=(j.get('submit_args') or {}).get('env') or {};cmd=command_text(j.get('command'));validate_shell_syntax(cmd,name)
  if j.get('sku')!='80G1-A100' or j.get('sla_tier')!='Standard' or j.get('mpi') is not False or j.get('process_count_per_node')!=1:raise ValueError('GPU/SLA drift')
  if env.get('_AZUREML_SINGULARITY_JOB_UAI')!='rankfun_aml' or env.get('STATE_LAYER')!=expected[name] or (j.get('submit_args') or {}).get('max_attempts')!=2:raise ValueError('UAI/layer drift')
  required=('evaluate_single_layer_state_bridge.py','--num-pairs 150','--layer','state_capacity=one_hidden_vector','c4_stage_a_gate_pairs_seed42_n150_v1.json','/mnt/amulet/studentmodeling/models/Qwen3-32B')
  forbidden=('train_c1_lora.py','train_recurrent_student','password','account_key','access_token','/home/v-yuejiali/studentmodeling/models','msrcommonacr.azurecr.io/ml-common-py3.12:latest')
  if any(x not in cmd for x in required) or any(x in cmd.lower() for x in forbidden):raise ValueError('Command drift')
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args();validate(load(a.config));print(f'validated={a.config}')
if __name__=='__main__':main()
