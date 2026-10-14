#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path

def materialize(template: Path, receipt: Path, output: Path, launcher: Path) -> dict:
    value=json.loads(receipt.read_text()); artifacts=value.get('sha256') or {}
    if value.get('status')!='complete' or value.get('state_shape')!=[64,8,5120] or set(artifacts)!={'adapter_model','recurrent_state'}:
        raise ValueError('Immutable receipt drift')
    values={
      'HASH_ADAPTER_SHA256':artifacts['adapter_model'],
      'HASH_STATE_SHA256':artifacts['recurrent_state'],
      'HASH_RECEIPT_SHA256':hashlib.sha256(receipt.read_bytes()).hexdigest(),
      'HASH_LAUNCHER_SHA256':hashlib.sha256(launcher.read_bytes()).hexdigest(),
    }
    text=template.read_text()
    for old,new in values.items():
        if text.count(old)!=1: raise ValueError(f'Placeholder drift: {old}')
        text=text.replace(old,new)
    output.parent.mkdir(parents=True,exist_ok=True); partial=output.with_suffix('.yaml.partial')
    partial.write_text(text); os.replace(partial,output)
    return values

def main():
    p=argparse.ArgumentParser(); p.add_argument('--template',type=Path,required=True); p.add_argument('--receipt',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--launcher',type=Path,required=True); a=p.parse_args()
    print(json.dumps(materialize(a.template,a.receipt,a.output,a.launcher),sort_keys=True))
if __name__=='__main__': main()
