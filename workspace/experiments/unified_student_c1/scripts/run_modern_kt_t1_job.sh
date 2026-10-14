#!/usr/bin/env bash
set -euo pipefail

: "${KT_MODEL:?missing KT_MODEL}"
: "${KT_SEED:?missing KT_SEED}"
: "${KT_DATA_DIR:?missing KT_DATA_DIR}"
: "${KT_OUTPUT_DIR:?missing KT_OUTPUT_DIR}"
: "${KT_DATA_MANIFEST_SHA256:?missing KT_DATA_MANIFEST_SHA256}"

test "$(cat third_party/pykt_toolkit/PINNED_COMMIT)" = 3aabdc3cab06a57e745518b070d31bcbb823af2c
test "$(sha256sum "$KT_DATA_DIR/manifest.json" | cut -d' ' -f1)" = "$KT_DATA_MANIFEST_SHA256"
python - <<'PY'
import hashlib, json, os
from pathlib import Path
root=Path(os.environ['KT_DATA_DIR'])
value=json.loads((root/'manifest.json').read_text())
assert value['status']=='complete' and value['stage']=='modern_kt_t1_five_source_split_v1', value
assert {k:value['splits'][k]['rows'] for k in ('train','valid','test')} == {'train':99167,'valid':11358,'test':10913}, value
assert not any(value['overlap'].values()), value
for split in ('train','valid','test'):
    digest=hashlib.sha256()
    with (root/f'{split}.jsonl').open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(chunk)
    assert digest.hexdigest()==value['splits'][split]['sha256'], (split,digest.hexdigest())
PY

mkdir -p "$KT_OUTPUT_DIR"
python -u scripts/train_modern_kt_t1.py \
  --model "$KT_MODEL" \
  --train-jsonl "$KT_DATA_DIR/train.jsonl" \
  --valid-jsonl "$KT_DATA_DIR/valid.jsonl" \
  --test-jsonl "$KT_DATA_DIR/test.jsonl" \
  --pykt-root third_party/pykt_toolkit \
  --output-json "$KT_OUTPUT_DIR/metrics.json" \
  --predictions-jsonl "$KT_OUTPUT_DIR/predictions.jsonl" \
  --epochs 30 --patience 5 --batch-size 512 --max-seq-len 13 \
  --d-model 128 --dropout 0.1 --learning-rate 1e-3 \
  --seed "$KT_SEED" --device cuda

test -s "$KT_OUTPUT_DIR/metrics.json"
test "$(wc -l < "$KT_OUTPUT_DIR/predictions.jsonl")" -eq 10913
cp "$KT_OUTPUT_DIR/metrics.json" "$AMLT_OUTPUT_DIR/metrics.json"
cp "$KT_OUTPUT_DIR/predictions.jsonl" "$AMLT_OUTPUT_DIR/predictions.jsonl"
