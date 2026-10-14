#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "UNIFIED_T4_PAIRED_RESCORE_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

export PATH="/tmp/studentmodeling-unified-t4-paired-venv/bin:$PATH"
export PYTHONPATH="$AMLT_CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/hf_cache
export HF_HUB_CACHE=/tmp/hf_cache/hub
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error

model=/tmp/Qwen3-32B
adapter=/mnt/amulet/studentmodeling/adapters/qwen3_32b_full_trainvalid_no_t4_strictt6_1ep/final_adapter
eval_data=/mnt/amulet/studentmodeling/formal_evaluation_v1/test.jsonl
raw="$AMLT_OUTPUT_DIR/eval/raw"
scored="$AMLT_OUTPUT_DIR/eval/unified_t4_scored.jsonl"
receipt="$AMLT_OUTPUT_DIR/eval/completion.json"

test "$(sha256sum scripts/evaluate_formal_protocols.py | cut -d' ' -f1)" = b347ed4d2228725a325975c0ffab162ca32a2e0252583de841c00b3ab1e196de
test "$(sha256sum scripts/copy_model_parallel.py | cut -d' ' -f1)" = 62b37ab70bc2ffc81c8314ca69215da7facdf18dba3f023d83096d1c3526ea4c
test "$(sha256sum "$eval_data" | cut -d' ' -f1)" = bff7726953ae048bedf6dacdbb7500a54cb8de305e406f2497ef1f11c04167f8
test -s "$adapter/adapter_model.safetensors"
test "$(df --output=avail /tmp | tail -1)" -ge 125829120

mkdir -p "$HF_HUB_CACHE" "$raw"
python scripts/copy_model_parallel.py \
  --source /mnt/amulet/studentmodeling/models/Qwen3-32B \
  --destination "$model" --workers 16
test -s "$model/config.json"

python -u scripts/evaluate_formal_protocols.py \
  --model "$model" --adapter "$adapter" --eval-jsonl "$eval_data" \
  --output-dir "$raw" --tasks predict_prospective_error \
  --max-length 1536 --max-new-tokens 128 --generation-batch-size 2 \
  --state-token-alias '<|vision_pad|>' --seed 42 \
  --save-unredacted-predictions

python - "$raw/predictions.jsonl" "$raw/metrics.json" "$scored" "$receipt" \
  "$adapter/adapter_model.safetensors" "$eval_data" <<'PY'
import hashlib, html, json, pathlib, re, sys
from fractions import Fraction

predictions_path, metrics_path, scored_path, receipt_path, adapter_path, eval_path = map(pathlib.Path, sys.argv[1:])

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def extract(text, tag):
    match = re.search(rf'<{tag}>\s*(.*?)\s*</{tag}>', text, flags=re.I | re.S)
    return html.unescape(match.group(1).strip()) if match else None

def normalize(value):
    text = html.unescape(str(value or '')).casefold().strip()
    return re.sub(r'\s+', '', text.replace('−', '-').replace('×', '*').replace('÷', '/'))

def numeric(value):
    text = normalize(value).replace(',', '')
    if not text or len(text) > 80 or not re.fullmatch(r'[+\-]?(?:\d+(?:\.\d+)?|\.\d+)(?:/[+\-]?(?:\d+(?:\.\d+)?|\.\d+))?', text):
        return None
    try:
        if '/' in text:
            a, b = text.split('/', 1)
            return Fraction(a) / Fraction(b)
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None

def equivalent(a, b):
    if normalize(a) == normalize(b):
        return True
    av, bv = numeric(a), numeric(b)
    return av is not None and bv is not None and av == bv

rows = [json.loads(line) for line in predictions_path.open() if line.strip()]
assert len(rows) == 4600 and len({row['example_id'] for row in rows}) == 4600
output = []
for row in rows:
    assert row['task'] == 'predict_prospective_error'
    pc = extract(row['raw_output'], 'correctness')
    gc = extract(row['target'], 'correctness')
    pr = extract(row['raw_output'], 'predicted_response')
    gr = extract(row['target'], 'predicted_response')
    assert None not in (pc, gc, pr, gr), row['example_id']
    output.append({
        'example_id': row['example_id'],
        'task': row['task'],
        'is_correct': normalize(pc) == normalize(gc) and equivalent(pr, gr),
        'raw_output_sha256': hashlib.sha256(row['raw_output'].encode()).hexdigest(),
        'target_sha256': hashlib.sha256(row['target'].encode()).hexdigest(),
    })
with scored_path.open('w') as handle:
    for row in output:
        handle.write(json.dumps(row, sort_keys=True) + '\n')
metrics = json.load(metrics_path.open())
correct = sum(row['is_correct'] for row in output)
assert metrics['sampled_rows'] == 4600 and metrics['generation_rows'] == 4600
completion = {
    'stage': 'unified_t4_paired_rescore_v1',
    'status': 'complete',
    'rows': len(output),
    'correct': correct,
    'primary': correct / len(output),
    'adapter_sha256': sha(adapter_path),
    'evaluation_input_sha256': sha(eval_path),
    'scored_predictions_sha256': sha(scored_path),
    'contains_answer_text': False,
}
receipt_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + '\n')
print(json.dumps(completion, sort_keys=True))
PY

# Preserve only aggregate metrics, hashes, and a Boolean correctness bit per row.
rm -rf "$raw"
test -s "$scored" -a -s "$receipt"
