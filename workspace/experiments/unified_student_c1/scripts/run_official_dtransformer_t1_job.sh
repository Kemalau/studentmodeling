#!/usr/bin/env bash
set -euo pipefail

: "${KT_SEED:?missing KT_SEED}"
: "${KT_DATA_DIR:?missing KT_DATA_DIR}"
: "${KT_OUTPUT_DIR:?missing KT_OUTPUT_DIR}"
: "${KT_DATA_MANIFEST_SHA256:?missing KT_DATA_MANIFEST_SHA256}"

test "$(cat third_party/DTransformer/PINNED_COMMIT)" = 1dc459804911b3900ed23d9f6b1d7407d2e919a8
test "$(sha256sum "$KT_DATA_DIR/manifest.json" | cut -d' ' -f1)" = "$KT_DATA_MANIFEST_SHA256"

python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["KT_DATA_DIR"])
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
assert manifest["status"] == "complete"
assert manifest["stage"] == "modern_kt_t1_five_source_split_v1"
assert {key: manifest["splits"][key]["rows"] for key in ("train", "valid", "test")} == {
    "train": 99167,
    "valid": 11358,
    "test": 10913,
}
assert not any(manifest["overlap"].values())
for split in ("train", "valid", "test"):
    digest = hashlib.sha256()
    with (root / f"{split}.jsonl").open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == manifest["splits"][split]["sha256"]
PY

mkdir -p "$KT_OUTPUT_DIR"
python -u scripts/train_dtransformer_t1_baseline.py \
  --train-jsonl "$KT_DATA_DIR/train.jsonl" \
  --valid-jsonl "$KT_DATA_DIR/valid.jsonl" \
  --test-jsonl "$KT_DATA_DIR/test.jsonl" \
  --official-repo third_party/DTransformer \
  --official-commit 1dc459804911b3900ed23d9f6b1d7407d2e919a8 \
  --output-json "$KT_OUTPUT_DIR/metrics.json" \
  --seed "$KT_SEED" \
  --epochs 30 --patience 5 \
  --batch-size 64 --eval-batch-size 256 --max-seq-len 13 \
  --learning-rate 1e-3 --weight-decay 1e-5 \
  --d-model 128 --d-fc 256 --n-heads 8 --n-know 32 --n-layers 3 \
  --dropout 0.2 --lambda-cl 0.1 --proj --no-hard-neg \
  --device cuda --bf16

test -s "$KT_OUTPUT_DIR/metrics.json"
cp "$KT_OUTPUT_DIR/metrics.json" "$AMLT_OUTPUT_DIR/metrics.json"
