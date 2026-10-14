#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "STATE8_EEDI_REPAIR_BRANCH_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

: "${REPAIR_BRANCH:?}" "${REPAIR_SEED:?}" "${SOURCE_CHECKPOINT_ARM:?}"
: "${SOURCE_CHECKPOINT_STEP:?}" "${SOURCE_CHECKPOINT_ROOT:?}"
: "${EXPECTED_STUDENT_ADAPTER_SHA256:?}" "${EXPECTED_STUDENT_STATE_SHA256:?}"
: "${STATE8_EEDI_CAUSAL_TWINS:?}" "${STATE8_CAUSAL_REPEAT_FRACTION:?}"
: "${STATE8_CAUSAL_WEIGHT:?}" "${STATE8_CAUSAL_MARGIN:?}"
: "${STATE8_EEDI_GEOMETRY_WEIGHT:?}" "${SOURCE_ANCHOR_MODE:?}"
: "${REPAIR_REMOTE_ROOT:?}"

case "$REPAIR_BRANCH" in balanced_continuation|semantic_binding|specificity_triplet|identity_controls|causal_teacher_low|causal_teacher_high|early_label_low|early_label_high) ;; *) exit 2 ;; esac
case "$SOURCE_CHECKPOINT_ARM" in updater|full|curriculum_generic|curriculum_current_query) ;; *) exit 2 ;; esac
case "$SOURCE_CHECKPOINT_STEP" in 20|100|2000|3000|4000|5000) ;; *) exit 2 ;; esac
case "$SOURCE_ANCHOR_MODE" in updater_mse1|full_mse1) ;; *) exit 2 ;; esac
[[ "$REPAIR_SEED" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$STATE8_EEDI_CAUSAL_TWINS" =~ ^[1-9][0-9]*$ ]] || exit 2
for digest in "$EXPECTED_STUDENT_ADAPTER_SHA256" "$EXPECTED_STUDENT_STATE_SHA256"; do
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || exit 2
done
test "$(sha256sum scripts/train_state8_semantic_causal_repair.py | cut -d' ' -f1)" = 0726a30d3411b4e3b9468046924d93a06bc2f2913063d009773256b4a1b39b08
case "$SOURCE_CHECKPOINT_ARM" in
  updater|full)
    expected_root="/mnt/amulet/studentmodeling/checkpoint_snapshots/state8_eedi_same_question_20260811/${SOURCE_CHECKPOINT_ARM}/checkpoint-${SOURCE_CHECKPOINT_STEP}" ;;
  curriculum_generic)
    expected_root="/mnt/amulet/studentmodeling/checkpoint_snapshots/state8_diagnosis_first_eedi_curriculum_20260812/checkpoint-${SOURCE_CHECKPOINT_STEP}" ;;
  curriculum_current_query)
    expected_root="/mnt/amulet/studentmodeling/checkpoint_snapshots/state8_diagnosis_first_eedi_current_query_curriculum_20260812/checkpoint-${SOURCE_CHECKPOINT_STEP}" ;;
esac
test "$SOURCE_CHECKPOINT_ROOT" = "$expected_root"
anchor=full; [[ "$SOURCE_ANCHOR_MODE" == updater_mse1 ]] && anchor=updater
expected_output="/mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed${REPAIR_SEED}_taskanchor_source_anchor_${anchor}_mse1_eedi_same_question_v1_${REPAIR_BRANCH}_seed${REPAIR_SEED}"
test "$REPAIR_REMOTE_ROOT" = "$expected_output"
case "$REPAIR_BRANCH" in
  causal_teacher_low|causal_teacher_high)
    : "${STATE8_CAUSAL_TEACHER_TRAJECTORY_WEIGHT:?}"
    : "${STATE8_CAUSAL_TEACHER_COSINE_WEIGHT:?}"
    : "${STATE8_CAUSAL_TEACHER_RELATIVE_MSE_WEIGHT:?}"
    ;;
  early_label_low|early_label_high)
    : "${STATE8_EARLY_LABEL_BINDING_WEIGHT:?}"
    ;;
esac

source_runner=scripts/run_state8_eedi_same_question_causal_repair.sh
rendered=/tmp/run_state8_eedi_same_question_repair_branch.rendered.sh
expected_source=567cded4455bfd7e926e2c83dbf138568cd326dc1cd94653f30d8f4b20aad545
test "$(sha256sum "$source_runner" | cut -d' ' -f1)" = "$expected_source"

python - "$source_runner" "$rendered" <<'PY'
from pathlib import Path
import sys

source, output = map(Path, sys.argv[1:3])
text = source.read_text(encoding="utf-8")
needle = '''export STATE8_CURRENT_QUERY_ROUTING=eedi_only
bash -n "$patched_runner"
bash "$patched_runner"'''
replacement = r'''export STATE8_CURRENT_QUERY_ROUTING=eedi_only
python - "$patched_runner" <<'PY_REPAIR_BRANCH'
from pathlib import Path
import hashlib, os, sys

path = Path(sys.argv[1])
value = path.read_text(encoding="utf-8")
branch = os.environ["REPAIR_BRANCH"]
seed = os.environ["REPAIR_SEED"]
mode = os.environ["SOURCE_ANCHOR_MODE"]
twins = int(os.environ["STATE8_EEDI_CAUSAL_TWINS"])
source_root = os.environ["SOURCE_CHECKPOINT_ROOT"]

# The ordinary source-anchor launcher accepts completed final-fit receipts.
# Repair branches intentionally warm-start from an immutable checkpoint
# receipt, whose schema omits full-epoch counters.  Patch only that fail-closed
# provenance clause and bind the source-anchor generator to the patched base.
base_source = Path("scripts/run_state8_stateonly_repair_ab.sh")
base_output = Path("/tmp/run_state8_eedi_repair_base.rendered.sh")
base_value = base_source.read_text(encoding="utf-8")
receipt_old = (
    "elif si.get('status')=='complete' and si.get('sha256'):\n"
    " expected={key:si['sha256'][key] for key in observed}\n"
    " assert si['micro_steps']>=1000 and si['real_examples_seen_once'] is True,si"
)
receipt_new = (
    "elif si.get('status')=='complete' and si.get('sha256'):\n"
    " expected={key:si['sha256'][key] for key in observed}\n"
    " if 'micro_steps' in si:\n"
    "  assert si['micro_steps']>=1000 and si['real_examples_seen_once'] is True,si\n"
    " else:\n"
    "  assert set(si)=={'status','step','sha256'} and si['step'] in {20,100,2000,3000,4000,5000},si"
)
assert base_value.count(receipt_old) == 1, base_value.count(receipt_old)
base_output.write_text(base_value.replace(receipt_old, receipt_new), encoding="utf-8")
base_hash = hashlib.sha256(base_output.read_bytes()).hexdigest()

def replace(old: str, new: str, count: int = 1) -> None:
    global value
    observed = value.count(old)
    assert observed == count, (old, observed, count)
    value = value.replace(old, new)

replace("source_launcher=scripts/run_state8_stateonly_repair_ab.sh",
        f"source_launcher={base_output}")
replace("source_hash=2d52d2d84b64dcc1d177583ddb96b006e6afdeee74cb2d3241325ae170a059b7",
        f"source_hash={base_hash}")

if mode == "updater_mse1":
    old_seed, old_tag = "3072", "source_anchor_updater_mse1_eedi_same_question_v1"
else:
    old_seed, old_tag = "3071", "source_anchor_full_mse1_eedi_same_question_v1"
replace(f"export TRAIN_SEED={old_seed}", f"export TRAIN_SEED={seed}")
replace(f"export RUN_TAG={old_tag}", f"export RUN_TAG={old_tag}_{branch}_seed{seed}")
replace("export STATE8_CAUSAL_REPEAT_FRACTION=0.50",
        f"export STATE8_CAUSAL_REPEAT_FRACTION={os.environ['STATE8_CAUSAL_REPEAT_FRACTION']}")
replace("export STATE8_CAUSAL_WEIGHT=2.0",
        f"export STATE8_CAUSAL_WEIGHT={os.environ['STATE8_CAUSAL_WEIGHT']}")
replace("export STATE8_CAUSAL_MARGIN=0.5",
        f"export STATE8_CAUSAL_MARGIN={os.environ['STATE8_CAUSAL_MARGIN']}")
replace("export STATE8_EEDI_GEOMETRY_WEIGHT=0.05",
        f"export STATE8_EEDI_GEOMETRY_WEIGHT={os.environ['STATE8_EEDI_GEOMETRY_WEIGHT']}")
replace(
    "export STUDENT_ROOT=/mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed3066_trajectory01_matched_cp3000_semantic_preserve_v1",
    f"export STUDENT_ROOT={source_root}",
)
replace('export STUDENT_MODEL_DIR="$STUDENT_ROOT/model"',
        'export STUDENT_MODEL_DIR="$STUDENT_ROOT/adapter"')
replace('export STUDENT_STATE_PATH="$STUDENT_ROOT/model/recurrent_state.pt"',
        'export STUDENT_STATE_PATH="$STUDENT_ROOT/recurrent_state.pt"')
replace('export STUDENT_RECEIPT="$STUDENT_ROOT/metrics/publish_receipt.json"',
        'export STUDENT_RECEIPT="$STUDENT_ROOT/checkpoint_completion.json"')
replace("--twins 100 --seed 20260811 --offset 0",
        f"--twins {twins} --seed 20260811 --offset 0")
replace("--twins 25 --seed 20260811 --offset 100",
        f"--twins 25 --seed 20260811 --offset {twins}")
replace("assert train['output_pairs']==200 and train['offset']==0,train",
        f"assert train['output_pairs']=={2 * twins} and train['offset']==0,train")
replace("assert valid['output_pairs']==50 and valid['offset']==100,valid",
        f"assert valid['output_pairs']==50 and valid['offset']=={twins},valid")
replace("assert merged['rows']==700,merged",
        f"assert merged['rows']=={500 + 2 * twins},merged")
replace("assert merged['source_counts']=={'eedi_mining':200,'malrule':500},merged",
        f"assert merged['source_counts']=={{'eedi_mining':{2 * twins},'malrule':500}},merged")
replace(
    'expected_pairs=250 if os.environ[\'STATE8_CAUSAL_DIAGNOSIS_ONLY\']==\'1\' else 700',
    f'expected_pairs=250 if os.environ[\'STATE8_CAUSAL_DIAGNOSIS_ONLY\']==\'1\' else {500 + 2 * twins}',
)
path.write_text(value, encoding="utf-8")
PY_REPAIR_BRANCH
bash -n "$patched_runner"
bash "$patched_runner"'''
assert text.count(needle) == 1, text.count(needle)
output.write_text(text.replace(needle, replacement), encoding="utf-8")
PY

bash -n "$rendered"
bash "$rendered"
