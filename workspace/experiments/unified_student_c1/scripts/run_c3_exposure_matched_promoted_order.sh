#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "C3_PROMOTED_ORDER_FAILED line=%s rc=%s command=%s\n" "${LINENO}" "$rc" "${BASH_COMMAND}" >&2; exit "$rc"' ERR

: "${C3_PROMOTED_ENTRYPOINT_SHA256:?C3_PROMOTED_ENTRYPOINT_SHA256 is required}"

# Keep the widely reused task-retention launcher immutable.  Render one
# C3-only variant after verifying every replaced source fragment, then execute
# it.  This avoids changing any non-C3 experiment's provenance contract.
python - scripts/run_layerwise_task_retention_rehearsal.sh \
  /tmp/run_c3_exposure_matched_promoted_order.rendered.sh \
  "$C3_PROMOTED_ENTRYPOINT_SHA256" <<'PY'
import hashlib
import pathlib
import sys

source, output = map(pathlib.Path, sys.argv[1:3])
entrypoint_sha = sys.argv[3]
raw = source.read_bytes()
expected = "93ff8cd9bac2a06d84ff5f1d6089ed4161fd2e4b387b18b259a83a9e6d9501ac"
assert hashlib.sha256(raw).hexdigest() == expected
text = raw.decode("utf-8")
old = "scripts/train_recurrent_student_m4.py|scripts/train_recurrent_student_m4_open_t3.py|scripts/train_recurrent_student_m4_open_t3_selective_kd.py"
new = "scripts/train_recurrent_student_m4.py|scripts/train_recurrent_student_m4_open_t3.py|scripts/train_recurrent_student_m4_open_t3_c3_promoted_order.py|scripts/train_recurrent_student_m4_open_t3_selective_kd.py"
assert text.count(old) == 1
text = text.replace(old, new)
old = """  scripts/train_recurrent_student_m4_open_t3_selective_kd.py)
    EXPECTED_ENTRYPOINT_SHA256=2474b77459701fa36ec564c84b7f3b6b0d158c9ab128b5147111d252662b0931
    ;;"""
new = f"""  scripts/train_recurrent_student_m4_open_t3_c3_promoted_order.py)
    EXPECTED_ENTRYPOINT_SHA256={entrypoint_sha}
    ;;
{old}"""
assert text.count(old) == 1
text = text.replace(old, new)
old = """if os.environ.get('TRAIN_ENTRYPOINT') in {
 'scripts/train_recurrent_student_m4_external_state_teacher.py',
 'scripts/train_recurrent_student_m4_gold_semantic_teacher.py',
}:"""
new = old + """
 pass
if os.environ.get('TRAIN_ENTRYPOINT')=='scripts/train_recurrent_student_m4_open_t3_c3_promoted_order.py':
 assert config['c3_promoted_order_contract']=='round1500_promoted_order_plus_controlled_segment_v1',config
 assert config['c3_promoted_source_order_sha256']=='a212f65b3efdee343f673e9cd2485c9ce66cc58dbd20c1f82cdebd6704226134',config
 assert config['c3_promoted_source_manifest_sha256']=='adc6eacca29cf22c554875d0f1e5cd8e6cbd6722966228b3aafe8a3e62ca0ee7',config
 assert config['c3_controlled_segment_start']==24000 and config['c3_controlled_segment_rows']==1000,config
 assert config['c3_controlled_manifest_sha256']==sys.argv[3],config
if os.environ.get('TRAIN_ENTRYPOINT') in {
 'scripts/train_recurrent_student_m4_external_state_teacher.py',
 'scripts/train_recurrent_student_m4_gold_semantic_teacher.py',
}:"""
assert text.count(old) == 1
text = text.replace(old, new)
old = """if [[ \"$TRAIN_ENTRYPOINT\" == scripts/train_recurrent_student_m4_open_t3_selective_kd.py ]]; then"""
new = """if [[ \"$TRAIN_ENTRYPOINT\" == scripts/train_recurrent_student_m4_open_t3_c3_promoted_order.py ]]; then
  test -f \"$RUN_DIR/open_t3_training_view.json\"
  test -f \"$RUN_DIR/c3_promoted_real_order.json\"
  cp \"$RUN_DIR/open_t3_training_view.json\" \\
    \"$AMLT_OUTPUT_DIR/metrics/open_t3_training_view.json\"
  cp \"$RUN_DIR/c3_promoted_real_order.json\" \\
    \"$AMLT_OUTPUT_DIR/metrics/c3_promoted_real_order.json\"
fi
""" + old
assert text.count(old) == 1
text = text.replace(old, new)
output.write_text(text, encoding="utf-8")
print(hashlib.sha256(text.encode("utf-8")).hexdigest())
PY
bash -n /tmp/run_c3_exposure_matched_promoted_order.rendered.sh
bash /tmp/run_c3_exposure_matched_promoted_order.rendered.sh
