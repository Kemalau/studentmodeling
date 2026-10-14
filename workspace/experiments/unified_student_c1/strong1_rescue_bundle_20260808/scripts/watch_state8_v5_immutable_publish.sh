#!/usr/bin/env bash
set -uo pipefail

project_dir=/home/v-yuejiali/studentmodeling/workspace/experiments/unified_student_c1
experiment=state8-natural-dualpath-samerule-v5-20260805
train_dir="$project_dir/results/state8_natural_dualpath_samerule_v5_20260805"
state_dir="$project_dir/results/state8_v5_immutable_publish_20260805"
download="$state_dir/download"
account=csnewsandfeeds4150361735
container=amulet
prefix=studentmodeling/diagnostics/state8_natural_dualpath_samerule_v5_immutable
python_bin=/home/v-yuejiali/.venvs/studentmodeling/bin/python

mkdir -p "$state_dir"
cd "$project_dir" || exit 1
exec 9>"$state_dir/controller.lock"; flock -n 9 || exit 0
echo $$ > "$state_dir/controller.active.pid"
trap 'rm -f "$state_dir/controller.active.pid"' EXIT
rm -f "$state_dir/needs_attention"
heartbeat(){ date -u +%Y-%m-%dT%H:%M:%SZ > "$state_dir/controller.heartbeat"; }
log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$state_dir/controller.log"; }
fail(){ log "ERROR: $*"; touch "$state_dir/needs_attention"; exit 1; }

while [[ ! -f "$train_dir/remote_passed" ]]; do
  heartbeat
  [[ ! -f "$train_dir/needs_attention" ]] || fail "state8 v5 source job needs attention"
  sleep 60
done

rm -rf "$download"; mkdir -p "$download"
amlt results download "$experiment" --status pass \
  --include 'adapter_model.safetensors' --include 'recurrent_state.pt' \
  --include 'adapter_config.json' --include 'tokenizer.json' \
  --include 'tokenizer_config.json' --include 'special_tokens_map.json' \
  --include 'added_tokens.json' --include 'merges.txt' --include 'vocab.json' \
  --include 'completion.json' --include 'state8_natural_curriculum_view.json' \
  --output "$download" >> "$state_dir/controller.log" 2>&1 \
  || fail "state8 v5 immutable source download failed"
readarray -t adapters < <(find "$download" -type f -path '*/model/adapter_model.safetensors')
readarray -t states < <(find "$download" -type f -path '*/model/recurrent_state.pt')
readarray -t adapter_configs < <(find "$download" -type f -path '*/model/adapter_config.json')
readarray -t completions < <(find "$download" -type f -path '*/metrics/completion.json')
readarray -t views < <(find "$download" -type f -path '*/metrics/state8_natural_curriculum_view.json')
[[ ${#adapters[@]} -eq 1 && ${#states[@]} -eq 1 && ${#adapter_configs[@]} -eq 1 \
   && ${#completions[@]} -eq 1 && ${#views[@]} -eq 1 ]] \
  || fail "immutable source cardinality drift"
model_dir="$(dirname "${adapters[0]}")"
receipt="$state_dir/immutable_receipt.json"
"$python_bin" - "${adapters[0]}" "${states[0]}" "${completions[0]}" "${views[0]}" "$receipt" <<'PY' \
  || fail "immutable source validation failed"
import hashlib,json,pathlib,sys
adapter,state,completion,view,out=map(pathlib.Path,sys.argv[1:])
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
c=json.load(open(completion)); v=json.load(open(view))
observed={'adapter_model':digest(adapter),'recurrent_state':digest(state)}
assert c['status']=='diagnostic_complete' and c['diagnostic'] is True,c
assert c['micro_steps']==1000 and c['optimizer_steps']==250,c
assert c['history_integrity_pass'] is True,c
assert {key:c['sha256'][key] for key in observed}==observed,(c,observed)
assert v['state_token_count']==8 and v['output_schema']=='label_diagnosis_evidence_v1',v
assert v['history_dropout_schedule']==[0.0,0.25,0.5,0.75,1.0],v
receipt={'status':'complete','stage':'state8_v5_immutable_publish_v1',
 'source_experiment':'state8-natural-dualpath-samerule-v5-20260805',
 'source_result_id':'7214096439.76502-8475e586-be83','sha256':observed,
 'micro_steps':1000,'state_shape':[64,8,5120],
 'raw_history_visible_to_decoder':False,'persistent_history_kv_cache':False}
out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
PY

# Publish data first and the immutable receipt last.  Consumers require the
# receipt, so they can never observe a partially copied checkpoint as complete.
az storage blob upload-batch --account-name "$account" --auth-mode login \
  --destination "$container" --destination-path "$prefix/model" \
  --source "$model_dir" --overwrite true >> "$state_dir/controller.log" 2>&1 \
  || fail "immutable model upload failed"
az storage blob upload --account-name "$account" --auth-mode login \
  --container-name "$container" --name "$prefix/metrics/completion.json" \
  --file "${completions[0]}" --overwrite true >> "$state_dir/controller.log" 2>&1 \
  || fail "immutable completion upload failed"
az storage blob upload --account-name "$account" --auth-mode login \
  --container-name "$container" --name "$prefix/metrics/state8_natural_curriculum_view.json" \
  --file "${views[0]}" --overwrite true >> "$state_dir/controller.log" 2>&1 \
  || fail "immutable curriculum receipt upload failed"
az storage blob upload --account-name "$account" --auth-mode login \
  --container-name "$container" --name "$prefix/metrics/immutable_receipt.json" \
  --file "$receipt" --overwrite true >> "$state_dir/controller.log" 2>&1 \
  || fail "immutable receipt upload failed"

for name in model/adapter_model.safetensors model/adapter_config.json \
  model/recurrent_state.pt metrics/completion.json metrics/immutable_receipt.json; do
  exists="$(az storage blob exists --account-name "$account" --auth-mode login \
    --container-name "$container" --name "$prefix/$name" --query exists -o tsv)" \
    || fail "immutable existence check failed: $name"
  [[ "$exists" == true ]] || fail "immutable blob missing: $name"
done
touch "$state_dir/complete" "$state_dir/scientific_gate_pass" "$train_dir/immutable_published"
heartbeat
log "state8 v5 immutable checkpoint published and verified"
