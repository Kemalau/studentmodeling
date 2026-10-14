// Read-only, post-hoc paired audit of existing judgments; no inference or writes.
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import assert from 'node:assert/strict';

const root = fileURLToPath(new URL('../', import.meta.url));
const files = {
  unified: 'results/formal_specialist_v2/unrestricted_qwen_t3_v2/adapter_diagnose_misconception/predictions_redacted.jsonl',
  taskonly: 'results/taskonly_qwen_formal_eval_20260813/T3/t3_semantic/predictions_redacted.jsonl',
};
const expectedHashes = {
  unified: '1017b47d36d311cfcbaa06e918c20fbdffbff4faef248b385c746a39926176a9',
  taskonly: '7ce491cf804075cfe3a88dd54e3c16759ccb7ab940cf5d5b6a61c7a4c0d6ceca',
};
const data = {};
for (const [arm, relative] of Object.entries(files)) {
  const raw = readFileSync(resolve(root, relative));
  assert.equal(createHash('sha256').update(raw).digest('hex'), expectedHashes[arm]);
  const rows = raw.toString().trim().split('\n').map(JSON.parse);
  assert.equal(rows.length, 1494);
  data[arm] = new Map(rows.map(r => [r.example_id, r]));
  assert.equal(data[arm].size, 1494);
}
const groups = new Map();
for (const [id, u] of data.unified) {
  const o = data.taskonly.get(id);
  assert.ok(o, id);
  for (const field of ['task', 'source', 'protocol', 'target_sha256',
    'llm_judge_backend', 'llm_judge_model', 'llm_judge_rubric']) {
    assert.equal(u[field], o[field], `${id}: ${field}`);
  }
  assert.equal(u.task, 'diagnose_misconception');
  assert.equal(u.llm_judge_backend, 'trapi');
  assert.equal(typeof u.llm_judge_correct, 'boolean');
  assert.equal(typeof o.llm_judge_correct, 'boolean');
  const group = `${u.source}/${u.protocol}`;
  if (!groups.has(group)) groups.set(group, []);
  groups.get(group).push({ id, u: Number(u.llm_judge_correct), o: Number(o.llm_judge_correct) });
}
function exactMcNemar(b, c) {
  const n = b + c;
  let mass = 2 ** -n, tail = 0;
  for (let k = 0; k <= Math.min(b, c); k++) {
    if (k > 0) mass *= (n - k + 1) / k;
    tail += mass;
  }
  return Math.min(1, 2 * tail);
}
const subgroupCounts = [...groups].sort(([a], [b]) => a < b ? -1 : 1).map(([group, rows]) => {
  const uOnly = rows.filter(r => r.u && !r.o).length;
  const oOnly = rows.filter(r => r.o && !r.u).length;
  return { group, n: rows.length, unified_hits: rows.reduce((s, r) => s + r.u, 0),
    taskonly_hits: rows.reduce((s, r) => s + r.o, 0),
    unified_only_correct: uOnly, taskonly_only_correct: oOnly,
    delta: (uOnly - oOnly) / rows.length, row_McNemar_p_unadjusted: exactMcNemar(uOnly, oOnly) };
});
const eedi = groups.get('eedi_mining/open_taxonomy_generation');
assert.equal(eedi.length, 894);
const questions = new Map();
for (const r of eedi) {
  const match = /:([^:]+)_[A-D]$/.exec(r.id);
  assert.ok(match, r.id);
  const q = match[1];
  if (!questions.has(q)) questions.set(q, { n: 0, d: 0 });
  questions.get(q).n++;
  questions.get(q).d += r.u - r.o;
}
const units = [...questions].sort(([a], [b]) => a < b ? -1 : 1).map(([, x]) => x);
assert.equal(units.length, 373);
let seed = 20260907;
function rng() {
  seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5;
  return (seed >>> 0) / 4294967296;
}
const draws = [];
for (let b = 0; b < 20000; b++) {
  let d = 0, n = 0;
  for (let j = 0; j < units.length; j++) {
    const unit = units[Math.floor(rng() * units.length)];
    d += unit.d; n += unit.n;
  }
  draws.push(d / n);
}
draws.sort((a, b) => a - b);
console.log(JSON.stringify({
  status: 'complete', files, sha256: expectedHashes, subgroup_counts: subgroupCounts,
  eedi_question_cluster: { n_questions: units.length, n_rows: eedi.length,
    estimate: eedi.reduce((s, r) => s + r.u - r.o, 0) / eedi.length,
    paired_bootstrap_95ci: [draws[500], draws[19499]], replicates: 20000,
    rng: 'xorshift32; initial seed 20260907; question IDs lexically sorted',
    cluster: 'Eedi question ID; answer options kept together; ratio-of-sums estimator' },
  scope: 'Post-hoc source/protocol analysis of one training seed; unadjusted intervals; target exposure matched, total updates not matched. Does not identify which auxiliary task caused the gain or establish training-seed replication.',
}, null, 2));
