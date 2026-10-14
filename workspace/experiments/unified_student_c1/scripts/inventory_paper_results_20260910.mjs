import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const project = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repository = path.resolve(project, '../../..');
const results = path.join(project, 'results');
const output = path.join(repository, 'paper_iclr/notes/server_data_inventory_20260910');
const roots = [
  'formal_specialist_v1', 'formal_specialist_v2',
  'taskonly_qwen_formal_eval_20260813', 'neural_response_specialists_20260814',
  'modern_kt_t1_five_source_padfix_20260824', 'modern_kt_t1_combined_20260824',
  'modern_kt_t1_rekt_confirmatory_20260824', 'public_t6_future_kc_baseline_20260814',
  't1_t2_paired_addone_recovery_20260901',
  'c4_strict3seed_v2_seed13_download_20260824',
  'c4_strict3seed_v2_seed97_download_20260824',
];
const names = /^(metrics|formal_evaluation_matrix|irt_pfa|akt|dkt|bkt|task_specialists|t3_specialists|comparison|bootstrap|summary|seed13_audit|seed97_audit|complete|target_exposure_audit)\.json$/;
const metricKey = /accuracy|auroc|auprc|auc$|macro.?f1|balanced_accuracy|qwk|quadratic|mrr|recall|exact_match|math_equivalence|bert.?score|bs.?f1|semantic|llm_judge|format_valid|brier|nll$|^n$|support|sampled_rows|generation_rows|classification_rows|flip|hit|transfer|directed|noise|random|noop|same_rule|wrong_rule|paired|bootstrap|ci95|95ci|confidence_interval|^D$|^D_|_minus_|^value$|^secondary$/i;

function walk(directory) {
  if (!fs.existsSync(directory)) return [];
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
    const filename = path.join(directory, entry.name);
    if (entry.isDirectory()) return walk(filename);
    return entry.isFile() && names.test(entry.name) ? [filename] : [];
  });
}

function numericMetrics(value, keys = [], matched = false, records = []) {
  if (typeof value === 'number' && Number.isFinite(value) && matched) {
    records.push({ key: keys.join('.'), value });
  } else if (value && typeof value === 'object') {
    for (const [key, child] of Object.entries(value)) {
      if (/^(history|train_history|training_history|predictions|examples|records|per_example|per_query|raw_output|target|prompt)$/.test(key)) continue;
      numericMetrics(child, [...keys, key], metricKey.test(key) || matched, records);
    }
  }
  return records;
}

function pooledAdditiveMetrics(data) {
  const groups = Object.values(data.metrics_by_group ?? {});
  if (!groups.length || groups.some(group => !Number.isInteger(group.n) || group.n <= 0)) return null;
  const support = groups.reduce((total, group) => total + group.n, 0);
  const keys = [
    'normalized_exact_match', 'joint_exact_match', 'math_equivalence',
    'joint_math_equivalence', 'response_math_equivalence',
    'response_normalized_exact_match', 'correctness_accuracy', 'llm_judge_accuracy',
  ];
  return {
    n: support,
    aggregation: 'Sample-weighted existing per-example indicator means; never pooled AUROC/F1.',
    ...Object.fromEntries(keys.filter(key => groups.every(group => typeof group[key] === 'number'))
      .map(key => [key, groups.reduce((total, group) => total + group.n * group[key], 0) / support])),
  };
}

const files = roots.flatMap(root => walk(path.join(results, root)));
for (const name of fs.readdirSync(results)) {
  if (/^c4_.*\.(summary|delivery_audit)\.json$/.test(name)) files.push(path.join(results, name));
}
const artifacts = [...new Set(files)].sort().map(filename => {
  const contents = fs.readFileSync(filename);
  const data = JSON.parse(contents);
  return {
    path: path.relative(repository, filename),
    sha256: createHash('sha256').update(contents).digest('hex'),
    modified_utc: fs.statSync(filename).mtime.toISOString(),
    provenance: Object.fromEntries([
      'stage', 'status', 'architecture', 'seed', 'model', 'public_task',
      'judge_backend', 'judge_effective_model', 'judge_identity', 'judge_rubric',
      'judge_rubric_sha256', 'judge_errors', 'judged_rows',
    ].filter(key => data[key] !== undefined).map(key => [key, data[key]])),
    pooled_additive_metrics: pooledAdditiveMetrics(data),
    metrics: numericMetrics(data),
  };
});
const matrixPath = path.join(results, 'taskonly_qwen_formal_eval_20260813/formal_specialist_v3/formal_evaluation_matrix.json');
const matrix = JSON.parse(fs.readFileSync(matrixPath, 'utf8'));
const mainRows = matrix.rows.slice(0, 4);
assert.deepEqual(['T1', 'T2', 'T3', 'T4', 'T5'].map(task => mainRows[1][task].n), [10913, 3693, 1494, 4600, 976]);
assert.equal(['T1', 'T2', 'T3', 'T4', 'T5'].reduce((total, task) => total + mainRows[1][task].n, 0), 21676);
const link = filename => `[${path.relative(repository, filename)}](${path.relative(path.dirname(output), filename).split(path.sep).join('/')})`;
const format = value => typeof value === 'number' ? value.toFixed(6) : '--';
const lines = [
  '# Server result inventory (2026-09-10)', '',
  'Scope: existing local artifacts in the studentmodeling workspace. No new inference, judging, training, or cloud job-status query was performed. Local modification times do not establish experiment completion dates.', '',
  'The JSON and TSV preserve existing numeric metric fields with source paths and SHA-256 hashes. They include historical, partial, and diagnostic artifacts: presence is NOT approval for a paper claim. No student-level predictions are exported.', '',
  '## Canonical five-task matrix', '',
  `Source: ${link(matrixPath)}`, '',
  '| Model | T1 Accuracy | T2 MathEq | T3 judge accuracy | T4 JointMath | T5 Accuracy |',
  '|---|---:|---:|---:|---:|---:|',
  ...mainRows.map(row => `| ${row.model} | ${['T1', 'T2', 'T3', 'T4', 'T5'].map(task => format(row[task].value)).join(' | ')} |`), '',
  'T1/T2/T3/T4/T5 here mean correctness/response/diagnosis/future response/persistence. Legacy T5/T6 in older JSON files mean public T4/T5. BKT has only 9,929 rows. T3 specialist values are exact taxonomy classification, not the same semantic-judge protocol as the LLM rows.', '',
  '## Existing secondary scores', '',
  '| Model | T1 AUROC | T4 correctness accuracy | T5 AUROC |',
  '|---|---:|---:|---:|',
  ...mainRows.map(row => `| ${row.model} | ${['T1', 'T4', 'T5'].map(task => format(row[task].secondary)).join(' | ')} |`), '',
  '## Cross-task differences', '',
  '| Task | Unified minus task-only | 95% paired bootstrap CI |',
  '|---|---:|---|',
  ...Object.entries(matrix.summary.unified_minus_taskonly_paired).map(([task, item]) => `| ${task} | ${format(item.unified_minus_taskonly)} | ${item.paired_bootstrap_95ci ? item.paired_bootstrap_95ci.map(format).join(', ') : 'Unavailable: archived Unified raw response/target missing'} |`), '',
  'Intervals above are example-paired uncertainty, not training-seed SD. The Eedi question-clustered interval is a separate result. Never use difference intervals as error bars on absolute-score bars.', '',
  '## Missing-metric guardrails', '',
  '- The new manuscript EM/NM/SA/BS-F1 table is not the old task-primary-metric matrix.',
  '- Existing normalized_exact_match and joint_exact_match fields are exported as named; they are not automatically raw full-output EM.',
  '- T3 semantic results should use the v2 TRAPI judge artifacts, not the older v1 mixed exact-shortcut judge.',
  '- Base Qwen T3 = 0.634538 exceeds Unified T3 = 0.566265 in the canonical matrix.',
  '- Unified full-open, one-shot causal CF1, and later candidate-free A/B checkpoints must remain separate.', '',
  '## Indexed artifacts', '',
  `Indexed ${artifacts.length} files and ${artifacts.reduce((total, artifact) => total + artifact.metrics.length, 0)} numeric metric fields.`, '',
  '| Artifact | Numeric fields | SHA-256 |', '|---|---:|---|',
  ...artifacts.map(artifact => `| ${link(path.join(repository, artifact.path))} | ${artifact.metrics.length} | ${artifact.sha256} |`), '',
];
fs.mkdirSync(path.dirname(output), { recursive: true });
fs.writeFileSync(`${output}.json`, JSON.stringify({ scope: roots, matrix, artifacts }, null, 2) + '\n');
fs.writeFileSync(`${output}.tsv`, 'artifact\tsha256\tmetric_path\tvalue\n' + artifacts.flatMap(artifact => artifact.metrics.map(metric => [artifact.path, artifact.sha256, metric.key, metric.value].join('\t'))).join('\n') + '\n');
fs.writeFileSync(`${output}.md`, lines.join('\n'));
assert.equal(JSON.parse(fs.readFileSync(`${output}.json`, 'utf8')).artifacts.length, artifacts.length);
console.log(JSON.stringify({ files: artifacts.length, numeric_fields: artifacts.reduce((total, artifact) => total + artifact.metrics.length, 0), output }, null, 2));