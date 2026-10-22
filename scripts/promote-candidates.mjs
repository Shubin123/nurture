import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';

const read = path => JSON.parse(readFileSync(path, 'utf8'));
const hash = path => createHash('sha256').update(readFileSync(path)).digest('hex');
const evaluation = read('docs/reports/balanced-browser-evaluation.json');
const benchmark = read('docs/reports/balanced-browser-benchmark.json');
const challenge = read('docs/reports/balanced-challenge.json');
const native = read('docs/reports/balanced-evaluation.json');
const frozen = read('docs/reports/balanced-frozen-selection.json');
const workerHash = hash('docs/model-worker.js');
assert.equal(frozen.test_used_for_selection, false);
assert.equal(evaluation.held_out_examples, 7000);
assert.equal(evaluation.worker_sha256, workerHash);
assert.equal(benchmark.worker_sha256, workerHash);
const models = [];
for (const name of ['linear-balanced', 'bert-mini-balanced']) {
  const source = `ml/runs/candidate-models/${name}`;
  const reference = native.models[name.startsWith('linear') ? 'linear_balanced' : 'mini_balanced'];
  const original = native.models[name.startsWith('linear') ? 'linear_original' : 'mini_original'];
  const objective = value => (value.validation.original_span_subset.macro_character_f1 + 1 - value.validation.rated_non_toxic_subset.post_level_false_positive_rate) / 2;
  assert.ok(objective(reference) > objective(original), `${name}: balanced validation objective did not improve.`);
  assert.equal(evaluation.models[name].original_span_subset.examples, 2000);
  assert.equal(evaluation.models[name].rated_non_toxic_subset.examples, 5000);
  assert.ok(Math.abs(evaluation.models[name].browser_minus_native_macro_f1) < 0.01, `${name}: unexplained backend difference.`);
  assert.ok(challenge.models[name], `${name}: missing diagnostics.`);
  const timing = benchmark.models[name].benchmarks.find(value => value.nominal_input_tokens === 128);
  assert.ok(timing.worker.p95_ms < 100, `${name}: warm latency target missed.`);
  const assets = evaluation.assets.filter(asset => asset.model === name);
  assert.equal(assets.length, 2);
  for (const asset of assets) {
    assert.equal(hash(`${source}/${asset.filename}`), asset.sha256);
    assert.ok(benchmark.assets.some(other => other.model === name && other.filename === asset.filename && other.sha256 === asset.sha256));
    assert.ok(challenge.assets.some(other => other.model === name && other.filename === asset.filename && other.sha256 === asset.sha256));
  }
  const weights = assets.find(asset => !asset.filename.endsWith('.json'));
  assert.ok(weights.bytes < 30_000_000);
  const target = `docs/models/${name}`;
  mkdirSync(target, { recursive: true });
  for (const asset of assets) copyFileSync(`${source}/${asset.filename}`, `${target}/${asset.filename}`);
  if (!name.startsWith('linear')) copyFileSync(`${source}/config.json`, `${target}/config.json`);
  models.push({ id: name, label: name.startsWith('linear') ? 'Fastest · balanced character-context · 64 KiB weights' : 'Contextual · balanced BERT Mini INT8 · 10.8 MiB weights', weight_bytes: weights.bytes, assets, browser_report: 'balanced-browser-evaluation.json', benchmark_report: 'balanced-browser-benchmark.json', diagnostic_report: 'balanced-challenge.json' });
}
const release = {
  version: 2, mode: 'explicit opt-in review only', automatic_filter_gate_passed: false,
  memory_gate_passed: false, worker_sha256: workerHash,
  dataset_profile: 'balanced', models,
  limitations: ['Automatic precision and independently low benign-false-positive gates remain unmet.', 'No full native/process memory bound is claimed.', 'Quoted words remain difficult; diagnostics must not be described as solved.', 'No OCR, video or audio censorship.'],
  legacy: 'Original weights and first-release reports remain available for research comparison; they are not the selected default models.',
};
writeFileSync('docs/reports/release.json', JSON.stringify(release, null, 2) + '\n');
console.log('Promoted verified candidates to opt-in REVIEW ONLY. No production safety gate is claimed.');
