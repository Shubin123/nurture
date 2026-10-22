import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';

const read = path => JSON.parse(readFileSync(path, 'utf8'));
const before = read('ml/runs/balanced-pre-ascii-predictions.json');
const after = read('ml/runs/balanced-browser-predictions.json');
const models = {};
for (const [original, optimized] of [['linear', 'linear-balanced'], ['bert-mini', 'bert-mini-balanced']]) {
  const baseline = before.models[original];
  const current = after.models[optimized];
  assert.equal(current.length, 7000);
  assert.equal(baseline.length, current.length);
  for (let index = 0; index < current.length; index += 1) {
    for (const field of ['id', 'predicted', 'merged_word_predictions', 'tokens', 'threshold']) {
      assert.deepEqual(current[index][field], baseline[index][field], `${optimized}:${index}:${field}`);
    }
  }
  models[optimized] = { examples: current.length, prediction_mismatches: 0, display_mismatches: 0 };
}
const report = {
  optimization: 'ASCII FNV-1a fast path; UTF-8 fallback unchanged',
  before_worker_sha256: before.worker_sha256,
  after_worker_sha256: after.worker_sha256,
  models,
};
writeFileSync('docs/reports/hash-fast-path-equivalence.json', JSON.stringify(report, null, 2) + '\n');
console.log('PASS: zero prediction/display differences across 14,000 model-document pairs.');
