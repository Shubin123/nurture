import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { fingerprint } from './model-profile.mjs';

const read = path => JSON.parse(readFileSync(path, 'utf8'));
const candidate = read('ml/runs/balanced-browser-predictions.json');
const served = read('ml/runs/release-browser-predictions.json');
assert.equal(served.profile, 'deployed');
assert.equal(candidate.profile, 'balanced');
assert.equal(served.worker_sha256, fingerprint('docs/model-worker.js'));
assert.equal(served.worker_sha256, candidate.worker_sha256);
assert.deepEqual(served.assets, candidate.assets);
assert.deepEqual(Object.keys(served.models), Object.keys(candidate.models));
const models = {};
for (const [name, values] of Object.entries(served.models)) {
  assert.equal(values.length, 7000);
  assert.equal(candidate.models[name].length, values.length);
  for (let index = 0; index < values.length; index += 1) {
    for (const field of ['id', 'predicted', 'merged_word_predictions', 'tokens', 'threshold']) {
      assert.deepEqual(values[index][field], candidate.models[name][index][field], `${name}:${index}:${field}`);
    }
  }
  models[name] = { examples: values.length, prediction_mismatches: 0, display_mismatches: 0 };
}
writeFileSync('docs/reports/served-candidate-equivalence.json', JSON.stringify({
  worker_sha256: served.worker_sha256,
  assets: served.assets,
  scope: 'Actual served versioned assets, using the deployed profile without candidate request interception, compared with the frozen candidate browser run.',
  models,
}, null, 2) + '\n');
console.log('PASS: served assets reproduce every candidate prediction/display decision across 14,000 model-document pairs.');
