import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';

const hash = path => createHash('sha256').update(readFileSync(path)).digest('hex');
const manifest = JSON.parse(readFileSync('docs/reports/artifacts.json'));
for (const asset of manifest.assets) {
  const path = `docs/${asset.path}`;
  assert.equal(hash(path), asset.sha256, path);
  assert.equal(readFileSync(path).length, asset.bytes, path);
}
for (const path of ['docs/reports/browser-benchmark.json', 'docs/reports/browser-evaluation.json']) {
  assert.equal(JSON.parse(readFileSync(path)).worker_sha256, hash('docs/model-worker.js'), path);
}
const evaluation = JSON.parse(readFileSync('docs/reports/evaluation.json'));
for (const size of ['tiny', 'mini']) assert.equal(hash(`docs/models/bert-${size}/model.int8.onnx`), evaluation.models[`bert_${size}_int8`].sha256);
assert.equal(JSON.parse(readFileSync('docs/reports/frozen-selection.json')).test_used_for_selection, false);
console.log('PASS: all model/runtime hashes and browser-worker report fingerprints match the release artifacts.');
