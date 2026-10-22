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
const release = JSON.parse(readFileSync('docs/reports/release.json'));
assert.equal(release.worker_sha256, hash('docs/model-worker.js'));
assert.equal(release.automatic_filter_gate_passed, false);
for (const path of ['docs/reports/balanced-browser-benchmark.json', 'docs/reports/balanced-browser-evaluation.json']) {
  assert.equal(JSON.parse(readFileSync(path)).worker_sha256, hash('docs/model-worker.js'), path);
}
for (const model of release.models) {
  const report = JSON.parse(readFileSync(`docs/reports/${model.browser_report}`));
  assert.equal(report.held_out_examples, 7000);
  assert.equal(report.models[model.id].original_span_subset.examples, 2000);
  assert.equal(report.models[model.id].rated_non_toxic_subset.examples, 5000);
  for (const asset of model.assets) {
    const path = `models/${model.id}/${asset.filename}`;
    assert.equal(hash(`docs/${path}`), asset.sha256, path);
    assert.ok(manifest.assets.some(entry => entry.path === path && entry.sha256 === asset.sha256), path);
    assert.ok(report.assets.some(entry => entry.model === model.id && entry.filename === asset.filename && entry.sha256 === asset.sha256), path);
  }
}
const evaluation = JSON.parse(readFileSync('docs/reports/evaluation.json'));
for (const size of ['tiny', 'mini']) assert.equal(hash(`docs/models/bert-${size}/model.int8.onnx`), evaluation.models[`bert_${size}_int8`].sha256);
assert.equal(JSON.parse(readFileSync('docs/reports/frozen-selection.json')).test_used_for_selection, false);
assert.equal(JSON.parse(readFileSync('docs/reports/balanced-frozen-selection.json')).test_used_for_selection, false);
console.log('PASS: all model/runtime hashes and browser-worker report fingerprints match the release artifacts.');
