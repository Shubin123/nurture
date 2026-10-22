import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';

export const profile = process.env.MODEL_PROFILE || 'deployed';
if (!['deployed', 'balanced'].includes(profile)) throw new Error('MODEL_PROFILE must be deployed or balanced.');
const release = existsSync('docs/reports/release.json') ? JSON.parse(readFileSync('docs/reports/release.json', 'utf8')) : null;
export const modelNames = profile === 'balanced' ? ['linear-balanced', 'bert-mini-balanced'] : release ? release.models.map(model => model.id) : ['linear', 'bert-tiny', 'bert-mini'];
export const prefix = profile === 'balanced' ? 'balanced-' : release ? 'release-' : '';
export const dataPath = profile === 'balanced' || release ? 'ml/data/balanced/test.jsonl' : 'ml/data/test.jsonl';
export const isLinear = name => name.startsWith('linear');
export const fingerprint = path => createHash('sha256').update(readFileSync(path)).digest('hex');

export function profileEvidence() {
  return {
    profile,
    assets: modelNames.flatMap(name => {
      const directory = profile === 'balanced' ? `ml/runs/candidate-models/${name}` : `docs/models/${name}`;
      return (isLinear(name) ? ['config.json', 'weights.int8'] : ['inference.json', 'model.int8.onnx']).map(filename => {
        const path = `${directory}/${filename}`;
        return { model: name, filename, bytes: readFileSync(path).length, sha256: fingerprint(path) };
      });
    }),
  };
}

export async function configureProfile(context) {
  if (profile !== 'balanced') return;
  assert.equal(fingerprint('ml/runs/candidate-models/vocab.txt'), fingerprint('docs/models/vocab.txt'), 'Candidate vocabulary differs from the served tokenizer.');
  for (const name of modelNames) {
    for (const filename of isLinear(name) ? ['config.json', 'weights.int8'] : ['inference.json', 'model.int8.onnx']) {
      const path = `ml/runs/candidate-models/${name}/${filename}`;
      await context.route(`**/models/${name}/${filename}`, route => route.fulfill({ body: readFileSync(path), contentType: filename.endsWith('.json') ? 'application/json' : 'application/octet-stream' }));
    }
  }
}
