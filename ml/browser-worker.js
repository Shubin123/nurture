import * as ort from 'onnxruntime-web/wasm';
import { LinearModel, WordPiece, chunksFor, mergeWordPieces } from './browser-runtime.js';

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL('vendor/', self.location.href).href;
let model = null;
let activeName = null;
let queue = Promise.resolve();

async function fetchAsset(path, type = 'json') {
  const response = await fetch(new URL(path, self.location.href));
  if (!response.ok) throw new Error(`Could not load ${path}: ${response.status}`);
  return response[type]();
}

async function load(name) {
  if (!['linear', 'bert-tiny', 'bert-mini', 'linear-balanced', 'bert-mini-balanced'].includes(name)) throw new Error('Unknown model.');
  if (activeName === name) return { cached: true, model: name, threshold: model.config.threshold };
  if (model?.session) await model.session.release();
  model = null;
  activeName = null;
  const started = performance.now();
  if (name.startsWith('linear')) {
    const [config, buffer] = await Promise.all([fetchAsset(`models/${name}/config.json`), fetchAsset(`models/${name}/weights.int8`, 'arrayBuffer')]);
    model = new LinearModel(config, new Int8Array(buffer));
  } else {
    const [config, vocabulary, buffer] = await Promise.all([fetchAsset(`models/${name}/inference.json`), fetchAsset('models/vocab.txt', 'text'), fetchAsset(`models/${name}/model.int8.onnx`, 'arrayBuffer')]);
    const session = await ort.InferenceSession.create(buffer, { executionProviders: ['wasm'], graphOptimizationLevel: 'all', freeDimensionOverrides: { sequence: 128 } });
    model = { config, tokenizer: new WordPiece(vocabulary), session };
  }
  activeName = name;
  return { model: name, load_ms: performance.now() - started, threshold: model.config.threshold, runtime: name.startsWith('linear') ? 'JavaScript CPU; no WASM initialization or remote inference' : 'single-threaded CPU / WASM; no remote inference' };
}

async function neuralPredict(text, batchSize = 8) {
  const tokens = model.tokenizer.tokenize(text);
  const chunks = chunksFor(tokens);
  const sums = new Float64Array(tokens.length);
  const counts = new Uint16Array(tokens.length);
  for (let start = 0; start < chunks.length; start += batchSize) {
    const batch = chunks.slice(start, start + batchSize);
    const ids = new BigInt64Array(batch.length * 128);
    const masks = new BigInt64Array(ids.length);
    for (let index = 0; index < batch.length; index += 1) {
      const offset = index * 128;
      const chunk = batch[index];
      ids[offset] = 101n;
      masks[offset] = 1n;
      for (let token = 0; token < chunk.tokens.length; token += 1) {
        ids[offset + token + 1] = BigInt(chunk.tokens[token].id);
        masks[offset + token + 1] = 1n;
      }
      ids[offset + chunk.tokens.length + 1] = 102n;
      masks[offset + chunk.tokens.length + 1] = 1n;
    }
    const outputs = await model.session.run({ input_ids: new ort.Tensor('int64', ids, [batch.length, 128]), attention_mask: new ort.Tensor('int64', masks, [batch.length, 128]) });
    const values = outputs.logits.data;
    for (let index = 0; index < batch.length; index += 1) {
      const chunk = batch[index];
      for (let token = 0; token < chunk.tokens.length; token += 1) {
        const offset = (index * 128 + token + 1) * 2;
        const probability = 1 / (1 + Math.exp(Math.max(-50, Math.min(50, values[offset] - values[offset + 1]))));
        sums[chunk.start + token] += probability;
        counts[chunk.start + token] += 1;
      }
    }
    Object.values(outputs).forEach(output => output.dispose());
  }
  return tokens.map((token, index) => ({ ...token, score: sums[index] / counts[index] }));
}

async function predict(text, batchSize = model?.config.batch_size || 8) {
  if (!model) throw new Error('Load a model first.');
  if (typeof text !== 'string' || text.length > 50000) throw new Error('Use at most 50,000 characters.');
  const started = performance.now();
  const tokens = activeName.startsWith('linear') ? model.predict(text) : await neuralPredict(text, batchSize);
  return { model: activeName, tokens, words: activeName.startsWith('linear') ? tokens : mergeWordPieces(text, tokens), threshold: model.config.threshold, elapsed_ms: performance.now() - started };
}

self.onmessage = event => {
  const { id, action, name, text, batchSize } = event.data;
  queue = queue.then(async () => {
    try {
      const result = action === 'load' ? await load(name) : action === 'predict' ? await predict(text, batchSize) : null;
      if (!result) throw new Error('Unknown action.');
      self.postMessage({ id, result });
    } catch (error) {
      self.postMessage({ id, error: error.message });
    }
  });
};
