import * as ort from 'onnxruntime-web/wasm';
import { ByteDecoder, SAMPLE_RATE, alignmentTimes, logMel, splitWords } from './whisper-runtime.js';

// Offline only: refuse every request that is not to this site's own files.
const localFetch = self.fetch.bind(self);
self.fetch = (input, init) => {
  const url = new URL(input instanceof Request ? input.url : String(input), self.location.href);
  if (url.origin !== self.location.origin) return Promise.reject(new Error(`Blocked non-local request: ${url.origin}`));
  return localFetch(input, init);
};

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL('vendor/', self.location.href).href;
const MODEL = 'models/whisper-tiny-en/';
const LAYERS = 4;
const HEADS = 6;
let whisper = null;

// Pinned files (see docs/reports/asr-model.json); bump CACHE when any of them changes.
const CACHE = 'nurture-speech-aeaa137-ort1.30.0';
const FILES = [
  [MODEL + 'generation_config.json', 1646],
  [MODEL + 'vocab.json', 999186],
  [MODEL + 'onnx/encoder_model_quantized.onnx', 10097112],
  [MODEL + 'onnx/decoder_model_merged_quantized.onnx', 30729881],
  ['vendor/ort-wasm-simd-threaded.wasm', 14239897]
];

// Kept in the browser's Cache Storage after the first download, so later visits load offline.
async function fetchAll(progress) {
  let store = null;
  try {
    store = await caches.open(CACHE);
    for (const name of await caches.keys()) if (name.startsWith('nurture-speech-') && name !== CACHE) await caches.delete(name);
  } catch {
    store = null;
  }
  const total = FILES.reduce((sum, [, bytes]) => sum + bytes, 0);
  const loaded = FILES.map(() => 0);
  let cached = 0;
  const report = () => progress({ loaded: loaded.reduce((a, b) => a + b, 0), total, cached: cached === FILES.length });
  return Promise.all(FILES.map(async ([path, bytes], index) => {
    const url = new URL(path, self.location.href).href;
    const hit = await store?.match(url).catch(() => null);
    if (hit) {
      cached += 1;
      const buffer = await hit.arrayBuffer();
      loaded[index] = bytes;
      report();
      return buffer;
    }
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Could not load ${path}: ${response.status}`);
    const reader = response.body.getReader();
    const chunks = [];
    let size = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      size += value.length;
      loaded[index] = Math.min(size, bytes);
      report();
    }
    const buffer = new Uint8Array(size);
    let offset = 0;
    for (const chunk of chunks) {
      buffer.set(chunk, offset);
      offset += chunk.length;
    }
    await store?.put(url, new Response(buffer)).catch(() => {});
    return buffer.buffer;
  }));
}

async function load(progress) {
  if (whisper) return { cached: true };
  const started = performance.now();
  const options = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
  const [generationBytes, vocabularyBytes, encoderBytes, decoderBytes, wasmBytes] = await fetchAll(progress);
  const text = new TextDecoder();
  const generation = JSON.parse(text.decode(generationBytes));
  const vocabulary = JSON.parse(text.decode(vocabularyBytes));
  progress({ preparing: true });
  ort.env.wasm.wasmBinary = wasmBytes;
  const encoder = await ort.InferenceSession.create(encoderBytes, options);
  const decoder = await ort.InferenceSession.create(decoderBytes, options);
  const eot = generation.eos_token_id;
  const suppressed = new Set([...generation.suppress_tokens]);
  whisper = { encoder, decoder, generation, eot, suppressed, decoder_text: new ByteDecoder(vocabulary) };
  return { load_ms: performance.now() - started };
}

function emptyPast() {
  const feeds = {};
  for (let layer = 0; layer < LAYERS; layer += 1) {
    for (const kind of ['decoder', 'encoder']) for (const part of ['key', 'value']) {
      feeds[`past_key_values.${layer}.${kind}.${part}`] = new ort.Tensor('float32', new Float32Array(0), [1, HEADS, 0, 64]);
    }
  }
  return feeds;
}

function ids(values) {
  return new ort.Tensor('int64', BigInt64Array.from(values, BigInt), [1, values.length]);
}

function pick(logits, step) {
  const vocabulary = logits.dims[2];
  const offset = (logits.dims[1] - 1) * vocabulary;
  let best = whisper.eot;
  let bestScore = -Infinity;
  // Text tokens only (below end-of-text), plus end-of-text after the first token.
  for (let id = 0; id <= whisper.eot; id += 1) {
    if (whisper.suppressed.has(id) || (step === 0 && whisper.generation.begin_suppress_tokens.includes(id))) continue;
    const score = logits.data[offset + id];
    if (score > bestScore) [best, bestScore] = [id, score];
  }
  return best;
}

async function transcribe(audio) {
  if (!whisper) throw new Error('Speech model is not loaded.');
  const started = performance.now();
  const { encoder, decoder, generation, eot } = whisper;
  const features = new ort.Tensor('float32', logMel(audio), [1, 80, 3000]);
  const { last_hidden_state: hidden } = await encoder.run({ input_features: features });
  const prompt = [generation.decoder_start_token_id, generation.no_timestamps_token_id];
  const limit = Math.min(224, Math.ceil(audio.length / SAMPLE_RATE * 8) + 8);
  const text = [];
  let feeds = { input_ids: ids(prompt), encoder_hidden_states: hidden, use_cache_branch: new ort.Tensor('bool', [false], [1]), ...emptyPast() };
  const encoderCache = {};
  for (let step = 0; step < limit; step += 1) {
    const output = await decoder.run(feeds);
    // Cached steps return empty encoder presents; keep the ones from the first step.
    if (step === 0) for (let layer = 0; layer < LAYERS; layer += 1) for (const part of ['key', 'value']) encoderCache[`past_key_values.${layer}.encoder.${part}`] = output[`present.${layer}.encoder.${part}`];
    const token = pick(output.logits, step);
    if (token === eot) break;
    text.push(token);
    feeds = { input_ids: ids([token]), encoder_hidden_states: hidden, use_cache_branch: new ort.Tensor('bool', [true], [1]), ...encoderCache };
    for (let layer = 0; layer < LAYERS; layer += 1) for (const part of ['key', 'value']) feeds[`past_key_values.${layer}.decoder.${part}`] = output[`present.${layer}.decoder.${part}`];
  }
  if (!text.length) return { words: [], inference_ms: performance.now() - started };

  // One uncached pass over the whole sequence gives cross-attention for every token.
  const sequence = [...prompt, ...text, eot];
  const aligned = await decoder.run({ input_ids: ids(sequence), encoder_hidden_states: hidden, use_cache_branch: new ort.Tensor('bool', [false], [1]), ...emptyPast() });
  const frames = Math.max(1, Math.min(1500, Math.floor(audio.length / 160 / 2)));
  const heads = generation.alignment_heads.map(([layer, head]) => {
    const attention = aligned[`cross_attentions.${layer}`];
    const source = attention.dims[3];
    const weights = new Float32Array(sequence.length * frames);
    for (let t = 0; t < sequence.length; t += 1) {
      const row = (head * sequence.length + t) * source;
      weights.set(attention.data.subarray(row, row + frames), t * frames);
    }
    return weights;
  });
  const jumps = alignmentTimes(heads, sequence.length, frames);
  const words = [];
  let boundary = 0;
  for (const word of splitWords(whisper.decoder_text, [...text, eot], eot)) {
    if (word.ids[0] >= eot) break;
    const next = boundary + word.ids.length;
    words.push({ text: word.text, start: jumps[boundary], end: jumps[Math.min(next, jumps.length - 1)] });
    boundary = next;
  }
  return { words, inference_ms: performance.now() - started };
}

self.onmessage = async event => {
  const { id, action } = event.data;
  try {
    const result = action === 'load' ? await load(progress => self.postMessage({ id, progress })) : action === 'transcribe' ? await transcribe(event.data.audio) : null;
    if (!result) throw new Error('Unknown action.');
    self.postMessage({ id, result });
  } catch (error) {
    self.postMessage({ id, error: error.message });
  }
};
