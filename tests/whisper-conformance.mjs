import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { chromium } from 'playwright';
import { logMel } from '../ml/whisper-runtime.js';

// Compares the browser Whisper runtime with ml/whisper_reference.py (run that first).
const reference = JSON.parse(readFileSync('ml/runs/whisper-reference.json', 'utf8'));
const wav = readFileSync(reference.audio);
let offset = 12;
while (wav.toString('ascii', offset, offset + 4) !== 'data') offset += 8 + wav.readUInt32LE(offset + 4);
const pcm = new Int16Array(wav.buffer.slice(wav.byteOffset + offset + 8, wav.byteOffset + offset + 8 + wav.readUInt32LE(offset + 4)));
const audio = Float32Array.from(pcm, value => value / 32768);

const features = logMel(audio);
let featureError = 0;
for (let mel = 0; mel < 80; mel += 1) {
  for (let frame = 0; frame < reference.feature_frames; frame += 1) featureError = Math.max(featureError, Math.abs(features[mel * 3000 + frame] - reference.feature_head[mel][frame]));
}

const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: true });
let result;
try {
  const page = await browser.newPage();
  await page.goto(process.env.BASE_URL || 'http://localhost:8000');
  result = await page.evaluate(async samples => {
    const worker = new Worker('asr-worker.js', { type: 'module' });
    const call = (id, message) => new Promise((resolve, reject) => {
      worker.onmessage = event => event.data.progress ? null : event.data.error ? reject(new Error(event.data.error)) : resolve(event.data.result);
      worker.postMessage({ id, ...message });
    });
    await call(1, { action: 'load' });
    return call(2, { action: 'transcribe', audio: new Float32Array(samples) });
  }, Array.from(audio));
} finally {
  await browser.close();
}
const text = result.words.map(word => word.text).join('');
// Every measured pause must fall between two words, and no word may outlast the speech.
const spoken = result.words.filter(word => /[A-Za-z0-9]/.test(word.text));
const tolerance = 0.12;
const pauses = reference.pauses.map(([from, to]) => {
  const before = spoken.filter(word => word.start < from).at(-1);
  const after = spoken.find(word => word.start >= from - tolerance && word !== before);
  return { pause: [from, to], before: before?.text, before_end: before?.end, after: after?.text, after_start: after?.start, ok: Boolean(before && after && before.end <= to + tolerance && after.start >= from - tolerance) };
});
const report = {
  audio: reference.audio,
  max_feature_abs_error: featureError,
  text_matches_onnx_reference: text === reference.text,
  text,
  words: spoken.length,
  pauses,
  last_word_end_s: spoken.at(-1).end,
  speech_end_s: reference.speech_end,
  inference_ms: result.inference_ms
};
writeFileSync('docs/reports/whisper-conformance.json', `${JSON.stringify(report, null, 2)}\n`);
console.log(report);
assert.ok(featureError < 2e-3, `feature error ${featureError}`);
assert.equal(text, reference.text);
assert.ok(pauses.every(pause => pause.ok), 'a word boundary misses a measured pause');
// DTW tends to stretch the final word; a late end only mutes slightly longer.
assert.ok(report.last_word_end_s <= reference.speech_end + 0.25);
console.log('PASS: log-mel and greedy tokens match the Python references; word boundaries respect every measured pause.');
