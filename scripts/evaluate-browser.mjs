import { chromium } from 'playwright';
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';

const rows = readFileSync('ml/data/test.jsonl', 'utf8').trim().split('\n').map(line => JSON.parse(line));
const predictions = {};
const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: true });
try {
  const page = await browser.newPage();
  await page.goto(process.env.BASE_URL || 'http://localhost:8000');
  await page.evaluate(() => {
    window.evaluationWorker = new Worker('model-worker.js', { type: 'module' });
    const pending = new Map();
    let id = 0;
    window.evaluationWorker.onmessage = event => {
      const entry = pending.get(event.data.id);
      pending.delete(event.data.id);
      if (event.data.error) entry.reject(new Error(event.data.error));
      else entry.resolve(event.data.result);
    };
    window.evaluationWorker.onerror = event => { for (const entry of pending.values()) entry.reject(new Error(event.message)); };
    window.evaluateCall = (action, values) => new Promise((resolve, reject) => {
      const request = ++id;
      pending.set(request, { resolve, reject });
      window.evaluationWorker.postMessage({ id: request, action, ...values });
    });
  });
  for (const name of ['linear', 'bert-tiny', 'bert-mini']) {
    await page.evaluate(name => window.evaluateCall('load', { name }), name);
    const model = [];
    for (let index = 0; index < rows.length; index += 1) {
      const row = rows[index];
      const result = await page.evaluate(text => window.evaluateCall('predict', { text }), row.text);
      const codepoints = new Map();
      let offset = 0;
      for (const [position, letter] of [...row.text].entries()) {
        codepoints.set(offset, position);
        offset += letter.length;
      }
      codepoints.set(offset, [...row.text].length);
      const predicted = new Set();
      for (const token of result.tokens) {
        if (token.score < result.threshold) continue;
        for (let position = codepoints.get(token.start); position < codepoints.get(token.end); position += 1) predicted.add(position);
      }
      const merged = new Set();
      for (const word of result.words) {
        if (word.score < result.threshold) continue;
        for (let position = codepoints.get(word.start); position < codepoints.get(word.end); position += 1) merged.add(position);
      }
      model.push({ id: row.id, predicted: [...predicted].sort((left, right) => left - right), merged_word_predictions: [...merged].sort((left, right) => left - right), tokens: result.tokens.length, elapsed_ms: result.elapsed_ms, threshold: result.threshold });
      if ((index + 1) % 500 === 0) console.log(name, index + 1, '/', rows.length);
    }
    predictions[name] = model;
    writeFileSync('ml/runs/browser-predictions.json', JSON.stringify({ worker_sha256: createHash('sha256').update(readFileSync('docs/model-worker.js')).digest('hex'), browser: await browser.version(), models: predictions }));
  }
} finally {
  await browser.close();
}
