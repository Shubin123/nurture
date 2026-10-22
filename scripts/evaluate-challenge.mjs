import { chromium } from 'playwright';
import { readFileSync, writeFileSync } from 'node:fs';

const challenge = JSON.parse(readFileSync('ml/challenge.json', 'utf8'));
const report = { provenance: challenge.provenance, annotation: challenge.annotation, models: {} };
const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: true });
try {
  const page = await browser.newPage();
  await page.goto(process.env.BASE_URL || 'http://localhost:8000');
  for (const name of ['linear', 'bert-tiny', 'bert-mini']) {
    const predictions = await page.evaluate(async ({ name, examples }) => {
      const worker = new Worker('model-worker.js', { type: 'module' });
      let id = 0;
      const pending = new Map();
      worker.onmessage = event => {
        const request = pending.get(event.data.id);
        pending.delete(event.data.id);
        if (event.data.error) request.reject(new Error(event.data.error));
        else request.resolve(event.data.result);
      };
      worker.onerror = event => { for (const request of pending.values()) request.reject(new Error(event.message)); };
      const call = (action, values) => new Promise((resolve, reject) => {
        const request = ++id;
        pending.set(request, { resolve, reject });
        worker.postMessage({ id: request, action, ...values });
      });
      try {
        await call('load', { name });
        const predictions = [];
        for (const example of examples) {
          const result = await call('predict', { text: example.text });
          predictions.push(result.words.filter(word => word.score >= result.threshold));
        }
        return predictions;
      } finally {
        worker.terminate();
      }
    }, { name, examples: challenge.examples });
    const slices = {};
    for (const [index, example] of challenge.examples.entries()) {
      const predicted = predictions[index];
      const slice = slices[example.slice] ??= { examples: 0, examples_flagged: 0, expected_positive_examples: 0, expected_positive_examples_flagged: 0 };
      slice.examples += 1;
      slice.examples_flagged += predicted.length > 0 ? 1 : 0;
      slice.expected_positive_examples += example.terms.length > 0 ? 1 : 0;
      slice.expected_positive_examples_flagged += example.terms.length > 0 && predicted.length > 0 ? 1 : 0;
    }
    report.models[name] = { slices, examples: challenge.examples.map((example, index) => ({ ...example, predicted: predictions[index] })) };
  }
  writeFileSync('docs/reports/challenge.json', JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify(Object.fromEntries(Object.entries(report.models).map(([name, value]) => [name, value.slices])), null, 2));
} finally {
  await browser.close();
}
