import { chromium } from 'playwright';
import { readFileSync, writeFileSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import os from 'node:os';
import { createHash } from 'node:crypto';
import { WordPiece, linearTokens } from '../ml/browser-runtime.js';
import { modelNames, prefix, profileEvidence, configureProfile, isLinear } from './model-profile.mjs';

const results = { worker_sha256: createHash('sha256').update(readFileSync('docs/model-worker.js')).digest('hex'), hardware: os.cpus()[0].model, architecture: os.arch(), os: `${os.type()} ${os.release()}`, headless: true, runtime: JSON.parse(readFileSync('node_modules/onnxruntime-web/package.json')).version, threads: 1, network: 'localhost, warm HTTP cache per model, no artificial throttling', scope: 'End-to-end worker text tokenization + inference + score aggregation. Worker processing excludes message serialization; roundtrip is reported separately. Inference uses overlap; useful input tokens exclude duplicates and padding.', models: {} };
const sleep = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));
Object.assign(results, profileEvidence());
results.network = 'Localhost asset requests; balanced candidate weights/config are supplied by local request interception. No artificial throttling. Initialization is not an Internet download estimate; inference timing follows five warmups.';
function stats(values) {
  const sorted = [...values].sort((left, right) => left - right);
  return { runs: values.length, mean_ms: values.reduce((total, value) => total + value, 0) / values.length, p50_ms: sorted[Math.floor(sorted.length * 0.5)], p95_ms: sorted[Math.floor(sorted.length * 0.95)] };
}

for (const name of modelNames) {
  const server = await chromium.launchServer({ executablePath: process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: true });
  const browser = await chromium.connect(server.wsEndpoint());
  let monitor;
  try {
    const page = await browser.newPage();
    await configureProfile(page.context());
    await page.goto(process.env.BASE_URL || 'http://localhost:8000');
    await sleep(500);
    let monitorOutput = '';
    monitor = spawn('.venv/bin/python', ['ml/sample_memory.py', String(server.process().pid)], { stdio: ['ignore', 'pipe', 'inherit'] });
    monitor.stdout.on('data', data => { monitorOutput += data.toString(); });
    await sleep(200);
    await page.evaluate(() => {
      window.labWorker = new Worker('model-worker.js', { type: 'module' });
      let id = 0;
      const pending = new Map();
      window.labWorker.onmessage = event => {
        const entry = pending.get(event.data.id);
        if (!entry) return;
        pending.delete(event.data.id);
        if (event.data.error) entry.reject(new Error(event.data.error));
        else entry.resolve(event.data.result);
      };
      window.labWorker.onerror = event => { for (const entry of pending.values()) entry.reject(new Error(event.message)); pending.clear(); };
      window.callWorker = (action, values = {}) => new Promise((resolve, reject) => { const request = ++id; pending.set(request, { resolve, reject }); window.labWorker.postMessage({ id: request, action, ...values }); });
    });
    const loaded = await page.evaluate(name => window.callWorker('load', { name }), name);
    const benchmarks = [];
    const vocabulary = new WordPiece(readFileSync('docs/models/vocab.txt', 'utf8'));
    const sourceText = 'Please discuss this idea politely instead of calling another person stupid or an idiot. '.repeat(100);
    const sourceTokens = isLinear(name) ? linearTokens(sourceText) : vocabulary.tokenize(sourceText);
    for (const count of [126, 128, 1024]) {
      const input = sourceText.slice(0, sourceTokens[count - 1].end);
      const measured = await page.evaluate(async text => {
        for (let warm = 0; warm < 5; warm += 1) await window.callWorker('predict', { text });
        const workerTimes = [], roundtripTimes = [];
        let tokenCount = 0;
        for (let run = 0; run < 50; run += 1) {
          const started = performance.now();
          const result = await window.callWorker('predict', { text });
          roundtripTimes.push(performance.now() - started);
          workerTimes.push(result.elapsed_ms);
          tokenCount = result.tokens.length;
        }
        return { text_characters: text.length, useful_tokens: tokenCount, workerTimes, roundtripTimes };
      }, input);
      const worker = stats(measured.workerTimes);
      benchmarks.push({ nominal_input_tokens: count, text_characters: measured.text_characters, useful_tokens: measured.useful_tokens, worker, roundtrip: stats(measured.roundtripTimes), useful_tokens_per_second: measured.useful_tokens / (worker.mean_ms / 1000) });
    }
    const batchSweep = [];
    if (!isLinear(name)) {
      for (const batchSize of [1, 4, 8, 16]) {
        const text = sourceText.slice(0, sourceTokens[1023].end);
        const values = await page.evaluate(async ({ text, batchSize }) => {
          for (let warm = 0; warm < 3; warm += 1) await window.callWorker('predict', { text, batchSize });
          const values = [];
          for (let run = 0; run < 20; run += 1) values.push((await window.callWorker('predict', { text, batchSize })).elapsed_ms);
          return values;
        }, { text, batchSize });
        const measured = stats(values);
        batchSweep.push({ batch_size: batchSize, ...measured, useful_tokens_per_second: 1024 / (measured.mean_ms / 1000) });
      }
    }
    const responsive = await page.evaluate(async () => {
      let ticks = 0;
      const timer = setInterval(() => { ticks += 1; }, 5);
      await window.callWorker('predict', { text: 'Please discuss ideas politely. '.repeat(1000) });
      clearInterval(timer);
      return { main_thread_timer_ticks_during_30000_character_job: ticks };
    });
    monitor.kill('SIGTERM');
    await once(monitor, 'exit');
    monitor = null;
    results.models[name] = { load: loaded, browser: await browser.version(), benchmarks, batch_sweep: batchSweep, responsiveness: responsive, memory_proxy: JSON.parse(monitorOutput) };
    console.log(name, JSON.stringify(results.models[name]));
    writeFileSync(`docs/reports/${prefix}browser-benchmark.json`, JSON.stringify(results, null, 2) + '\n');
  } finally {
    if (monitor) monitor.kill('SIGTERM');
    await browser.close();
    await server.close();
  }
}
