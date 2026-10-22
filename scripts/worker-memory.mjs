import { chromium } from 'playwright';
import { writeFileSync } from 'node:fs';
import { modelNames, prefix, profileEvidence, configureProfile, fingerprint } from './model-profile.mjs';

const report = { ...profileEvidence(), worker_sha256: fingerprint('docs/model-worker.js'), scope: 'Experimental Chrome Runtime.getHeapUsage for the model worker: V8 heap, embedder and backing storage. Not a complete process/native allocation measurement, a sampled bound or a portable memory guarantee.', models: {} };
for (const name of modelNames) {
  const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: true });
  try {
    const page = await browser.newPage();
    await configureProfile(page.context());
    await page.goto(process.env.BASE_URL || 'http://localhost:8000');
    await page.evaluate(() => {
      window.memoryWorker = new Worker('model-worker.js', { type: 'module' });
      let id = 0;
      const pending = new Map();
      window.memoryWorker.onmessage = event => {
        const entry = pending.get(event.data.id);
        pending.delete(event.data.id);
        if (event.data.error) entry.reject(new Error(event.data.error));
        else entry.resolve(event.data.result);
      };
      window.memoryCall = (action, values) => new Promise((resolve, reject) => { const request = ++id; pending.set(request, { resolve, reject }); window.memoryWorker.postMessage({ id: request, action, ...values }); });
    });
    await page.waitForFunction(() => window.memoryWorker !== undefined);
    const connection = await browser.newBrowserCDPSession();
    const { targetInfos } = await connection.send('Target.getTargets');
    const target = targetInfos.find(target => target.type === 'worker' && target.url.endsWith('/model-worker.js'));
    if (!target) throw new Error('Model worker target is unavailable.');
    const { sessionId } = await connection.send('Target.attachToTarget', { targetId: target.targetId, flatten: false });
    let requestId = 0;
    const pending = new Map();
    connection.on('Target.receivedMessageFromTarget', event => {
      if (event.sessionId !== sessionId) return;
      const message = JSON.parse(event.message);
      const entry = pending.get(message.id);
      if (!entry) return;
      pending.delete(message.id);
      clearTimeout(entry.timeout);
      if (message.error) entry.reject(new Error(message.error.message));
      else entry.resolve(message.result);
    });
    const command = method => new Promise((resolve, reject) => {
      const id = ++requestId;
      const timeout = setTimeout(() => { pending.delete(id); reject(new Error('CDP memory sampling timed out.')); }, 10000);
      pending.set(id, { resolve, reject, timeout });
      connection.send('Target.sendMessageToTarget', { sessionId, message: JSON.stringify({ id, method }) }).catch(reject);
    });
    const samples = [];
    samples.push({ stage: 'worker_before_model_load', ...await command('Runtime.getHeapUsage') });
    await page.evaluate(name => window.memoryCall('load', { name }), name);
    samples.push({ stage: 'after_model_load', ...await command('Runtime.getHeapUsage') });
    for (const repetitions of [5, 40, 1000]) {
      await page.evaluate(repetitions => window.memoryCall('predict', { text: 'Please discuss ideas politely instead of calling people stupid. '.repeat(repetitions).slice(0, 50000) }), repetitions);
      samples.push({ stage: `after_${repetitions}_sentence_repetitions`, ...await command('Runtime.getHeapUsage') });
    }
    await command('HeapProfiler.collectGarbage');
    samples.push({ stage: 'after_forced_gc_resident_only', ...await command('Runtime.getHeapUsage') });
    report.models[name] = { browser: await browser.version(), samples };
    console.log(name, JSON.stringify(samples));
    await connection.send('Target.detachFromTarget', { sessionId });
  } finally {
    await browser.close();
  }
}
writeFileSync(`docs/reports/${prefix}worker-memory.json`, JSON.stringify(report, null, 2) + '\n');
