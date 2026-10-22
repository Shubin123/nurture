const loadButton = document.querySelector('#load-model');
const runButton = document.querySelector('#run-model');
const unloadButton = document.querySelector('#unload-model');
const modelSelect = document.querySelector('#model-choice');
const modelStatus = document.querySelector('#model-status');
const modelInput = document.querySelector('#model-text');
const modelOutput = document.querySelector('#model-output');
let worker = null;
let requestId = 0;
let generation = 0;
const pending = new Map();

function request(action, values = {}) {
  if (!worker) worker = new Worker('model-worker.js', { type: 'module' });
  worker.onmessage = event => {
    const entry = pending.get(event.data.id);
    if (!entry) return;
    pending.delete(event.data.id);
    clearTimeout(entry.timeout);
    if (event.data.error) entry.reject(new Error(event.data.error));
    else entry.resolve(event.data.result);
  };
  worker.onerror = () => unload('Model worker failed. Try reloading, or use the rules demo.');
  return new Promise((resolve, reject) => {
    const id = ++requestId;
    const timeout = setTimeout(() => unload('Model request timed out. Worker stopped; use the rules demo or try again.'), 120000);
    pending.set(id, { resolve, reject, timeout });
    worker.postMessage({ id, action, ...values });
  });
}

function unload(message = 'Model unloaded. No model inference is running.') {
  generation += 1;
  worker?.terminate();
  worker = null;
  for (const entry of pending.values()) {
    clearTimeout(entry.timeout);
    entry.reject(new Error('Model stopped.'));
  }
  pending.clear();
  loadButton.disabled = false;
  modelSelect.disabled = false;
  runButton.disabled = true;
  unloadButton.disabled = true;
  modelOutput.replaceChildren();
  modelStatus.textContent = message;
}

loadButton.addEventListener('click', async () => {
  if (!document.querySelector('#model-consent').checked) {
    modelStatus.textContent = 'Confirm the local model download first.';
    return;
  }
  const current = generation;
  loadButton.disabled = true;
  runButton.disabled = true;
  unloadButton.disabled = false;
  modelSelect.disabled = true;
  modelStatus.textContent = 'Downloading local weights and initializing the worker. First load includes the shared WASM runtime.';
  try {
    const result = await request('load', { name: modelSelect.value });
    if (current !== generation) return;
    modelStatus.textContent = `Loaded ${result.model} in ${(result.load_ms ?? 0).toFixed(0)} ms. Experimental review mode; text stays on this device.`;
    runButton.disabled = false;
  } catch (error) {
    if (current === generation) unload(`Could not load: ${error.message}`);
  }
});

runButton.addEventListener('click', async () => {
  const current = generation;
  runButton.disabled = true;
  modelStatus.textContent = 'Analyzing locally in a worker…';
  modelOutput.replaceChildren();
  try {
    const text = modelInput.value;
    const approvedRules = NurtureFilter.parseRules(document.querySelector('#rules').value);
    const replacements = new Map(approvedRules.map(rule => [rule.word, rule.replacement]));
    const result = await request('predict', { text });
    if (current !== generation) return;
    const fragment = document.createDocumentFragment();
    let position = 0;
    let flagged = 0;
    let replaced = 0;
    for (const word of result.words) {
      if (word.score < result.threshold || word.start < position) continue;
      fragment.append(document.createTextNode(text.slice(position, word.start)));
      const original = text.slice(word.start, word.end);
      const replacement = replacements.get(original.toLowerCase());
      if (replacement !== undefined) {
        fragment.append(document.createTextNode(replacement));
        const star = document.createElement('sup');
        star.textContent = '*';
        star.setAttribute('aria-label', 'word replaced');
        fragment.append(star);
        replaced += 1;
      } else {
        const highlight = document.createElement('mark');
        highlight.textContent = original;
        highlight.title = `Experimental model score ${word.score.toFixed(3)}; review suggestion, not a calibrated probability.`;
        fragment.append(highlight);
      }
      flagged += 1;
      position = word.end;
    }
    fragment.append(document.createTextNode(text.slice(position)));
    modelOutput.replaceChildren(fragment);
    modelStatus.textContent = `${result.elapsed_ms.toFixed(1)} ms worker processing · ${result.tokens.length} tokens · ${flagged} flagged · ${replaced} approved replacements. Highlights need review; scores are not safety guarantees.`;
  } catch (error) {
    if (current === generation) modelStatus.textContent = error.message;
  } finally {
    if (current === generation) runButton.disabled = false;
  }
});

unloadButton.addEventListener('click', () => unload());
document.querySelector('#model-consent').addEventListener('change', event => {
  if (!event.target.checked) unload('Consent withdrawn. Model worker stopped.');
});
window.addEventListener('pagehide', () => unload());

Promise.all(['reports/balanced-browser-evaluation.json', 'reports/balanced-browser-benchmark.json'].map(async path => {
  const response = await fetch(path);
  if (!response.ok) throw new Error('Published measurements are unavailable.');
  return response.json();
})).then(([evaluation, benchmark]) => {
  const fragment = document.createDocumentFragment();
  for (const name of ['linear-balanced', 'bert-mini-balanced']) {
    const accuracy = evaluation.models[name].original_span_subset.macro_character_f1;
    const falsePositiveRate = evaluation.models[name].rated_non_toxic_subset.post_level_false_positive_rate;
    const timings = benchmark.models[name].benchmarks;
    const latency = timings.find(item => item.nominal_input_tokens === 128).worker.p95_ms;
    const throughput = timings.find(item => item.nominal_input_tokens === 1024).useful_tokens_per_second;
    const row = document.createElement('tr');
    for (const value of [name, accuracy.toFixed(4), `${(falsePositiveRate * 100).toFixed(2)}%`, `${latency.toFixed(1)} ms`, Math.round(throughput).toLocaleString()]) {
      const cell = document.createElement('td');
      cell.textContent = value;
      row.append(cell);
    }
    fragment.append(row);
  }
  document.querySelector('#model-results').replaceChildren(fragment);
}).catch(() => {
  document.querySelector('#model-results').textContent = 'Measurements could not be loaded. Use the linked raw reports.';
});
