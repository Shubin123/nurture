// Word rules and cited word lists, shared by the viewer (index.html) and the lab (lab.html).
const rulesInput = document.querySelector('#rules');
const sample = document.querySelector('#sample');
const output = document.querySelector('#output');
const filterStatus = document.querySelector('#filter-status');
const lexiconList = document.querySelector('#lexicon-list');
const lexiconFile = document.querySelector('#lexicon-file');
const listReplacement = document.querySelector('#list-replacement');
const exceptionsInput = document.querySelector('#exceptions');
const lexicons = new Map();

function currentRules() {
  const replacement = listReplacement.value.trim().slice(0, 80) || 'bleep';
  const lists = [...lexicons.values()].filter(list => list.enabled).map(list => ({ entries: list.entries, replacement, source: list.id }));
  return NurtureFilter.combineRules(NurtureFilter.parseRules(rulesInput.value), lists, exceptionsInput.value);
}

function markedFragment(parts) {
  const fragment = document.createDocumentFragment();
  for (const part of parts) {
    fragment.append(document.createTextNode(part.text));
    if (part.changed) {
      const marker = document.createElement('sup');
      marker.textContent = '*';
      marker.title = part.source === 'rules' ? 'Word replaced by your local rules' : `Matched word list: ${lexicons.get(part.source)?.title ?? part.source}`;
      marker.setAttribute('aria-label', 'word replaced');
      fragment.append(marker);
    }
  }
  return fragment;
}

function render() {
  if (!sample) return;
  try {
    const parts = NurtureFilter.segments(sample.value, currentRules());
    const count = parts.filter(part => part.changed).length;
    output.replaceChildren(markedFragment(parts));
    const lists = [...lexicons.values()].filter(list => list.enabled).length;
    filterStatus.textContent = `${count} replacement${count === 1 ? '' : 's'}. Your rules plus ${lists} word list${lists === 1 ? '' : 's'}; this demo does not use a model.`;
  } catch (error) {
    output.replaceChildren();
    filterStatus.textContent = error.message;
  }
}

function addLexicon(source, text) {
  const entries = NurtureFilter.parseWordList(text);
  lexicons.set(source.id, { ...source, entries, enabled: source.enabled ?? Boolean(source.default_enabled) });
  renderLexicons();
  render();
}

function renderLexicons() {
  const fragment = document.createDocumentFragment();
  for (const list of lexicons.values()) {
    const label = document.createElement('label');
    label.className = 'check source';
    const box = document.createElement('input');
    box.type = 'checkbox';
    box.checked = list.enabled;
    box.addEventListener('change', () => {
      list.enabled = box.checked;
      render();
    });
    const body = document.createElement('span');
    body.append(`${list.title} · ${list.entries.length.toLocaleString()} entries`);
    const detail = document.createElement('small');
    if (list.url) {
      detail.append(`${list.author}. `);
      const link = document.createElement('a');
      link.href = list.url;
      link.textContent = 'Source';
      const copy = document.createElement('a');
      copy.href = `lexicons/${list.file}`;
      copy.textContent = 'vendored copy';
      detail.append(link, ' · ', copy, `. ${list.description} ${list.license_note}`);
    } else {
      detail.append('Local file for this page session only; not uploaded or saved.');
    }
    body.append(detail);
    label.append(box, body);
    fragment.append(label);
  }
  lexiconList.replaceChildren(fragment);
}

async function loadLexicons() {
  try {
    const manifest = await (await fetch('lexicons/sources.json')).json();
    for (const source of manifest.sources) {
      const response = await fetch(`lexicons/${source.file}`);
      if (!response.ok) throw new Error(`${source.id}: ${response.status}`);
      addLexicon(source, await response.text());
    }
  } catch (error) {
    lexiconList.textContent = `Word lists unavailable: ${error.message}`;
  }
}

lexiconFile.addEventListener('change', async () => {
  const file = lexiconFile.files[0];
  if (!file) return;
  try {
    if (file.size > 2_000_000) throw new Error('Use a word list under 2 MB.');
    addLexicon({ id: `local-${file.name}`, title: file.name, enabled: true }, await file.text());
  } catch (error) {
    lexiconFile.setCustomValidity(error.message);
    lexiconFile.reportValidity();
    lexiconFile.setCustomValidity('');
  }
  lexiconFile.value = '';
});

for (const input of [rulesInput, sample, listReplacement, exceptionsInput]) input?.addEventListener('input', render);
render();
loadLexicons();
