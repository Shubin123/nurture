const rulesInput = document.querySelector('#rules');
const sample = document.querySelector('#sample');
const output = document.querySelector('#output');
const filterStatus = document.querySelector('#filter-status');
const lexiconList = document.querySelector('#lexicon-list');
const lexiconFile = document.querySelector('#lexicon-file');
const listReplacement = document.querySelector('#list-replacement');
const exceptionsInput = document.querySelector('#exceptions');
const shareButton = document.querySelector('#share');
const stopButton = document.querySelector('#stop');
const preview = document.querySelector('#preview');
const delayedPreview = document.querySelector('#delayed-preview');
const captions = document.querySelector('#captions');
const shareStatus = document.querySelector('#share-status');
const speechConsent = document.querySelector('#speech-consent');
const speechMode = document.querySelector('#speech-mode');
const speechDelay = document.querySelector('#speech-delay');
const speechStatus = document.querySelector('#speech-status');
const lexicons = new Map();
let activeStream = null;
let activeContext = null;
let speechSession = null;
let speechReady = false;
let requesting = false;
let requestVersion = 0;

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
    filterStatus.textContent = error.message;
  }
  lexiconFile.value = '';
});

async function prepareSpeech() {
  if (!speechConsent.checked || speechReady) return;
  speechStatus.textContent = 'Loading the local speech model… choose a tab once it is ready.';
  try {
    const result = await NurtureSpeech.load();
    if (!speechConsent.checked) return;
    speechReady = true;
    speechStatus.textContent = `Speech model ready${result.load_ms ? ` in ${(result.load_ms / 1000).toFixed(1)} s` : ''}. Runs in a local worker on CPU/WASM.`;
  } catch (error) {
    speechStatus.textContent = `Speech model unavailable: ${error.message}`;
  }
}

let shownSpeech = '';
function showSpeech(state) {
  const key = [state.text, state.words, state.filtered, state.late, state.error, Math.round(state.inferenceMs)].join('|');
  if (key === shownSpeech) return;
  shownSpeech = key;
  try {
    captions.replaceChildren(markedFragment(NurtureFilter.segments(state.text, currentRules())));
    captions.scrollTop = captions.scrollHeight;
  } catch {
    captions.textContent = state.text;
  }
  const parts = [`Filtering speech · ${state.words} words heard · ${state.filtered} filtered`];
  if (state.inferenceMs) parts.push(`last transcription ${Math.round(state.inferenceMs)} ms`);
  if (state.late) parts.push(`${state.late} word${state.late === 1 ? '' : 's'} caught too late—increase the delay`);
  if (state.error) parts.push(`error: ${state.error}`);
  speechStatus.textContent = parts.join(' · ');
}

function stopSharing(message = 'Sharing stopped. No stream is retained.') {
  requestVersion += 1;
  speechSession?.stop();
  speechSession = null;
  activeContext?.close();
  activeContext = null;
  const stream = activeStream;
  activeStream = null;
  if (stream) stream.getTracks().forEach(track => track.stop());
  preview.srcObject = null;
  preview.hidden = true;
  delayedPreview.hidden = true;
  captions.hidden = true;
  captions.replaceChildren();
  stopButton.disabled = true;
  shareButton.disabled = requesting;
  speechDelay.disabled = false;
  shareStatus.textContent = message;
}

shareButton.addEventListener('click', async () => {
  if (!document.querySelector('#consent').checked) {
    shareStatus.textContent = 'First confirm that you are authorized to share this content.';
    return;
  }
  if (!navigator.mediaDevices?.getDisplayMedia) {
    shareStatus.textContent = 'Screen sharing is unavailable here. Use a supporting desktop browser over HTTPS or localhost.';
    return;
  }
  const filterSpeech = speechConsent.checked;
  if (filterSpeech && !speechReady) {
    shareStatus.textContent = 'Wait for the speech model to finish loading, or turn speech filtering off.';
    prepareSpeech();
    return;
  }
  requesting = true;
  const version = ++requestVersion;
  shareButton.disabled = true;
  // Created inside the click so the browser allows audio playback.
  const context = filterSpeech ? new AudioContext() : null;
  shareStatus.textContent = 'Choose the source in the browser picker. Cancel to share nothing.';
  try {
    const selectedStream = await navigator.mediaDevices.getDisplayMedia({
      video: true,
      audio: filterSpeech ? { suppressLocalAudioPlayback: true, echoCancellation: false, noiseSuppression: false, autoGainControl: false } : false,
      preferCurrentTab: false,
      selfBrowserSurface: 'exclude',
      systemAudio: 'exclude'
    });
    if (version !== requestVersion || !document.querySelector('#consent').checked) {
      selectedStream.getTracks().forEach(track => track.stop());
      context?.close();
      return;
    }
    activeStream = selectedStream;
    preview.srcObject = activeStream;
    stopButton.disabled = false;
    activeStream.getVideoTracks()[0].addEventListener('ended', () => stopSharing(), { once: true });
    await preview.play();
    if (!filterSpeech) {
      context?.close();
      preview.hidden = false;
      shareStatus.textContent = 'Local, UNFILTERED preview only. No recording, upload, audio capture or model inference. This cannot change the source tab.';
      return;
    }
    activeContext = context;
    if (!activeStream.getAudioTracks().length) {
      preview.hidden = false;
      shareStatus.textContent = 'No audio was shared, so there is no speech to filter. Stop, choose a tab and keep “Also share tab audio” on. Showing an unfiltered video preview.';
      return;
    }
    const delaySeconds = Number(speechDelay.value);
    speechDelay.disabled = true;
    delayedPreview.hidden = false;
    captions.hidden = false;
    speechSession = await NurtureSpeech.start({ context, stream: activeStream, delaySeconds, mode: speechMode.value, getRules: currentRules, canvas: delayedPreview, video: preview, onUpdate: showSpeech });
    if (version !== requestVersion) return;
    shareStatus.textContent = `Filtering speech locally with a ${delaySeconds} s delay. The original tab is silenced while shared; pixels are not filtered. No recording or upload.`;
  } catch (error) {
    context?.close();
    if (activeContext === context) activeContext = null;
    stopSharing(error.name === 'NotAllowedError' ? 'Sharing cancelled or denied. Nothing is captured.' : `Sharing unavailable: ${error.message}`);
  } finally {
    requesting = false;
    shareButton.disabled = Boolean(activeStream);
  }
});

stopButton.addEventListener('click', () => stopSharing());
speechMode.addEventListener('change', () => speechSession?.setMode(speechMode.value));
document.querySelector('#consent').addEventListener('change', event => {
  if (!event.target.checked) stopSharing('Consent withdrawn. Sharing stopped.');
});
speechConsent.addEventListener('change', () => {
  if (speechConsent.checked) return prepareSpeech();
  if (speechSession) stopSharing('Speech filtering turned off. Sharing stopped.');
  NurtureSpeech.unload();
  speechReady = false;
  speechStatus.textContent = 'Speech filtering off. The speech model is unloaded.';
});
window.addEventListener('pagehide', () => stopSharing());
for (const input of [rulesInput, sample, listReplacement, exceptionsInput]) input.addEventListener('input', render);
render();
loadLexicons();
prepareSpeech();
