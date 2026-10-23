// Share a tab, play it back a few seconds late, and mute every flagged word in its speech.
const stage = document.querySelector('#stage');
const shareButton = document.querySelector('#share');
const stopButton = document.querySelector('#stop');
const preview = document.querySelector('#preview');
const delayedPreview = document.querySelector('#delayed-preview');
const captions = document.querySelector('#captions');
const stageMessage = document.querySelector('#stage-message');
const status = document.querySelector('#status');
const loadProgress = document.querySelector('#load-progress');
const speechMode = document.querySelector('#speech-mode');
const speechDelay = document.querySelector('#speech-delay');
let activeStream = null;
let activeContext = null;
let speechSession = null;
let requestVersion = 0;

function setLive(live, message = '') {
  stage.classList.toggle('live', live);
  stageMessage.textContent = message;
  stageMessage.hidden = !message;
  shareButton.hidden = live;
  stopButton.disabled = !live;
  speechDelay.disabled = live;
}

function showProgress({ loaded, total, cached, preparing }) {
  stageMessage.hidden = false;
  loadProgress.hidden = Boolean(preparing);
  if (preparing) {
    stageMessage.textContent = 'Preparing the speech model…';
    return;
  }
  loadProgress.value = loaded / total;
  const mb = bytes => (bytes / 1e6).toFixed(0);
  stageMessage.textContent = cached ? 'Loading the speech model from this browser…' : `Downloading the speech model · ${mb(loaded)} / ${mb(total)} MB (saved for next time)`;
}

let shownSpeech = '';
function showSpeech(state) {
  const key = [state.text, state.words, state.filtered, state.late, state.error].join('|');
  if (key === shownSpeech) return;
  shownSpeech = key;
  stageMessage.hidden = Boolean(state.text);
  // Subtitle-style: only the latest words, starting at a word boundary.
  let text = state.text;
  if (text.length > 160) text = text.slice(text.indexOf(' ', text.length - 160) + 1);
  captions.replaceChildren(markedFragment(NurtureFilter.segments(text, currentRules())));
  const parts = [`${state.words} words heard · ${state.filtered} ${speechMode.value === 'beep' ? 'beeped' : 'muted'}`];
  if (state.late) parts.push(`${state.late} caught too late, try a longer delay`);
  if (state.error) parts.push(`error: ${state.error}`);
  status.textContent = parts.join(' · ');
}

function stopSharing(message = '') {
  requestVersion += 1;
  speechSession?.stop();
  speechSession = null;
  if (activeContext?.state !== 'closed') activeContext?.close();
  activeContext = null;
  activeStream?.getTracks().forEach(track => track.stop());
  activeStream = null;
  preview.srcObject = null;
  captions.replaceChildren();
  delayedPreview.getContext('2d').clearRect(0, 0, delayedPreview.width, delayedPreview.height);
  setLive(false);
  status.textContent = message;
}

shareButton.addEventListener('click', async () => {
  if (!navigator.mediaDevices?.getDisplayMedia) {
    status.textContent = 'Tab sharing needs a desktop Chromium browser (Chrome or Edge).';
    return;
  }
  const version = ++requestVersion;
  // Created inside the click so the browser allows audio playback.
  const context = new AudioContext();
  activeContext = context;
  // Start loading the speech model while the picker is open.
  NurtureSpeech.load(showProgress).then(() => {
    loadProgress.hidden = true;
    if (!stage.classList.contains('live')) stageMessage.hidden = true;
  }, error => {
    loadProgress.hidden = true;
    stageMessage.textContent = `Speech model unavailable: ${error.message}`;
  });
  status.textContent = 'Choose a tab and turn on “Also share tab audio”.';
  try {
    const stream = await navigator.mediaDevices.getDisplayMedia({
      video: true,
      audio: { suppressLocalAudioPlayback: true, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      preferCurrentTab: false,
      selfBrowserSurface: 'exclude',
      systemAudio: 'exclude'
    });
    if (version !== requestVersion) return stream.getTracks().forEach(track => track.stop());
    activeStream = stream;
    if (!stream.getAudioTracks().length) return stopSharing('No audio was shared. Share a tab (not a window or screen) with “Also share tab audio” on.');
    stream.getVideoTracks()[0].addEventListener('ended', () => stopSharing(), { once: true });
    preview.srcObject = stream;
    await preview.play();
    setLive(true, 'Loading the speech model…');
    status.textContent = '';
    const delaySeconds = Number(speechDelay.value);
    const session = await NurtureSpeech.start({ context, stream, delaySeconds, mode: speechMode.value, getRules: currentRules, canvas: delayedPreview, video: preview, onUpdate: showSpeech, onProgress: showProgress });
    if (version !== requestVersion) return session.stop();
    speechSession = session;
    stageMessage.textContent = `Listening… playback is ${delaySeconds} s behind the tab.`;
  } catch (error) {
    if (version !== requestVersion) return;
    stopSharing(error.name === 'NotAllowedError' ? 'Sharing cancelled.' : `Could not start: ${error.message}`);
  }
});

stopButton.addEventListener('click', () => stopSharing());
speechMode.addEventListener('change', () => speechSession?.setMode(speechMode.value));
window.addEventListener('pagehide', () => stopSharing());
