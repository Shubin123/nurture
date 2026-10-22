const rulesInput = document.querySelector('#rules');
const sample = document.querySelector('#sample');
const output = document.querySelector('#output');
const filterStatus = document.querySelector('#filter-status');
const shareButton = document.querySelector('#share');
const stopButton = document.querySelector('#stop');
const preview = document.querySelector('#preview');
const shareStatus = document.querySelector('#share-status');
let activeStream = null;
let requesting = false;
let requestVersion = 0;

function render() {
  try {
    const rules = NurtureFilter.parseRules(rulesInput.value);
    const parts = NurtureFilter.segments(sample.value, rules);
    const fragment = document.createDocumentFragment();
    let count = 0;
    for (const part of parts) {
      fragment.append(document.createTextNode(part.text));
      if (part.changed) {
        const marker = document.createElement('sup');
        marker.textContent = '*';
        marker.title = 'Word replaced by your local rules';
        marker.setAttribute('aria-label', 'word replaced');
        fragment.append(marker);
        count += 1;
      }
    }
    output.replaceChildren(fragment);
    filterStatus.textContent = `${count} replacement${count === 1 ? '' : 's'}. English whole-word rules; this demo does not use a model.`;
  } catch (error) {
    output.replaceChildren();
    filterStatus.textContent = error.message;
  }
}

function stopSharing(message = 'Sharing stopped. No stream is retained.') {
  requestVersion += 1;
  const stream = activeStream;
  activeStream = null;
  if (stream) stream.getTracks().forEach(track => track.stop());
  preview.srcObject = null;
  preview.hidden = true;
  stopButton.disabled = true;
  shareButton.disabled = requesting;
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
  requesting = true;
  const version = ++requestVersion;
  shareButton.disabled = true;
  shareStatus.textContent = 'Choose the source in the browser picker. Cancel to share nothing.';
  try {
    const selectedStream = await navigator.mediaDevices.getDisplayMedia({
      video: true,
      audio: false,
      preferCurrentTab: false,
      selfBrowserSurface: 'exclude'
    });
    if (version !== requestVersion || !document.querySelector('#consent').checked) {
      selectedStream.getTracks().forEach(track => track.stop());
      return;
    }
    activeStream = selectedStream;
    preview.srcObject = activeStream;
    preview.hidden = false;
    stopButton.disabled = false;
    activeStream.getVideoTracks()[0].addEventListener('ended', () => stopSharing(), { once: true });
    shareStatus.textContent = 'Local, UNFILTERED preview only. No recording, upload, audio capture or model inference. This cannot change the source tab.';
    await preview.play();
  } catch (error) {
    stopSharing(error.name === 'NotAllowedError' ? 'Sharing cancelled or denied. Nothing is captured.' : `Sharing unavailable: ${error.message}`);
  } finally {
    requesting = false;
    shareButton.disabled = Boolean(activeStream);
  }
});

stopButton.addEventListener('click', () => stopSharing());
document.querySelector('#consent').addEventListener('change', event => {
  if (!event.target.checked) stopSharing('Consent withdrawn. Sharing stopped.');
});
window.addEventListener('pagehide', () => stopSharing());
rulesInput.addEventListener('input', render);
sample.addEventListener('input', render);
render();
