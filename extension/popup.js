const status = document.querySelector('#status');
const buttons = [...document.querySelectorAll('button')];

async function withTab(callback) {
  buttons.forEach(button => { button.disabled = true; });
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab?.id || !/^https?:\/\//.test(tab.url || '')) throw new Error('Select an ordinary HTTP(S) page. Browser settings and protected pages are unsupported.');
    await callback(tab.id);
  } catch (error) {
    status.textContent = `Not applied: ${error.message}`;
  } finally {
    buttons.forEach(button => { button.disabled = false; });
  }
}

document.querySelector('#enable').addEventListener('click', () => {
  if (!document.querySelector('#consent').checked) {
    status.textContent = 'Confirm permission before enabling this tab.';
    return;
  }
  let rules;
  try {
    rules = NurtureFilter.parseRules(document.querySelector('#rules').value);
    if (!rules.length) throw new Error('Add at least one replacement rule.');
  } catch (error) {
    status.textContent = error.message;
    return;
  }
  withTab(async tabId => {
    await chrome.scripting.executeScript({ target: { tabId }, files: ['filter.js', 'content.js'] });
    await chrome.scripting.executeScript({ target: { tabId }, func: settings => globalThis.nurtureSession.start(settings), args: [rules] });
    status.textContent = 'Enabled on this tab. Text is processed locally; * marks replacements. Stop here or reload to reset.';
  });
});

document.querySelector('#stop').addEventListener('click', () => withTab(async tabId => {
  await chrome.scripting.executeScript({ target: { tabId }, func: () => globalThis.nurtureSession?.stop() });
  status.textContent = 'Stopped. Still-present replacement spans were restored.';
}));
