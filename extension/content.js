(() => {
  if (globalThis.nurtureSession) return;
  const excluded = 'script,style,noscript,textarea,input,select,option,code,pre,svg,math,[contenteditable]:not([contenteditable="false"]),[role="textbox"],[data-nurture-replacement]';
  const changes = new Map();
  const queue = new Set();
  let observer = null;
  let timer = null;
  let rules = [];
  let control = null;

  function replaceNode(node) {
    if (!node.isConnected || !node.parentElement || node.parentElement.closest(excluded)) return;
    const parts = NurtureFilter.segments(node.data, rules);
    if (!parts.some(part => part.changed)) return;
    const fragment = document.createDocumentFragment();
    for (const part of parts) {
      if (!part.changed) {
        fragment.append(document.createTextNode(part.text));
        continue;
      }
      const span = document.createElement('span');
      span.dataset.nurtureReplacement = '';
      span.append(document.createTextNode(part.text));
      const marker = document.createElement('sup');
      marker.textContent = '*';
      marker.title = 'Word replaced by Nurture';
      marker.setAttribute('aria-label', 'word replaced');
      marker.style.fontSize = '0.7em';
      span.append(marker);
      changes.set(span, { original: part.original, rendered: span.textContent });
      fragment.append(span);
    }
    node.replaceWith(fragment);
  }

  function scan(root) {
    if (!root.isConnected) return;
    if (root.nodeType === Node.TEXT_NODE) {
      replaceNode(root);
      return;
    }
    if (root.nodeType !== Node.ELEMENT_NODE || root.closest(excluded)) return;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach(replaceNode);
  }

  function flush() {
    timer = null;
    observer.disconnect();
    for (const span of changes.keys()) if (!span.isConnected) changes.delete(span);
    queue.forEach(scan);
    queue.clear();
    observer.observe(document.body, { subtree: true, childList: true, characterData: true });
  }

  function stop() {
    observer?.disconnect();
    observer = null;
    clearTimeout(timer);
    timer = null;
    queue.clear();
    for (const [span, saved] of changes) {
      if (span.isConnected && span.textContent === saved.rendered) span.replaceWith(document.createTextNode(saved.original));
    }
    changes.clear();
    control?.remove();
    control = null;
  }

  function start(settings) {
    stop();
    rules = settings;
    control = document.createElement('button');
    control.dataset.nurtureReplacement = '';
    control.textContent = 'Nurture active · Stop & restore';
    control.style.cssText = 'position:fixed;bottom:16px;right:16px;z-index:2147483647;padding:12px;background:#d5ed9d;color:#101916;border:2px solid #101916;border-radius:8px;font:14px system-ui;cursor:pointer';
    control.addEventListener('click', stop);
    document.body.append(control);
    observer = new MutationObserver(records => {
      for (const record of records) {
        if (record.type === 'characterData') queue.add(record.target);
        else record.addedNodes.forEach(node => queue.add(node));
      }
      if (!timer) timer = setTimeout(flush, 100);
    });
    queue.add(document.body);
    flush();
  }

  globalThis.nurtureSession = { start, stop };
})();
