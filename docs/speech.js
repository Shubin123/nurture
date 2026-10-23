// Local speech filtering for a shared tab: delay its audio, transcribe with Whisper in a worker,
// and mute (or beep) every word your rules match before the delayed audio reaches the speakers.
const NurtureSpeech = (() => {
  const RATE = 16000;
  const STEP_S = 1.2;
  const MAX_WINDOW_S = 8;
  const COMMIT_GUARD_S = 1;
  const OVERLAP_S = 0.6;
  const PAD_S = 0.08;
  const SILENCE_RMS = 0.004;
  let worker = null;
  let ready = null;
  let requestId = 0;
  const pending = new Map();

  function call(action, values = {}, transfer = []) {
    if (!worker) {
      worker = new Worker('asr-worker.js', { type: 'module' });
      worker.onmessage = event => {
        const entry = pending.get(event.data.id);
        if (!entry) return;
        pending.delete(event.data.id);
        if (event.data.error) entry.reject(new Error(event.data.error));
        else entry.resolve(event.data.result);
      };
      worker.onerror = event => unload(`Speech worker failed: ${event.message || 'unknown error'}`);
    }
    return new Promise((resolve, reject) => {
      const id = ++requestId;
      pending.set(id, { resolve, reject });
      worker.postMessage({ id, action, ...values }, transfer);
    });
  }

  function load() {
    ready ??= call('load').catch(error => {
      unload();
      throw error;
    });
    return ready;
  }

  function unload() {
    worker?.terminate();
    worker = null;
    ready = null;
    for (const entry of pending.values()) entry.reject(new Error('Speech model stopped.'));
    pending.clear();
  }

  // Character ranges of each word inside the joined window text, so phrase rules can span words.
  function flaggedRanges(words, rules) {
    let text = '';
    const ranges = words.map(word => {
      const start = text.length;
      text += word.text;
      return [start, text.length];
    });
    const flagged = [];
    let position = 0;
    for (const part of NurtureFilter.segments(text, rules)) {
      const length = part.changed ? part.original.length : part.text.length;
      if (part.changed) {
        const hit = words.filter((word, index) => ranges[index][0] < position + length && ranges[index][1] > position);
        if (hit.length) flagged.push({ start: hit[0].start, end: hit[hit.length - 1].end, original: part.original });
      }
      position += length;
    }
    return flagged;
  }

  async function start({ context, stream, delaySeconds, mode, getRules, canvas, video, onUpdate }) {
    await load();
    await context.audioWorklet.addModule('speech-worklet.js');
    const rate = context.sampleRate;
    const delayFrames = Math.round(delaySeconds * rate);
    const node = new AudioWorkletNode(context, 'nurture-speech-filter', {
      numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [2],
      processorOptions: { delayFrames, batchFrames: 4096, fadeFrames: Math.round(0.005 * rate), mode }
    });
    const source = context.createMediaStreamSource(stream);
    source.connect(node).connect(context.destination);
    await context.resume();

    const state = { stopped: false, busy: false, latestFrame: 0, words: 0, filtered: 0, late: 0, inferenceMs: 0, error: '' };
    let audio = new Float32Array(RATE * 30);
    let base = 0;
    let total = 0;
    let windowStart = 0;
    let committedUntil = 0;
    let pendingWords = false;
    let accumulator = 0;
    let accumulated = 0;
    let currentIndex = 0;
    const transcript = [];
    let revealed = 0;
    let shownText = '';

    function append(value) {
      if (total - base >= audio.length) {
        const from = Math.max(base, windowStart - Math.round(OVERLAP_S * RATE));
        const keep = audio.subarray(from - base, total - base);
        const next = new Float32Array(Math.max(audio.length, keep.length * 2));
        next.set(keep);
        audio = next;
        base = from;
      }
      audio[total - base] = value;
      total += 1;
    }

    // Box-filter decimation from the context rate to 16 kHz.
    function resample(frame, samples) {
      for (let i = 0; i < samples.length; i += 1) {
        const index = Math.floor((frame + i) * RATE / rate);
        if (index !== currentIndex && accumulated) {
          append(accumulator / accumulated);
          accumulator = 0;
          accumulated = 0;
        }
        currentIndex = index;
        accumulator += samples[i];
        accumulated += 1;
      }
    }

    function mute(startSeconds, endSeconds) {
      node.port.postMessage({ mute: [Math.max(0, Math.floor((startSeconds - PAD_S) * rate)), Math.ceil((endSeconds + PAD_S) * rate)] });
    }

    async function pump() {
      if (state.stopped || state.busy || total - windowStart < STEP_S * RATE) return;
      const end = total;
      windowStart = Math.max(windowStart, end - MAX_WINDOW_S * RATE);
      // Re-include a little committed audio: Whisper often drops a word cut at the clip start.
      const begin = Math.max(base, windowStart - Math.round(OVERLAP_S * RATE));
      const slice = audio.slice(begin - base, end - base);
      const tail = audio.subarray(Math.max(windowStart, end - Math.round(0.8 * RATE)) - base, end - base);
      let energy = 0;
      for (const sample of tail) energy += sample * sample;
      const silent = Math.sqrt(energy / tail.length) < SILENCE_RMS;
      // Silence after speech: transcribe once more and commit everything still pending.
      if (silent && !pendingWords) {
        windowStart = Math.max(windowStart, end - RATE / 2);
        return;
      }
      state.busy = true;
      try {
        const result = await call('transcribe', { audio: slice }, [slice.buffer]);
        if (state.stopped) return;
        state.inferenceMs = result.inference_ms;
        const offset = begin / RATE;
        const words = result.words
          .map(word => ({ text: word.text, start: offset + word.start, end: offset + Math.max(word.start, word.end) }))
          .filter(word => (word.start + word.end) / 2 > committedUntil);
        // Overlapping passes can repeat the last committed word with slightly shifted timestamps.
        const last = transcript[transcript.length - 1];
        if (last && words[0] && words[0].start < committedUntil + 0.3 && NurtureFilter.tokenize(words[0].text).join(' ') === NurtureFilter.tokenize(last.text).join(' ')) words.shift();
        for (const hit of flaggedRanges(words, getRules())) mute(hit.start, hit.end);
        const commitBefore = silent ? Infinity : end / RATE - COMMIT_GUARD_S;
        const committed = words.filter(word => word.end <= commitBefore);
        pendingWords = committed.length < words.length;
        for (const word of committed) transcript.push({ text: word.text, frame: Math.floor(word.start * rate) });
        state.words += committed.filter(word => /[\p{L}\p{N}]/u.test(word.text)).length;
        state.filtered += flaggedRanges(committed, getRules()).length;
        if (committed.length) committedUntil = committed[committed.length - 1].end;
        if (silent) windowStart = end;
        else if (committed.length) windowStart = Math.max(windowStart, Math.round(committedUntil * RATE));
        else if (!words.length) windowStart = Math.max(windowStart, end - RATE);
        state.error = '';
      } catch (error) {
        state.error = error.message;
      } finally {
        state.busy = false;
      }
      pump();
    }

    node.port.onmessage = event => {
      if (event.data.late) {
        state.late += 1;
        return;
      }
      state.latestFrame = event.data.frame + event.data.samples.length;
      resample(event.data.frame, event.data.samples);
      pump();
    };

    // Delayed video so pictures stay in sync with the filtered audio.
    const frames = [];
    const videoDelayMs = () => delaySeconds * 1000 + ((context.outputLatency || 0) + (context.baseLatency || 0)) * 1000;
    let lastCapture = 0;
    function capture() {
      if (state.stopped) return;
      const now = performance.now();
      if (now - lastCapture >= 1000 / 15 && video.videoWidth) {
        lastCapture = now;
        const width = Math.min(960, video.videoWidth);
        const height = Math.round(width * video.videoHeight / video.videoWidth);
        createImageBitmap(video, { resizeWidth: width, resizeHeight: height, resizeQuality: 'low' })
          .then(bitmap => state.stopped ? bitmap.close() : frames.push({ time: now, bitmap }))
          .catch(() => {});
      }
      video.requestVideoFrameCallback(capture);
    }
    video.requestVideoFrameCallback(capture);

    let animation = 0;
    function draw() {
      if (state.stopped) return;
      const due = performance.now() - videoDelayMs();
      let frame = null;
      while (frames.length && frames[0].time <= due) {
        frame?.bitmap.close();
        frame = frames.shift();
      }
      if (frame) {
        if (canvas.width !== frame.bitmap.width) canvas.width = frame.bitmap.width;
        if (canvas.height !== frame.bitmap.height) canvas.height = frame.bitmap.height;
        canvas.getContext('2d').drawImage(frame.bitmap, 0, 0);
        frame.bitmap.close();
      }
      const played = state.latestFrame - delayFrames;
      while (revealed < transcript.length && transcript[revealed].frame <= played) revealed += 1;
      const text = transcript.slice(0, revealed).map(word => word.text).join('').trimStart().slice(-1500);
      if (text !== shownText) shownText = text;
      onUpdate({ ...state, text: shownText });
      animation = requestAnimationFrame(draw);
    }
    animation = requestAnimationFrame(draw);

    return {
      setMode(next) { node.port.postMessage({ mode: next }); },
      stop() {
        state.stopped = true;
        cancelAnimationFrame(animation);
        for (const frame of frames.splice(0)) frame.bitmap.close();
        node.port.postMessage({ stop: true });
        source.disconnect();
        node.disconnect();
      }
    };
  }

  return { load, unload, start, flaggedRanges };
})();
