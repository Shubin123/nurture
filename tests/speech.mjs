import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { chromium } from 'playwright';

// End-to-end: a committed speech clip (macOS `say`, 16 kHz mono) is "shared" as a tab,
// transcribed locally and flagged words are muted.
const baseUrl = process.env.BASE_URL || 'http://localhost:8000';
const wav = readFileSync('tests/fixtures/speech.wav');
const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: true, args: ['--autoplay-policy=no-user-gesture-required'] });
try {
  const page = await browser.newPage();
  const errors = [];
  const hosts = new Set();
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => message.type() === 'error' && console.log('console:', message.text()));
  page.context().on('request', request => hosts.add(new URL(request.url()).hostname));
  await page.route('**/__fixture/speech.wav', route => route.fulfill({ body: wav, contentType: 'audio/wav' }));
  await page.goto(baseUrl);
  await page.waitForFunction(() => document.querySelector('#lexicon-list input'));

  // Worklet unit check: delayed output, silenced inside a mute interval.
  const worklet = await page.evaluate(async () => {
    const rate = 48000;
    const context = new OfflineAudioContext(1, rate, rate);
    await context.audioWorklet.addModule('speech-worklet.js');
    const node = new AudioWorkletNode(context, 'nurture-speech-filter', { outputChannelCount: [1], processorOptions: { delayFrames: 4800, batchFrames: 4096, fadeFrames: 240, mode: 'mute', mutes: [[12000, 24000]] } });
    const tone = new OscillatorNode(context, { frequency: 440 });
    tone.connect(node).connect(context.destination);
    tone.start();
    const data = (await context.startRendering()).getChannelData(0);
    const peak = (from, to) => Math.max(...Array.from(data.subarray(from, to), Math.abs));
    return { beforeDelay: peak(0, 4700), audible: peak(6000, 16000), muted: peak(17500, 28500), after: peak(30000, 40000) };
  });
  assert.equal(worklet.beforeDelay, 0);
  assert.ok(worklet.audible > 0.5 && worklet.after > 0.5);
  assert.ok(worklet.muted < 1e-6, `muted peak ${worklet.muted}`);

  // Nothing loads before the user asks to share.
  assert.equal([...hosts].length, 1);
  await page.evaluate(() => { navigator.mediaDevices.getDisplayMedia = async () => { throw new DOMException('denied', 'NotAllowedError'); }; });
  await page.locator('#share').click();
  await page.waitForFunction(() => document.querySelector('#status').textContent.includes('cancelled'));
  await page.setViewportSize({ width: 375, height: 812 });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.evaluate(() => {
    navigator.mediaDevices.getDisplayMedia = async () => {
      const audio = new Audio('/__fixture/speech.wav');
      const context = new AudioContext();
      const destination = context.createMediaStreamDestination();
      context.createMediaElementSource(audio).connect(destination);
      const canvas = Object.assign(document.createElement('canvas'), { width: 320, height: 180 });
      const paint = () => { const g = canvas.getContext('2d'); g.fillStyle = `hsl(${performance.now() / 10 % 360} 60% 50%)`; g.fillRect(0, 0, 320, 180); requestAnimationFrame(paint); };
      paint();
      await context.resume();
      // Leading silence lets the pipeline settle, as a real tab would before speech starts.
      setTimeout(() => audio.play(), 500);
      return new MediaStream([...canvas.captureStream(30).getVideoTracks(), ...destination.stream.getAudioTracks()]);
    };
  });
  await page.locator('#share').click();
  await page.waitForFunction(() => document.querySelector('#status').textContent.includes('words heard'), null, { timeout: 120000 }).catch(async error => { console.log(await page.locator('#status').innerText(), await page.locator('#stage-message').innerText(), errors); throw error; });
  // Wait until the whole clip (about 9 s + 4 s delay) has played and the transcript stops changing.
  await page.waitForFunction(() => /happened/i.test(document.querySelector('#captions').textContent), null, { timeout: 60000 }).catch(async error => { console.log('captions:', await page.locator('#captions').innerText(), '\nstatus:', await page.locator('#status').innerText()); throw error; });
  await page.waitForTimeout(4000);
  const captions = await page.locator('#captions').innerText();
  const status = await page.locator('#status').innerText();
  console.log(`captions: ${captions}\nstatus: ${status}`);
  assert.match(captions, /unhelpful\*/);
  assert.match(captions, /person\*/);
  assert.doesNotMatch(captions, /stupid|idiot/i);
  assert.match(status, /\b3 muted/);
  assert.match(captions, /what the bleep\*/i);
  assert.match(captions, /that was a really unhelpful\*/i);
  assert.doesNotMatch(status, /too late/);
  assert.equal(await page.locator('#delayed-preview').isHidden(), false);
  await page.locator('#stop').click();
  assert.equal(await page.locator('#captions').innerText(), '');
  assert.equal(await page.locator('#share').isVisible(), true);
  assert.deepEqual(errors, []);
  assert.deepEqual([...hosts], [new URL(baseUrl).hostname]);
  console.log('PASS: delayed/muted worklet output, local Whisper transcription, filtered captions, no late mutes, no third-party requests.');
} finally {
  await browser.close();
}
