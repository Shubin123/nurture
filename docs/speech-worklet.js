// Delays shared audio so flagged words can be silenced before they are heard.
// Input frames are numbered from 0; a mute [start, end] refers to input frame numbers.
class NurtureSpeechFilter extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const { delayFrames, batchFrames, fadeFrames, mode, mutes = [] } = options.processorOptions;
    this.delay = delayFrames;
    this.size = delayFrames + 256;
    this.rings = [new Float32Array(this.size), new Float32Array(this.size)];
    this.frame = 0;
    this.fade = fadeFrames;
    this.mode = mode;
    this.mutes = mutes.map(([start, end]) => [start, end]);
    this.batch = new Float32Array(batchFrames);
    this.batchStart = 0;
    this.batchFill = 0;
    this.phase = 0;
    this.stopped = false;
    this.port.onmessage = event => {
      if (event.data.mute) {
        const [start, end] = event.data.mute;
        // Report a mute that arrived after its audio was already played, unless an earlier pass covered it.
        const covered = this.mutes.some(([from, to]) => from <= end && to >= start);
        if (start < this.frame - this.delay && !covered) this.port.postMessage({ late: true, frame: start });
        this.mutes.push([start, end]);
        this.mutes.sort((a, b) => a[0] - b[0]);
      }
      if (event.data.mode) this.mode = event.data.mode;
      if (event.data.stop) this.stopped = true;
    };
  }

  gain(frame) {
    let gain = 1;
    for (const [start, end] of this.mutes) {
      if (start - this.fade > frame) break;
      if (end + this.fade < frame) continue;
      const distance = frame < start ? start - frame : frame > end ? frame - end : 0;
      gain = Math.min(gain, distance / this.fade);
    }
    return gain;
  }

  process(inputs, outputs) {
    if (this.stopped) return false;
    const input = inputs[0];
    const output = outputs[0];
    const length = output[0].length;
    const playFrame = this.frame - this.delay;
    while (this.mutes.length && this.mutes[0][1] + this.fade < playFrame) this.mutes.shift();
    for (let i = 0; i < length; i += 1) {
      const position = (this.frame + i) % this.size;
      let mono = 0;
      for (let channel = 0; channel < 2; channel += 1) {
        const sample = input[channel]?.[i] ?? input[0]?.[i] ?? 0;
        this.rings[channel][position] = sample;
        mono += sample / 2;
      }
      this.batch[this.batchFill] = mono;
      this.batchFill += 1;
      if (this.batchFill === this.batch.length) {
        this.port.postMessage({ frame: this.batchStart, samples: this.batch.slice() });
        this.batchStart += this.batchFill;
        this.batchFill = 0;
      }
      const frame = playFrame + i;
      const gain = frame < 0 ? 0 : this.gain(frame);
      const read = (frame % this.size + this.size) % this.size;
      const tone = this.mode === 'beep' ? 0.12 * Math.sin(this.phase) * (1 - gain) : 0;
      this.phase = (this.phase + 2 * Math.PI * 1000 / sampleRate) % (2 * Math.PI);
      for (let channel = 0; channel < output.length; channel += 1) {
        output[channel][i] = frame < 0 ? 0 : this.rings[Math.min(channel, 1)][read] * gain + tone;
      }
    }
    this.frame += length;
    return true;
  }
}

registerProcessor('nurture-speech-filter', NurtureSpeechFilter);
