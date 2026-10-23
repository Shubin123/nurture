// Minimal Whisper (English-only) inference: log-mel features, greedy decoding with a KV cache,
// byte-level BPE decoding and word timestamps from cross-attention DTW, as in openai/whisper timing.py.
export const SAMPLE_RATE = 16000;
const N_FFT = 400;
const HOP = 160;
const N_MELS = 80;
const N_FRAMES = 3000;
const BINS = N_FFT / 2 + 1;

function hzToMel(hz) {
  return hz < 1000 ? hz / (200 / 3) : 15 + Math.log(hz / 1000) / (Math.log(6.4) / 27);
}

function melToHz(mel) {
  return mel < 15 ? mel * (200 / 3) : 1000 * Math.exp((Math.log(6.4) / 27) * (mel - 15));
}

// Slaney-scale, Slaney-normalized filters (librosa.filters.mel defaults used by Whisper).
export function melFilters() {
  const top = hzToMel(SAMPLE_RATE / 2);
  const points = Array.from({ length: N_MELS + 2 }, (_, index) => melToHz(top * index / (N_MELS + 1)));
  const filters = new Float32Array(N_MELS * BINS);
  for (let mel = 0; mel < N_MELS; mel += 1) {
    const [left, center, right] = [points[mel], points[mel + 1], points[mel + 2]];
    const norm = 2 / (right - left);
    for (let bin = 0; bin < BINS; bin += 1) {
      const hz = bin * SAMPLE_RATE / N_FFT;
      const weight = Math.max(0, Math.min((hz - left) / (center - left), (right - hz) / (right - center)));
      filters[mel * BINS + bin] = weight * norm;
    }
  }
  return filters;
}

let tables = null;
function dftTables() {
  if (tables) return tables;
  const cos = new Float32Array(BINS * N_FFT);
  const sin = new Float32Array(BINS * N_FFT);
  for (let bin = 0; bin < BINS; bin += 1) {
    for (let n = 0; n < N_FFT; n += 1) {
      const angle = 2 * Math.PI * bin * n / N_FFT;
      cos[bin * N_FFT + n] = Math.cos(angle);
      sin[bin * N_FFT + n] = Math.sin(angle);
    }
  }
  const window = Float32Array.from({ length: N_FFT }, (_, n) => 0.5 - 0.5 * Math.cos(2 * Math.PI * n / N_FFT));
  tables = { cos, sin, window, filters: melFilters() };
  return tables;
}

// [80, 3000] log-mel for up to 30 s of 16 kHz audio, zero-padded like WhisperFeatureExtractor.
export function logMel(audio) {
  const { cos, sin, window, filters } = dftTables();
  const length = Math.min(audio.length, SAMPLE_RATE * 30);
  const sample = index => {
    const reflected = index < 0 ? -index : index;
    return reflected < length ? audio[reflected] : 0;
  };
  const output = new Float32Array(N_MELS * N_FRAMES).fill(-10);
  // Frames entirely past the audio are silent and stay at log10(1e-10).
  const active = Math.min(N_FRAMES, Math.ceil((length + N_FFT / 2) / HOP));
  const frame = new Float32Array(N_FFT);
  const power = new Float32Array(BINS);
  let peak = -Infinity;
  for (let t = 0; t < active; t += 1) {
    for (let n = 0; n < N_FFT; n += 1) frame[n] = sample(t * HOP + n - N_FFT / 2) * window[n];
    for (let bin = 0; bin < BINS; bin += 1) {
      let re = 0;
      let im = 0;
      const row = bin * N_FFT;
      for (let n = 0; n < N_FFT; n += 1) {
        re += frame[n] * cos[row + n];
        im -= frame[n] * sin[row + n];
      }
      power[bin] = re * re + im * im;
    }
    for (let mel = 0; mel < N_MELS; mel += 1) {
      let sum = 0;
      const row = mel * BINS;
      for (let bin = 0; bin < BINS; bin += 1) sum += filters[row + bin] * power[bin];
      const value = Math.log10(Math.max(sum, 1e-10));
      output[mel * N_FRAMES + t] = value;
      if (value > peak) peak = value;
    }
  }
  peak = Math.max(peak, -10);
  for (let index = 0; index < output.length; index += 1) output[index] = (Math.max(output[index], peak - 8) + 4) / 4;
  return output;
}

function bytesToUnicode() {
  const bytes = [];
  for (let byte = 33; byte <= 126; byte += 1) bytes.push(byte);
  for (let byte = 161; byte <= 172; byte += 1) bytes.push(byte);
  for (let byte = 174; byte <= 255; byte += 1) bytes.push(byte);
  const chars = [...bytes];
  let extra = 0;
  for (let byte = 0; byte < 256; byte += 1) {
    if (!bytes.includes(byte)) {
      bytes.push(byte);
      chars.push(256 + extra);
      extra += 1;
    }
  }
  return new Map(chars.map((char, index) => [String.fromCodePoint(char), bytes[index]]));
}

export class ByteDecoder {
  constructor(vocabulary) {
    this.tokens = [];
    for (const [token, id] of Object.entries(vocabulary)) this.tokens[id] = token;
    this.bytes = bytesToUnicode();
    this.utf8 = new TextDecoder('utf-8');
  }

  decode(ids) {
    const bytes = [];
    for (const id of ids) for (const char of this.tokens[id] ?? '') bytes.push(this.bytes.get(char) ?? 63);
    return this.utf8.decode(new Uint8Array(bytes));
  }
}

const PUNCTUATION = new Set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~');

// openai/whisper tokenizer.split_tokens_on_spaces for English.
export function splitWords(decoder, ids, eot) {
  const groups = [];
  let current = [];
  for (const id of ids) {
    current.push(id);
    const text = decoder.decode(current);
    if (!text.includes('�') || id >= eot) {
      groups.push({ text, ids: current });
      current = [];
    }
  }
  const words = [];
  for (const group of groups) {
    const special = group.ids[0] >= eot;
    const startsWord = special || group.text.startsWith(' ') || PUNCTUATION.has(group.text.trim()) || !words.length;
    if (startsWord) words.push({ text: group.text, ids: [...group.ids] });
    else {
      words[words.length - 1].text += group.text;
      words[words.length - 1].ids.push(...group.ids);
    }
  }
  return words;
}

export function dtw(cost, rows, columns) {
  const total = new Float64Array((rows + 1) * (columns + 1)).fill(Infinity);
  const trace = new Int8Array((rows + 1) * (columns + 1)).fill(-1);
  const at = (i, j) => i * (columns + 1) + j;
  total[0] = 0;
  for (let j = 1; j <= columns; j += 1) {
    for (let i = 1; i <= rows; i += 1) {
      const c0 = total[at(i - 1, j - 1)];
      const c1 = total[at(i - 1, j)];
      const c2 = total[at(i, j - 1)];
      let best = c2;
      let step = 2;
      if (c0 < c1 && c0 < c2) [best, step] = [c0, 0];
      else if (c1 < c0 && c1 < c2) [best, step] = [c1, 1];
      total[at(i, j)] = cost[(i - 1) * columns + j - 1] + best;
      trace[at(i, j)] = step;
    }
  }
  for (let j = 0; j <= columns; j += 1) trace[at(0, j)] = 2;
  for (let i = 0; i <= rows; i += 1) trace[at(i, 0)] = 1;
  const text = [];
  const time = [];
  let i = rows;
  let j = columns;
  while (i > 0 || j > 0) {
    text.push(i - 1);
    time.push(j - 1);
    const step = trace[at(i, j)];
    if (step === 0) {
      i -= 1;
      j -= 1;
    } else if (step === 1) i -= 1;
    else j -= 1;
  }
  return { text: text.reverse(), time: time.reverse() };
}

function medianFilter(values, width) {
  const half = Math.floor(width / 2);
  const output = new Float32Array(values.length);
  const window = new Float32Array(width);
  for (let index = 0; index < values.length; index += 1) {
    for (let offset = -half; offset <= half; offset += 1) {
      let position = index + offset;
      if (position < 0) position = -position;
      if (position >= values.length) position = 2 * (values.length - 1) - position;
      window[offset + half] = values[Math.min(Math.max(position, 0), values.length - 1)];
    }
    output[index] = window.slice().sort()[half];
  }
  return output;
}

// heads: per alignment head, Float32Array [tokens, frames] of cross-attention weights.
// Returns one jump time (seconds) per row after the start-of-transcript row.
export function alignmentTimes(heads, tokens, frames) {
  const matrix = new Float32Array(tokens * frames);
  for (const weights of heads) {
    for (let f = 0; f < frames; f += 1) {
      let mean = 0;
      for (let t = 0; t < tokens; t += 1) mean += weights[t * frames + f];
      mean /= tokens;
      let variance = 0;
      for (let t = 0; t < tokens; t += 1) variance += (weights[t * frames + f] - mean) ** 2;
      const std = Math.sqrt(variance / tokens) || 1;
      for (let t = 0; t < tokens; t += 1) weights[t * frames + f] = (weights[t * frames + f] - mean) / std;
    }
    for (let t = 0; t < tokens; t += 1) {
      const filtered = medianFilter(weights.subarray(t * frames, (t + 1) * frames), 7);
      for (let f = 0; f < frames; f += 1) matrix[t * frames + f] += filtered[f] / heads.length;
    }
  }
  // Drop the start-of-transcript row and the final end-of-text row.
  const rows = tokens - 2;
  const cost = new Float32Array(rows * frames);
  for (let index = 0; index < cost.length; index += 1) cost[index] = -matrix[frames + index];
  const path = dtw(cost, rows, frames);
  const jumps = [];
  for (let index = 0; index < path.text.length; index += 1) {
    if (index === 0 || path.text[index] !== path.text[index - 1]) jumps.push(path.time[index] / 50);
  }
  return jumps;
}
