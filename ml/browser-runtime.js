const encoder = new TextEncoder();

export function hash(value) {
  let result = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    if (code > 127) {
      result = 2166136261;
      for (const byte of encoder.encode(value)) result = Math.imul(result ^ byte, 16777619) >>> 0;
      return result % 65536;
    }
    result = Math.imul(result ^ code, 16777619) >>> 0;
  }
  return result % 65536;
}

export function linearTokens(text) {
  return [...text.matchAll(/[A-Za-z]+(?:'[A-Za-z]+)?|[^\s]/gu)].map(match => ({ text: match[0], start: match.index, end: match.index + match[0].length }));
}

export function linearFeatures(tokens, index) {
  const word = tokens[index].text.toLowerCase();
  const previous = index ? tokens[index - 1].text.toLowerCase() : '<BOS>';
  const following = index + 1 < tokens.length ? tokens[index + 1].text.toLowerCase() : '<EOS>';
  const values = [`w=${word}`, `p=${previous}`, `n=${following}`, `pw=${previous}|${word}`, `wn=${word}|${following}`];
  const wrapped = [...`^${word}$`];
  for (const width of [2, 3, 4]) {
    for (let start = 0; start <= wrapped.length - width; start += 1) values.push(`c=${wrapped.slice(start, start + width).join('')}`);
  }
  return [...new Set(values.map(hash))].sort((left, right) => left - right);
}

export class LinearModel {
  constructor(config, weights) {
    this.config = config;
    this.weights = weights;
  }

  predict(text) {
    const tokens = linearTokens(text);
    return tokens.map((token, index) => {
      let logit = this.config.bias;
      for (const feature of linearFeatures(tokens, index)) logit += this.weights[feature] * this.config.scale;
      return { ...token, score: 1 / (1 + Math.exp(-Math.max(-50, Math.min(50, logit)))) };
    });
  }
}

function isChinese(code) {
  return (code >= 0x4e00 && code <= 0x9fff) || (code >= 0x3400 && code <= 0x4dbf) || (code >= 0x20000 && code <= 0x2a6df) || (code >= 0x2a700 && code <= 0x2b73f) || (code >= 0x2b740 && code <= 0x2b81f) || (code >= 0x2b820 && code <= 0x2ceaf) || (code >= 0xf900 && code <= 0xfaff) || (code >= 0x2f800 && code <= 0x2fa1f);
}

export class WordPiece {
  constructor(vocabulary) {
    this.vocabulary = new Map(vocabulary.trimEnd().split('\n').map((word, index) => [word, index]));
  }

  tokenize(text, allowSpecial = true) {
    const special = allowSpecial ? [...text.matchAll(/\[(?:CLS|SEP|MASK|UNK|PAD)\]/gu)] : [];
    if (special.length) {
      const tokens = [];
      let position = 0;
      for (const match of special) {
        tokens.push(...this.tokenize(text.slice(position, match.index), false).map(token => ({ ...token, start: token.start + position, end: token.end + position })));
        tokens.push({ id: this.vocabulary.get(match[0]), start: match.index, end: match.index + match[0].length });
        position = match.index + match[0].length;
      }
      tokens.push(...this.tokenize(text.slice(position), false).map(token => ({ ...token, start: token.start + position, end: token.end + position })));
      return tokens;
    }
    const basic = [];
    let current = [];
    function flush() {
      if (current.length) basic.push(current);
      current = [];
    }
    let position = 0;
    for (const original of text) {
      const start = position;
      position += original.length;
      const code = original.codePointAt(0);
      if (/\s/u.test(original)) { flush(); continue; }
      if (code === 0 || code === 0xfffd || /[\p{Cc}\p{Cf}]/u.test(original)) continue;
      const punctuation = /\p{P}/u.test(original) || (code >= 33 && code <= 47) || (code >= 58 && code <= 64) || (code >= 91 && code <= 96) || (code >= 123 && code <= 126);
      if (punctuation || isChinese(code)) flush();
      const normalized = original.toLowerCase().normalize('NFD').replace(/\p{Mn}/gu, '');
      for (const letter of normalized) current.push({ letter, start, end: position });
      if (punctuation || isChinese(code)) flush();
    }
    flush();
    const tokens = [];
    for (const characters of basic) {
      if (characters.length > 100) {
        tokens.push({ id: 100, start: characters[0].start, end: characters.at(-1).end });
        continue;
      }
      const pieces = [];
      let start = 0;
      while (start < characters.length) {
        let end = characters.length;
        let found = null;
        while (end > start) {
          const word = (start ? '##' : '') + characters.slice(start, end).map(item => item.letter).join('');
          if (this.vocabulary.has(word)) { found = this.vocabulary.get(word); break; }
          end -= 1;
        }
        if (found === null) break;
        pieces.push({ id: found, start: characters[start].start, end: characters[end - 1].end });
        start = end;
      }
      if (start < characters.length) tokens.push({ id: 100, start: characters[0].start, end: characters.at(-1).end });
      else tokens.push(...pieces);
    }
    return tokens;
  }
}

export function chunksFor(tokens, maxLength = 128, stride = 32) {
  const chunks = [];
  const capacity = maxLength - 2;
  for (let start = 0; start < tokens.length; start += capacity - stride) {
    const chunk = tokens.slice(start, start + capacity);
    chunks.push({ start, tokens: chunk });
    if (start + capacity >= tokens.length) break;
  }
  return chunks;
}

export function mergeWordPieces(text, pieces) {
  const words = [];
  for (const piece of pieces) {
    if (!Number.isFinite(piece.score)) continue;
    const previous = words.at(-1);
    if (previous && previous.end === piece.start && /^[\p{L}\p{N}\p{M}_]+$/u.test(text.slice(previous.start, piece.end))) {
      previous.end = piece.end;
      previous.score = Math.max(previous.score, piece.score);
    } else words.push({ start: piece.start, end: piece.end, score: piece.score });
  }
  return words.map(word => ({ ...word, text: text.slice(word.start, word.end) }));
}
