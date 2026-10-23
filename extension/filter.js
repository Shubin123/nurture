(function (root) {
  const tokenPattern = /[\p{L}\p{N}\p{M}_]+/gu;
  // Joiners allowed inside a multi-token list entry, e.g. "butt-head", "nigger's", "beat off".
  const joiner = /^[\s\-'’]{1,3}$/u;

  function tokenize(text) {
    return Array.from(text.matchAll(tokenPattern), match => match[0].toLowerCase());
  }

  function parseRules(input) {
    const lines = input.split(/\r?\n/).map(line => line.trim()).filter(Boolean);
    if (lines.length > 100) throw new Error('Use at most 100 rules.');
    const seen = new Set();
    return lines.map(line => {
      const separator = line.indexOf('=');
      if (separator < 1) throw new Error('Use one word = replacement per line.');
      const word = line.slice(0, separator).trim().toLowerCase();
      const replacement = line.slice(separator + 1).trim();
      if (!/^[a-z]{1,40}$/.test(word)) throw new Error('Match terms must be single English words, up to 40 letters.');
      if (!replacement || replacement.length > 80) throw new Error('Replacements must contain 1–80 characters.');
      if (seen.has(word)) throw new Error(`Duplicate word: ${word}`);
      seen.add(word);
      return { word, replacement };
    });
  }

  // Plain-text word list: one entry per line, blank lines and "#" comments ignored.
  function parseWordList(input, maxEntries = 50000) {
    const entries = [];
    const seen = new Set();
    for (const line of input.split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith('#')) continue;
      const tokens = tokenize(trimmed);
      const key = tokens.join(' ');
      if (!tokens.length || tokens.length > 8 || seen.has(key)) continue;
      seen.add(key);
      entries.push(tokens);
      if (entries.length > maxEntries) throw new Error(`Word lists may contain at most ${maxEntries} entries.`);
    }
    return entries;
  }

  // Combine explicit rules (highest priority) with word lists, minus exceptions.
  function combineRules(rules, lists = [], exceptions = '') {
    const blocked = new Set(parseWordList(exceptions).map(tokens => tokens.join(' ')));
    const seen = new Set();
    const combined = [];
    const add = (tokens, replacement, source) => {
      const key = tokens.join(' ');
      if (seen.has(key) || blocked.has(key)) return;
      seen.add(key);
      combined.push({ word: key, tokens, replacement, source });
    };
    for (const rule of rules) add(rule.tokens ?? [rule.word], rule.replacement, rule.source ?? 'rules');
    for (const list of lists) for (const tokens of list.entries) add(tokens, list.replacement, list.source);
    return combined;
  }

  function segments(text, rules) {
    const index = new Map();
    for (const rule of rules) {
      const tokens = rule.tokens ?? [rule.word];
      if (!index.has(tokens[0])) index.set(tokens[0], []);
      index.get(tokens[0]).push({ tokens, replacement: rule.replacement, source: rule.source });
    }
    for (const candidates of index.values()) candidates.sort((a, b) => b.tokens.length - a.tokens.length);
    const matches = Array.from(text.matchAll(tokenPattern));
    const output = [];
    let position = 0;
    for (let i = 0; i < matches.length; i += 1) {
      const candidates = index.get(matches[i][0].toLowerCase());
      if (!candidates) continue;
      const found = candidates.find(({ tokens }) => tokens.every((token, offset) => {
        const match = matches[i + offset];
        if (!match || match[0].toLowerCase() !== token) return false;
        if (offset === 0) return true;
        const previous = matches[i + offset - 1];
        return joiner.test(text.slice(previous.index + previous[0].length, match.index));
      }));
      if (!found) continue;
      const last = matches[i + found.tokens.length - 1];
      const end = last.index + last[0].length;
      if (matches[i].index > position) output.push({ text: text.slice(position, matches[i].index), changed: false });
      output.push({ text: found.replacement, original: text.slice(matches[i].index, end), changed: true, source: found.source });
      position = end;
      i += found.tokens.length - 1;
    }
    if (position < text.length) output.push({ text: text.slice(position), changed: false });
    return output;
  }

  const api = { parseRules, parseWordList, combineRules, segments, tokenize };
  root.NurtureFilter = api;
  if (typeof module !== 'undefined') module.exports = api;
})(globalThis);
