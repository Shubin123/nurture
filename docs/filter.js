(function (root) {
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

  function segments(text, rules) {
    const replacements = new Map(rules.map(rule => [rule.word, rule.replacement]));
    const output = [];
    const tokens = /[\p{L}\p{N}\p{M}_]+/gu;
    let position = 0;
    for (const match of text.matchAll(tokens)) {
      const replacement = replacements.get(match[0].toLowerCase());
      if (replacement === undefined) continue;
      if (match.index > position) output.push({ text: text.slice(position, match.index), changed: false });
      output.push({ text: replacement, original: match[0], changed: true });
      position = match.index + match[0].length;
    }
    if (position < text.length) output.push({ text: text.slice(position), changed: false });
    return output;
  }

  const api = { parseRules, segments };
  root.NurtureFilter = api;
  if (typeof module !== 'undefined') module.exports = api;
})(globalThis);
