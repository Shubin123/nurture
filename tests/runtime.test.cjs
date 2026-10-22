const { test } = require('node:test');
const assert = require('node:assert/strict');

test('browser hash and character feature extraction are deterministic', async () => {
  const { hash, linearTokens, linearFeatures } = await import('../ml/browser-runtime.js');
  assert.equal(hash('hello'), 11435);
  const tokens = linearTokens("An idiot's café!");
  assert.deepEqual(tokens.map(token => token.text), ['An', "idiot's", 'caf', 'é', '!']);
  assert.deepEqual(linearFeatures(tokens, 1), linearFeatures(tokens, 1));
});

test('overlapping chunks cover the document without truncating it', async () => {
  const { chunksFor } = await import('../ml/browser-runtime.js');
  for (const length of [0, 1, 126, 127, 1000]) {
    const tokens = Array.from({ length }, (_, index) => index);
    const chunks = chunksFor(tokens);
    const covered = new Set(chunks.flatMap(chunk => chunk.tokens));
    assert.equal(covered.size, length);
    assert.ok(chunks.every(chunk => chunk.tokens.length <= 126));
  }
});

test('wordpiece merging never joins punctuation-separated words', async () => {
  const { mergeWordPieces } = await import('../ml/browser-runtime.js');
  const merged = mergeWordPieces('idiot!', [{ start: 0, end: 2, score: 0.7 }, { start: 2, end: 5, score: 0.8 }, { start: 5, end: 6, score: 0.1 }]);
  assert.equal(merged[0].text, 'idiot');
  assert.equal(merged[0].score, 0.8);
  assert.equal(merged[1].text, '!');
});
