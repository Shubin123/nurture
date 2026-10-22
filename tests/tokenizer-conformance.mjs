import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { WordPiece, linearTokens, linearFeatures } from '../ml/browser-runtime.js';

const tokenizer = new WordPiece(readFileSync('docs/models/vocab.txt', 'utf8'));
const fixtures = JSON.parse(readFileSync('ml/runs/tokenizer-fixtures.json', 'utf8'));
const mismatches = [];
for (const [index, fixture] of fixtures.entries()) {
  const codepoints = new Map();
  let utf16 = 0;
  for (const [position, letter] of [...fixture.text].entries()) {
    codepoints.set(utf16, position);
    utf16 += letter.length;
  }
  codepoints.set(utf16, [...fixture.text].length);
  const pieces = tokenizer.tokenize(fixture.text);
  try {
    assert.deepEqual(pieces.map(piece => piece.id), fixture.ids);
    assert.deepEqual(pieces.map(piece => [codepoints.get(piece.start), codepoints.get(piece.end)]), fixture.offsets);
    const tokens = linearTokens(fixture.text);
    assert.deepEqual(tokens.map(token => [token.text, codepoints.get(token.start), codepoints.get(token.end)]), fixture.linear_tokens);
    assert.deepEqual(tokens.map((_, token) => linearFeatures(tokens, token)), fixture.features);
  } catch (error) {
    mismatches.push({ index, error: error.message.slice(0, 1200) });
  }
}
const report = { fixtures: fixtures.length, mismatches: mismatches.length, details: mismatches };
writeFileSync('docs/reports/tokenizer-conformance.json', JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report, null, 2));
assert.equal(mismatches.length, 0, 'Browser/Python tokenization mismatch.');
