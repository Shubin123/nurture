const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { parseRules, segments } = require('../docs/filter.js');
const rules = parseRules('stupid = unhelpful\nidiot = person');
const render = text => segments(text, rules).map(part => part.text + (part.changed ? '*' : '')).join('');

test('replaces whole English words and marks every replacement', () => {
  assert.equal(render('STUPID, stupid! idiot.'), 'unhelpful*, unhelpful*! person*.');
});
test('does not replace substrings, unicode lookalikes or longer tokens', () => {
  assert.equal(render('stupidly éstupid stupidé stupid_ 1stupid stupid\u0301'), 'stupidly éstupid stupidé stupid_ 1stupid stupid\u0301');
});
test('preserves punctuation and does not recursively filter replacement text', () => {
  const parts = segments('stupid', parseRules('stupid = idiot\nidiot = person'));
  assert.equal(parts[0].text, 'idiot');
  assert.equal(parts[0].original, 'stupid');
});
test('rejects malformed, duplicate and unbounded rules', () => {
  for (const input of ['bad', 'bad = ', 'two words = no', 'bad = x\nBAD = y', `bad = ${'x'.repeat(81)}`]) assert.throws(() => parseRules(input));
  assert.throws(() => parseRules(Array(101).fill('bad = x').join('\n')));
});
test('empty rules and empty input preserve text', () => {
  assert.deepEqual(parseRules(' \n'), []);
  assert.deepEqual(segments('', rules), []);
  assert.equal(segments('unchanged', [])[0].text, 'unchanged');
});
test('replacement text is literal, including HTML and dollar characters', () => {
  assert.equal(segments('word', parseRules('word = <b>$&</b>'))[0].text, '<b>$&</b>');
});
test('extension uses identical engine and least-privilege manifest', () => {
  assert.equal(readFileSync('docs/filter.js', 'utf8'), readFileSync('extension/filter.js', 'utf8'));
  const manifest = JSON.parse(readFileSync('extension/manifest.json', 'utf8'));
  assert.deepEqual(manifest.permissions, ['activeTab', 'scripting']);
  assert.equal(manifest.host_permissions, undefined);
  assert.equal(manifest.content_scripts, undefined);
});
