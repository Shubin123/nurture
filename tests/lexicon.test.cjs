const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { createHash } = require('node:crypto');
const { parseRules, parseWordList, combineRules, segments } = require('../docs/filter.js');
const show = (text, rules) => segments(text, rules).map(part => part.text + (part.changed ? '*' : '')).join('');

test('word lists ignore comments, blanks and duplicates and keep phrases as tokens', () => {
  assert.deepEqual(parseWordList('# comment\n\nDarn\ndarn\nbeat-off\nnigger\'s\nbeat off'), [['darn'], ['beat', 'off'], ['nigger', 's']]);
});
test('list entries match hyphenated, possessive and spaced phrases but not partial phrases', () => {
  const rules = combineRules([], [{ entries: parseWordList('beat-off\ndarn'), replacement: 'bleep', source: 'list' }]);
  assert.equal(show('Beat-off, beat off, beat it, darned, DARN.', rules), 'bleep*, bleep*, beat it, darned, bleep*.');
});
test('user rules override list replacements and exceptions remove list entries', () => {
  const lists = [{ entries: parseWordList('stupid\nblack\nidiot'), replacement: 'bleep', source: 'list' }];
  const rules = combineRules(parseRules('stupid = unhelpful'), lists, 'black');
  assert.equal(show('stupid black idiot', rules), 'unhelpful* black bleep*');
  assert.equal(segments('idiot', rules)[0].source, 'list');
});
test('vendored CMU list matches its manifest checksum and is cited', () => {
  const manifest = JSON.parse(readFileSync('docs/lexicons/sources.json', 'utf8'));
  for (const source of manifest.sources) {
    const bytes = readFileSync(`docs/lexicons/${source.file}`);
    assert.equal(createHash('sha256').update(bytes).digest('hex'), source.sha256);
    assert.equal(parseWordList(bytes.toString('utf8')).length, source.entries);
    assert.ok(source.citation.includes(source.url));
  }
  assert.equal(manifest.sources[0].url, 'https://www.cs.cmu.edu/~biglou/resources/bad-words.txt');
});
