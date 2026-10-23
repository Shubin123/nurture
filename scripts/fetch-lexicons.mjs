import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import NurtureFilter from '../docs/filter.js';

// Vendors every word list in docs/lexicons/sources.json. Pass --refresh to re-download existing files.
const manifestPath = 'docs/lexicons/sources.json';
const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
const refresh = process.argv.includes('--refresh');
for (const source of manifest.sources) {
  if (!/^[a-z0-9-]{1,64}$/.test(source.id)) throw new Error(`Invalid source id: ${source.id}`);
  if (!source.url || !source.citation) throw new Error(`${source.id}: url and citation are required.`);
  const path = `docs/lexicons/${source.id}.txt`;
  let bytes = refresh ? null : await readFile(path).catch(() => null);
  if (!bytes) {
    const response = await fetch(source.url);
    if (!response.ok) throw new Error(`${source.id}: HTTP ${response.status}`);
    bytes = Buffer.from(await response.arrayBuffer());
    source.retrieved = new Date().toISOString().slice(0, 10);
  }
  const sha256 = createHash('sha256').update(bytes).digest('hex');
  if (source.sha256 && source.sha256 !== sha256 && !refresh) throw new Error(`${source.id}: checksum mismatch. Use --refresh to accept a new upstream version.`);
  await writeFile(path, bytes);
  source.file = `${source.id}.txt`;
  source.sha256 = sha256;
  source.entries = NurtureFilter.parseWordList(bytes.toString('utf8')).length;
  console.log(`${source.id}: ${source.entries} entries, sha256 ${sha256.slice(0, 12)}…`);
}
await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
