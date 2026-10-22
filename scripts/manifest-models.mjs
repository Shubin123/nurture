import { createHash } from 'node:crypto';
import { readFileSync, readdirSync, writeFileSync } from 'node:fs';

function files(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap(entry => entry.isDirectory() ? files(`${directory}/${entry.name}`) : [`${directory}/${entry.name}`]);
}
const artifacts = [...files('docs/models'), ...files('docs/vendor'), 'docs/model-worker.js', 'docs/model-worker.js.LEGAL.txt'];
const manifest = {
  format: 1,
  runtime: JSON.parse(readFileSync('node_modules/onnxruntime-web/package.json', 'utf8')).version,
  assets: artifacts.map(path => {
    const data = readFileSync(path);
    return { path: path.replace(/^docs\//, ''), bytes: data.length, sha256: createHash('sha256').update(data).digest('hex') };
  }),
};
writeFileSync('docs/reports/artifacts.json', JSON.stringify(manifest, null, 2) + '\n');
console.log(`${manifest.assets.length} artifacts, ${manifest.assets.reduce((total, value) => total + value.bytes, 0)} bytes.`);
