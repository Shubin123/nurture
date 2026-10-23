import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';

// Pinned ONNX export of openai/whisper-tiny.en with cross-attention alignment heads for word timestamps.
const repo = 'onnx-community/whisper-tiny.en_timestamped';
const revision = 'aeaa13760958b03fac5062f457d317d3319c3168';
const target = 'docs/models/whisper-tiny-en';
const files = ['config.json', 'generation_config.json', 'vocab.json', 'onnx/encoder_model_quantized.onnx', 'onnx/decoder_model_merged_quantized.onnx'];
const checksums = {};
for (const file of files) {
  const path = `${target}/${file}`;
  let bytes = await readFile(path).catch(() => null);
  if (!bytes) {
    const response = await fetch(`https://huggingface.co/${repo}/resolve/${revision}/${file}`);
    if (!response.ok) throw new Error(`${file}: HTTP ${response.status}`);
    bytes = Buffer.from(await response.arrayBuffer());
    await mkdir(dirname(path), { recursive: true });
    await writeFile(path, bytes);
  }
  checksums[file] = { bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex') };
}
await writeFile('docs/reports/asr-model.json', `${JSON.stringify({ repo, revision, base_model: 'openai/whisper-tiny.en', license: 'Apache-2.0', files: checksums }, null, 2)}\n`);
console.log(`Fetched ${files.length} pinned ASR files into ${target}.`);
