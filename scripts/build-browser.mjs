import { build } from 'esbuild';
import { copyFile, mkdir } from 'node:fs/promises';

await mkdir('docs/vendor', { recursive: true });
await build({ entryPoints: ['ml/browser-worker.js'], bundle: true, minify: true, format: 'esm', platform: 'browser', target: 'es2022', outfile: 'docs/model-worker.js', legalComments: 'external' });
await build({ entryPoints: ['ml/asr-worker.js'], bundle: true, minify: true, format: 'esm', platform: 'browser', target: 'es2022', outfile: 'docs/asr-worker.js', legalComments: 'external' });
for (const filename of ['ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']) await copyFile(`node_modules/onnxruntime-web/dist/${filename}`, `docs/vendor/${filename}`);
await copyFile('licenses/ONNX-RUNTIME-LICENSE', 'docs/vendor/ONNX-RUNTIME-LICENSE');
await copyFile('licenses/ONNX-RUNTIME-THIRD-PARTY-NOTICES', 'docs/vendor/ONNX-RUNTIME-THIRD-PARTY-NOTICES');
console.log('Built self-hosted workers and WASM assets. No CDN or Actions required.');
