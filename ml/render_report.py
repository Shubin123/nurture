import json

from common import REPORTS

native = json.loads((REPORTS / 'evaluation.json').read_text())
browser = json.loads((REPORTS / 'browser-evaluation.json').read_text())
performance = json.loads((REPORTS / 'browser-benchmark.json').read_text())
lines = [
    '# Measured model results', '',
    'Local measurements, not deployment or safety guarantees.', '',
    'Hardware: Apple M4, 24 GiB RAM. Headless Chrome ' + browser['browser'] + ', CPU-only, one WASM thread. Native ONNX Runtime uses two threads.', '',
    'Dataset: 7,917 training, 689 validation, 2,000 held-out test comments after normalized deduplication. Checkpoints and thresholds use validation only.', '',
    '| Model | Weight/model bytes | Browser test character F1 | Whole-word display F1 | p95 ms / 128 useful tokens | Useful tokens/s / 1,024-token document |',
    '| --- | ---: | ---: | ---: | ---: | ---: |',
]
for name in ('linear', 'bert-tiny', 'bert-mini'):
    native_name = name.replace('-', '_') + '_int8' if name != 'linear' else 'linear_int8'
    value = browser['models'][name]
    timing = performance['models'][name]['benchmarks']
    small = next(item for item in timing if item['nominal_input_tokens'] == 128)
    large = next(item for item in timing if item['nominal_input_tokens'] == 1024)
    lines.append(f"| {name} | {native['models'][native_name]['bytes']:,} | {value['subword_character_metrics']['macro_character_f1']:.4f} | {value['display_whole_word_metrics']['macro_character_f1']:.4f} | {small['worker']['p95_ms']:.1f} | {large['useful_tokens_per_second']:,.0f} |")
lines.extend([
    '', '## Interpretation', '',
    '- F1 is macro per-document character-span F1, not classification accuracy. Whole-word display expansion has separate metrics.',
    '- The character model is the recommended throughput baseline. Mini offers a modest contextual benchmark improvement at substantially higher compute cost. Tiny is not a demonstrated accuracy upgrade over the character model.',
    '- Warm speed tests use a repeated English sentence, five warmups and 50 runs. Actual variable-length held-out throughput is in browser-evaluation.json. Useful tokens exclude overlap and padding.',
    '- 126 useful WordPieces plus two special tokens fit one neural chunk; 128 useful WordPieces require two overlapping chunks.',
    '- Native/browser dynamic quantization and batching differ slightly. Quantization reduces weights about fourfold, but is not uniformly faster on native CPU.',
    '- Localhost initialization is not a real Internet cold-download measurement. Neural models additionally need the shared approximately 14 MiB WASM runtime and vocabulary.',
    '- Whole-Chrome RSS sampling is a noisy process-tree proxy and can double-count shared pages. The 150 MB additional-model-memory target is not proven; no memory pass is claimed.',
    '- Empty gold spans are not benign labels: entire toxic comments can lack localized annotations. The small authored challenge is diagnostic, not independent fairness or production false-positive validation.',
    '- The 99% precision / low benign-false-positive release gate is NOT met. Models stay opt-in and review-only; only user-approved rules supply replacements.',
    '', '## Evidence', '',
    '- evaluation.json: native FP32/INT8 accuracy, document-bootstrap confidence intervals and length slices.',
    '- browser-evaluation.json: all 2,000 held-out comments evaluated through the served browser worker.',
    '- browser-benchmark.json: worker/roundtrip latency, batch sweeps, responsiveness and process-tree RSS proxy.',
    '- browser-benchmark-baseline.json: pre-static-sequence measurement; differences are small/noisy, not a claimed speedup.',
    '- challenge.json: provisional benign, identity, quotation/negation and direct-insult examples with predictions.',
    '- tokenizer-conformance.json: token IDs, Unicode offsets and hashed features agree on 695 validation/diagnostic fixtures.',
    '- dataset.json, frozen-selection.json, *-training.json, *-export.json and artifacts.json: provenance, selection, training and integrity records.',
])
(REPORTS / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines[:16]))
