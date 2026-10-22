import json

import numpy as np

from common import REPORTS, RUNS, character_metrics, load_split, write_json

rows = load_split('test')
raw = json.loads((RUNS / 'browser-predictions.json').read_text())
native = json.loads((REPORTS / 'evaluation.json').read_text())
report = {'worker_sha256': raw['worker_sha256'], 'browser': raw['browser'], 'held_out_examples': len(rows), 'runtime': 'ONNX Runtime Web single-thread WASM + JavaScript baseline', 'models': {}}
for name, values in raw['models'].items():
    assert [row['id'] for row in rows] == [value['id'] for value in values]
    metric = character_metrics(rows, [value['predicted'] for value in values])
    merged = character_metrics(rows, [value['merged_word_predictions'] for value in values])
    native_name = name.replace('-', '_') + '_int8' if name != 'linear' else 'linear_int8'
    report['models'][name] = {
        'threshold': values[0]['threshold'], 'subword_character_metrics': metric,
        'display_whole_word_metrics': merged,
        'native_macro_character_f1': native['models'][native_name]['test']['macro_character_f1'],
        'browser_minus_native_macro_f1': metric['macro_character_f1'] - native['models'][native_name]['test']['macro_character_f1'],
        'held_out_worker_processing': {
            'total_useful_tokens': sum(value['tokens'] for value in values),
            'total_processing_ms': sum(value['elapsed_ms'] for value in values),
            'useful_tokens_per_second': sum(value['tokens'] for value in values) / (sum(value['elapsed_ms'] for value in values) / 1000),
            'per_document_p50_ms': float(np.median([value['elapsed_ms'] for value in values])),
            'per_document_p95_ms': float(np.quantile([value['elapsed_ms'] for value in values], 0.95)),
            'scope': 'Actual held-out comments with variable lengths, no warmup, separate from fixed-size warm latency benchmark.',
        },
        'note': 'Browser inference batches overlapping chunks within a document. Native evaluation batches chunks across documents. Dynamic quantization can produce small differences. Display merges contiguous subwords using maximum score; its accuracy is reported separately.',
    }
write_json(REPORTS / 'browser-evaluation.json', report)
print(json.dumps(report, indent=2))
