import argparse
import json

import numpy as np

from common import REPORTS, RUNS, character_metrics, load_split, write_json

parser = argparse.ArgumentParser()
parser.add_argument('--profile', choices=['original', 'balanced', 'deployed'], default='deployed')
args = parser.parse_args()
released = args.profile == 'deployed' and (REPORTS / 'release.json').exists()
balanced = args.profile == 'balanced' or released
prefix = 'balanced-' if args.profile == 'balanced' else 'release-' if released else ''
rows = load_split('test', 'balanced' if balanced else 'original')
raw = json.loads((RUNS / f'{prefix}browser-predictions.json').read_text())
native = json.loads((REPORTS / ('balanced-evaluation.json' if balanced else 'evaluation.json')).read_text())
report = {'profile': raw.get('profile', 'deployed'), 'assets': raw.get('assets', []), 'worker_sha256': raw['worker_sha256'], 'browser': raw['browser'], 'held_out_examples': len(rows), 'runtime': 'ONNX Runtime Web single-thread WASM + JavaScript baseline', 'models': {}}
for name, values in raw['models'].items():
    assert [row['id'] for row in rows] == [value['id'] for value in values]
    metric = character_metrics(rows, [value['predicted'] for value in values])
    merged = character_metrics(rows, [value['merged_word_predictions'] for value in values])
    native_name = ('mini_balanced' if name.startswith('bert-mini') else 'linear_balanced') if balanced else (name.replace('-', '_') + '_int8' if name != 'linear' else 'linear_int8')
    native_score = native['models'][native_name]['test']['combined' if balanced else 'macro_character_f1']
    native_f1 = native_score['macro_character_f1'] if balanced else native_score
    report['models'][name] = {
        'threshold': values[0]['threshold'], 'subword_character_metrics': metric,
        'display_whole_word_metrics': merged,
        'native_macro_character_f1': native_f1,
        'browser_minus_native_macro_f1': metric['macro_character_f1'] - native_f1,
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
    if balanced:
        entry = report['models'][name]
        for subset, rated_negative in [('original_span_subset', False), ('rated_non_toxic_subset', True)]:
            indices = [index for index, row in enumerate(rows) if (row.get('source') == 'Civil Comments zero-rated negative') == rated_negative]
            selected_rows = [rows[index] for index in indices]
            selected = [values[index] for index in indices]
            entry[subset] = character_metrics(selected_rows, [value['predicted'] for value in selected])
            entry[subset]['display_whole_word_metrics'] = character_metrics(selected_rows, [value['merged_word_predictions'] for value in selected])
            if rated_negative:
                entry[subset]['post_level_false_positive_rate'] = sum(bool(value['predicted']) for value in selected) / len(selected)
                entry[subset]['display_post_level_false_positive_rate'] = sum(bool(value['merged_word_predictions']) for value in selected) / len(selected)
        entry['benchmark_reuse_note'] = native['benchmark_reuse_note']
write_json(REPORTS / f'{prefix}browser-evaluation.json', report)
print(json.dumps(report, indent=2))
