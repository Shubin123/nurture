import collections
import json
import platform
import time

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from common import REPORTS, ROOT, RUNS, character_metrics, digest, load_split, predict_characters, token_spans, tune_threshold, write_json
from train_linear import matrix, unpack
from train_neural import Chunks


def neural_scores(path, chunks, rows, batch_size=32):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    buckets = [{} for _ in rows]
    encoding = chunks.encoding
    for start in range(0, len(chunks), batch_size):
        feed = {key: np.array(encoding[key][start:start + batch_size], dtype=np.int64) for key in ('input_ids', 'attention_mask')}
        logits = session.run(None, feed)[0]
        probabilities = 1 / (1 + np.exp(np.clip(logits[:, :, 0] - logits[:, :, 1], -50, 50)))
        for local, values in enumerate(probabilities):
            chunk = start + local
            source = encoding['overflow_to_sample_mapping'][chunk]
            for offset, value in zip(encoding['offset_mapping'][chunk], values):
                if offset[1] <= offset[0]:
                    continue
                key = tuple(offset)
                total, count = buckets[source].get(key, (0, 0))
                buckets[source][key] = (total + float(value), count + 1)
    offsets, probabilities = [], []
    for bucket in buckets:
        ordered = sorted(bucket)
        offsets.append(ordered)
        probabilities.append([bucket[key][0] / bucket[key][1] for key in ordered])
    return offsets, probabilities


def summarize(rows, offsets, probabilities, threshold):
    predictions = [predict_characters(span, score, threshold) for span, score in zip(offsets, probabilities)]
    result = character_metrics(rows, predictions)
    result['threshold'] = threshold
    result['slices'] = {}
    for name, indices in {
        'long_500_plus_characters': [index for index, row in enumerate(rows) if len(row['text']) >= 500],
        'short_under_500_characters': [index for index, row in enumerate(rows) if len(row['text']) < 500],
        'no_gold_spans': [index for index, row in enumerate(rows) if not row['spans']],
    }.items():
        if indices:
            result['slices'][name] = character_metrics([rows[index] for index in indices], [predictions[index] for index in indices])
    result['automatic_replacement_gate'] = False
    result['automatic_replacement_gate_reason'] = 'Requires >=99% character precision, meaningful recall, and independently validated benign false-positive rate. Empty gold spans are not benign labels; production gate is not established by this benchmark.'
    return result


def native_benchmark(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    random = np.random.default_rng(1729)
    output = []
    for batch in (1, 8, 32):
        feed = {'input_ids': random.integers(1000, 20000, (batch, 128), dtype=np.int64), 'attention_mask': np.ones((batch, 128), dtype=np.int64)}
        for _ in range(10):
            session.run(None, feed)
        times = []
        for _ in range(100):
            started = time.perf_counter()
            session.run(None, feed)
            times.append((time.perf_counter() - started) * 1000)
        output.append({'batch': batch, 'sequence_length': 128, 'runs': 100, 'p50_ms': float(np.median(times)), 'p95_ms': float(np.quantile(times, 0.95)), 'wordpieces_per_second': float(batch * 128 / (np.mean(times) / 1000))})
    return output


def main():
    started = time.perf_counter()
    train, validation, test = load_split('train'), load_split('validation'), load_split('test')
    splits = {'validation': validation, 'test': test}
    report = {'dataset_manifest_sha256': digest(REPORTS / 'dataset.json'), 'host': platform.platform(), 'architecture': platform.machine(), 'native_runtime': ort.__version__, 'native_threads': 2, 'metric': 'Official-style macro per-document character F1; empty gold and prediction score 1. Micro metrics additionally reported.', 'models': {}}
    word_counts = collections.defaultdict(lambda: [0, 0])
    for row in train:
        gold = set(row['spans'])
        for word, start, end in token_spans(row['text']):
            word_counts[word.lower()][0] += int(sum(position in gold for position in range(start, end)) >= (end - start) / 2)
            word_counts[word.lower()][1] += 1
    lexicon = {word: positive / total for word, (positive, total) in word_counts.items() if total >= 3}
    validation_tokens = [token_spans(row['text']) for row in validation]
    dictionary_offsets = [[(start, end) for _, start, end in tokens] for tokens in validation_tokens]
    dictionary_scores = [[lexicon.get(word.lower(), 0) for word, _, _ in tokens] for tokens in validation_tokens]
    dictionary_selection, _ = tune_threshold(validation, dictionary_offsets, dictionary_scores)
    write_json(REPORTS / 'dictionary-selection.json', {'train_only_lexicon_entries': len(lexicon), 'minimum_occurrences': 3, 'validation_selection': dictionary_selection})
    linear_config = json.loads((ROOT / 'docs/models/linear/config.json').read_text())
    linear_float = np.load(RUNS / 'linear.npz')
    linear_quantized = np.frombuffer((ROOT / 'docs/models/linear/weights.int8').read_bytes(), dtype=np.int8).astype(np.float32) * linear_config['scale']
    neural_models = {}
    for size in ('tiny', 'mini'):
        tokenizer = AutoTokenizer.from_pretrained(RUNS / f'bert-{size}', use_fast=True)
        chunks = {split: Chunks(rows, tokenizer) for split, rows in splits.items()}
        selection = json.loads((RUNS / f'bert-{size}/selection.json').read_text())
        for kind, path in [('fp32', RUNS / f'bert-{size}-fp32.onnx'), ('int8', ROOT / f'docs/models/bert-{size}/model.int8.onnx')]:
            name = f'bert_{size}_{kind}'
            neural_models[name] = (path, chunks)
            offsets, probabilities = neural_scores(path, chunks['validation'], validation)
            threshold = selection['threshold']
            if kind == 'int8':
                selected, _ = tune_threshold(validation, offsets, probabilities)
                threshold = selected['threshold']
                config_path = ROOT / f'docs/models/bert-{size}/inference.json'
                config = json.loads(config_path.read_text())
                config['threshold'] = threshold
                write_json(config_path, config)
            report['models'][name] = {'bytes': path.stat().st_size, 'sha256': digest(path), 'validation': summarize(validation, offsets, probabilities, threshold), 'native_inference_only': native_benchmark(path)}
    write_json(REPORTS / 'frozen-selection.json', {'dictionary_threshold': dictionary_selection['threshold'], 'linear_fp32_threshold': float(linear_float['threshold']), 'linear_int8_threshold': linear_config['threshold'], 'neural_thresholds': {name: value['validation']['threshold'] for name, value in report['models'].items()}, 'test_used_for_selection': False})
    for split, rows in splits.items():
        sparse, _, offsets = matrix(rows)
        for name, weights, bias, threshold in [
            ('linear_fp32', linear_float['weights'], linear_float['bias'][0], float(linear_float['threshold'])),
            ('linear_int8', linear_quantized, linear_config['bias'], linear_config['threshold']),
        ]:
            probabilities = unpack(1 / (1 + np.exp(-np.clip(sparse @ weights + bias, -50, 50))), offsets)
            report['models'].setdefault(name, {'bytes': int(weights.size * (1 if name.endswith('int8') else 4))})[split] = summarize(rows, offsets, probabilities, threshold)
        tokens = [token_spans(row['text']) for row in rows]
        report['models'].setdefault('train_derived_dictionary', {'entries': len(lexicon)})[split] = summarize(rows, [[(start, end) for _, start, end in item] for item in tokens], [[lexicon.get(word.lower(), 0) for word, _, _ in item] for item in tokens], dictionary_selection['threshold'])
        if split == 'test':
            for name, (path, chunks) in neural_models.items():
                span, probability = neural_scores(path, chunks[split], rows)
                report['models'][name][split] = summarize(rows, span, probability, report['models'][name]['validation']['threshold'])
    report['elapsed_seconds'] = time.perf_counter() - started
    write_json(REPORTS / 'evaluation.json', report)
    for name, value in report['models'].items():
        print(name, json.dumps({key: value['test'][key] for key in ('macro_character_f1', 'micro_character_precision', 'micro_character_recall', 'empty_gold_prediction_rate', 'automatic_replacement_gate')}))


if __name__ == '__main__':
    main()
