import json
import math

import numpy as np
from transformers import AutoTokenizer

from common import REPORTS, ROOT, RUNS, character_metrics, digest, load_split, predict_characters, tune_threshold, write_json
from evaluate import neural_scores
from train_linear import matrix, unpack
from train_neural import Chunks


def summary(rows, offsets, probabilities, threshold):
    predictions = [predict_characters(span, score, threshold) for span, score in zip(offsets, probabilities)]
    result = {'threshold': threshold, 'combined': character_metrics(rows, predictions)}
    for name, is_benign in [('original_span_subset', False), ('rated_non_toxic_subset', True)]:
        indices = [index for index, row in enumerate(rows) if (row.get('source') == 'Civil Comments zero-rated negative') == is_benign]
        selected_rows = [rows[index] for index in indices]
        selected_predictions = [predictions[index] for index in indices]
        result[name] = character_metrics(selected_rows, selected_predictions)
        if is_benign:
            count = len(indices)
            flagged = sum(bool(prediction) for prediction in selected_predictions)
            rate = flagged / count
            denominator = 1 + 1.96 ** 2 / count
            center = (rate + 1.96 ** 2 / (2 * count)) / denominator
            margin = 1.96 * math.sqrt(rate * (1 - rate) / count + 1.96 ** 2 / (4 * count ** 2)) / denominator
            result[name]['post_level_false_positive_rate'] = rate
            result[name]['post_level_false_positive_rate_wilson_ci95'] = [max(0, center - margin), min(1, center + margin)]
            result[name]['interpretation'] = 'False positives relative to seven zero-rated Civil Comments post labels. Human ratings and this source are not universal safety truth.'
    return result


def main():
    validation, test = load_split('validation', 'balanced'), load_split('test', 'balanced')
    rows_by_split = {'validation': validation, 'test': test}
    matrices = {split: matrix(rows) for split, rows in rows_by_split.items()}
    report = {
        'dataset_manifest_sha256': digest(REPORTS / 'balanced-dataset.json'),
        'selection_criterion': 'Equal weight of original Toxic Spans validation macro character F1 and rated-non-toxic validation post acceptance; no test selection.',
        'models': {},
        'release_scope': 'Candidate research comparison. No automatic-filter production claim.',
        'benchmark_reuse_note': 'The original 2000-comment span test was inspected in the first experiment and is reused for comparison. The 5000 rated-negative test examples are a new frozen control set for this follow-up. Neither test subset selects thresholds.',
    }
    linear_models = {}
    for name, directory in [('linear_original', ROOT / 'docs/models/linear'), ('linear_balanced', RUNS / 'candidate-models/linear-balanced')]:
        config = json.loads((directory / 'config.json').read_text())
        weights = np.frombuffer((directory / 'weights.int8').read_bytes(), dtype=np.int8).astype(np.float32) * config['scale']
        linear_models[name] = (weights, config)
        entry = {'weight_bytes': int(weights.size), 'weight_sha256': digest(directory / 'weights.int8')}
        for split, rows in rows_by_split.items():
            if split != 'validation':
                continue
            sparse, _, offsets = matrices[split]
            probabilities = unpack(1 / (1 + np.exp(-np.clip(sparse @ weights + config['bias'], -50, 50))), offsets)
            entry[split] = summary(rows, offsets, probabilities, config['threshold'])
        report['models'][name] = entry
    tokenizer = AutoTokenizer.from_pretrained(RUNS / 'bert-mini-balanced', use_fast=True)
    chunks = {split: Chunks(rows, tokenizer) for split, rows in rows_by_split.items()}
    for name, directory in [('mini_original', ROOT / 'docs/models/bert-mini'), ('mini_balanced', RUNS / 'candidate-models/bert-mini-balanced')]:
        config_path = directory / 'inference.json'
        config = json.loads(config_path.read_text())
        path = directory / 'model.int8.onnx'
        entry = {'model_bytes': path.stat().st_size, 'model_sha256': digest(path)}
        offsets, probabilities = neural_scores(path, chunks['validation'], validation)
        if name == 'mini_balanced':
            selection, curve = tune_threshold(validation, offsets, probabilities, 'balanced_context')
            config['threshold'] = selection['threshold']
            write_json(config_path, config)
            write_json(REPORTS / 'balanced-frozen-selection.json', {'quantized_mini_selection': selection, 'validation_curve': curve, 'test_used_for_selection': False})
        entry['validation'] = summary(validation, offsets, probabilities, config['threshold'])
        report['models'][name] = entry
    write_json(REPORTS / 'balanced-validation-selection.json', report)
    for name, (weights, config) in linear_models.items():
        sparse, _, offsets = matrices['test']
        probabilities = unpack(1 / (1 + np.exp(-np.clip(sparse @ weights + config['bias'], -50, 50))), offsets)
        report['models'][name]['test'] = summary(test, offsets, probabilities, config['threshold'])
    for name, directory in [('mini_original', ROOT / 'docs/models/bert-mini'), ('mini_balanced', RUNS / 'candidate-models/bert-mini-balanced')]:
        config = json.loads((directory / 'inference.json').read_text())
        offsets, probabilities = neural_scores(directory / 'model.int8.onnx', chunks['test'], test)
        report['models'][name]['test'] = summary(test, offsets, probabilities, config['threshold'])
    write_json(REPORTS / 'balanced-evaluation.json', report)
    for name, value in report['models'].items():
        print(name, json.dumps({
            'span_test_macro_f1': value['test']['original_span_subset']['macro_character_f1'],
            'rated_non_toxic_test_false_positive_rate': value['test']['rated_non_toxic_subset']['post_level_false_positive_rate'],
            'threshold': value['test']['threshold'],
        }), flush=True)


if __name__ == '__main__':
    main()
