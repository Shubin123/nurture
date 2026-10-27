"""Evaluate a frozen validation-selected cascade against served quality references."""

import argparse
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from cascade import Cascade, CascadeConfig
from common import ROOT, RUNS, character_metrics, digest, load_split, predict_characters, write_json
from evaluate import neural_scores, summarize
from train_cascade import local_tokenizer, metrics
from train_linear import matrix, unpack
from train_neural import Chunks


def merge_scores(chunks, rows, scores):
    buckets = [{} for _ in rows]
    for index, values in enumerate(scores):
        source = chunks.encoding['overflow_to_sample_mapping'][index]
        for offset, value in zip(chunks.encoding['offset_mapping'][index], values):
            if offset[1] <= offset[0]:
                continue
            key = tuple(offset)
            total, count = buckets[source].get(key, (0.0, 0))
            buckets[source][key] = (total + float(value), count + 1)
    predictions = [predict_characters(list(bucket), [total / count for total, count in bucket.values()], 0.5) for bucket in buckets]
    return character_metrics(rows, predictions)


@torch.inference_mode()
def measure(model, chunks, batch_size, temperatures, thresholds, seed):
    model.eval()
    scores, full_scores, labels, exits, predictions, full_predictions = [], [], [], [], [], []
    times, full_times = [], []
    randomizer = random.Random(seed)
    first = next(iter(DataLoader(chunks, batch_size=batch_size)))
    force_final = [1.0] * (len(model.heads) - 1) + [0.0]
    for _ in range(5):
        model.predict(first['input_ids'], first['attention_mask'], temperatures, thresholds)
        model.predict(first['input_ids'], first['attention_mask'], temperatures, force_final)
    maximum_lookup_difference = 0.0
    for batch in DataLoader(chunks, batch_size=batch_size):
        options = [('cascade', thresholds), ('full', force_final)]
        randomizer.shuffle(options)
        outputs = {}
        for name, gate in options:
            started = time.perf_counter()
            values, levels = model.predict(batch['input_ids'], batch['attention_mask'], temperatures, gate)
            (times if name == 'cascade' else full_times).append(time.perf_counter() - started)
            outputs[name] = (values, levels)
        values, levels = outputs['cascade']
        full, _ = outputs['full']
        original_lookup = model.config.lookup
        model.config.lookup = 'reference'
        reference, _ = model.predict(batch['input_ids'], batch['attention_mask'], temperatures, force_final)
        model.config.lookup = original_lookup
        maximum_lookup_difference = max(maximum_lookup_difference, float((reference - full).abs().max()))
        valid = batch['labels'].ne(-100)
        labels.append(batch['labels'][valid])
        predictions.append(values.argmax(-1)[valid])
        full_predictions.append(full.argmax(-1)[valid])
        scores.extend(values[:, :, 1])
        full_scores.extend(full[:, :, 1])
        exits.extend(levels.tolist())
    document_exits = [0] * len(set(chunks.encoding['overflow_to_sample_mapping']))
    for source, level in zip(chunks.encoding['overflow_to_sample_mapping'], exits):
        document_exits[source] = max(document_exits[source], level)
    return {'token_metrics': metrics(torch.cat(labels), torch.cat(predictions)),
            'forced_final_token_metrics': metrics(torch.cat(labels), torch.cat(full_predictions)),
            'chunk_exit_fractions': [exits.count(level) / len(exits) for level in range(len(model.heads))],
            'document_max_exit_fractions': [document_exits.count(level) / len(document_exits) for level in range(len(model.heads))],
            'native_model_seconds': sum(times), 'forced_final_native_model_seconds': sum(full_times),
            'forced_final_over_cascade_time_ratio': sum(full_times) / sum(times),
            'processed_labelled_chunk_tokens_per_second': sum(len(values) for values in labels) / sum(times),
            'lookup_reference_max_probability_difference': maximum_lookup_difference}, scores, full_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', default='cascade-kernel-space')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('Batch size must be positive.')
    torch.set_num_threads(4)
    selection_path = ROOT / f'docs/reports/{args.selection}-selection.json'
    selection = json.loads(selection_path.read_text())
    checkpoint = RUNS / args.selection / 'selected.pt'
    if digest(checkpoint) != selection['checkpoint_sha256'] or selection['test_used']:
        raise ValueError('Expected an unchanged validation-only selection and checkpoint.')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = Cascade(CascadeConfig(**saved['config']))
    model.load_state_dict(saved['state_dict'])
    rows = load_split('test', selection['args']['profile'])
    chunks = Chunks(rows, local_tokenizer())
    result, scores, full_scores = measure(model, chunks, args.batch_size, saved['temperatures'], saved['thresholds'], selection['args']['seed'])
    result['character_metrics'] = merge_scores(chunks, rows, scores)
    result['forced_final_character_metrics'] = merge_scores(chunks, rows, full_scores)
    references = {}
    directory = ROOT / 'docs/models/linear-balanced'
    config = json.loads((directory / 'config.json').read_text())
    weights = np.frombuffer((directory / 'weights.int8').read_bytes(), dtype=np.int8).astype(np.float32) * config['scale']
    sparse, _, offsets = matrix(rows)
    probabilities = unpack(1 / (1 + np.exp(-np.clip(sparse @ weights + config['bias'], -50, 50))), offsets)
    references['served_linear_balanced'] = summarize(rows, offsets, probabilities, config['threshold'])
    directory = ROOT / 'docs/models/bert-mini-balanced'
    config = json.loads((directory / 'inference.json').read_text())
    offsets, probabilities = neural_scores(directory / 'model.int8.onnx', chunks, rows)
    references['served_mini_balanced_native_int8'] = summarize(rows, offsets, probabilities, config['threshold'])
    report = {'selection_sha256': digest(selection_path), 'checkpoint_sha256': digest(checkpoint), 'profile': selection['args']['profile'],
              'examples': len(rows), 'selected': selection['selected'], 'result': result, 'references': references,
              'source_sha256': {name: digest(ROOT / 'ml' / name) for name in ('cascade.py', 'vq_search.py', 'evaluate_cascade_search.py')},
              'reference_artifact_sha256': {name: digest(ROOT / name) for name in ('docs/models/linear-balanced/weights.int8', 'docs/models/linear-balanced/config.json', 'docs/models/bert-mini-balanced/model.int8.onnx', 'docs/models/bert-mini-balanced/inference.json')},
              'limitations': ['Frozen validation selection; test outcomes do not change candidates or calibration. Old repository test set is not a fresh independent holdout.',
                              'Served references have different training data/budgets and frozen thresholds. Native ONNX is not browser inference.',
                              'Single-pass native timings exclude tokenization, transfer, aggregation and reference-equivalence checks. Not browser throughput.',
                              'Lookup microbenchmark improvements do not establish end-to-end or state-of-the-art quality advantages.']}
    write_json(ROOT / f'docs/reports/{args.selection}-evaluation.json', report)


if __name__ == '__main__':
    main()
