import argparse
import importlib.metadata
import json
import time

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier

from common import REPORTS, ROOT, RUNS, SEED, character_metrics, features, load_split, predict_characters, token_spans, tune_threshold, write_json


def matrix(rows):
    indices, pointers, labels, offsets = [], [0], [], []
    for row in rows:
        tokens = token_spans(row['text'])
        gold = set(row['spans'])
        offsets.append([(start, end) for _, start, end in tokens])
        for index, (_, start, end) in enumerate(tokens):
            indices.extend(features(tokens, index))
            pointers.append(len(indices))
            labels.append(int(sum(position in gold for position in range(start, end)) >= (end - start) / 2))
    sparse = csr_matrix((np.ones(len(indices), dtype=np.float32), indices, pointers), shape=(len(labels), 65536))
    return sparse, np.array(labels), offsets


def unpack(values, offsets):
    result = []
    position = 0
    for spans in offsets:
        result.append(values[position:position + len(spans)])
        position += len(spans)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=['original', 'balanced'], default='original')
    args = parser.parse_args()
    suffix = '-balanced' if args.profile == 'balanced' else ''
    criterion = 'balanced_context' if args.profile == 'balanced' else 'macro_character_f1'
    started = time.perf_counter()
    train, validation = load_split('train', args.profile), load_split('validation', args.profile)
    train_matrix, labels, _ = matrix(train)
    valid_matrix, _, offsets = matrix(validation)
    candidates = []
    RUNS.mkdir(parents=True, exist_ok=True)
    best_model = None
    for alpha in (1e-5, 1e-4, 1e-3):
        model = SGDClassifier(loss='log_loss', penalty='l2', alpha=alpha, max_iter=50, tol=1e-5, random_state=SEED, average=True)
        model.fit(train_matrix, labels)
        probabilities = unpack(model.predict_proba(valid_matrix)[:, 1], offsets)
        best, curve = tune_threshold(validation, offsets, probabilities, criterion)
        candidates.append({'alpha': alpha, **best})
        if best_model is None or best['selection_score'] > best_model[0]:
            best_model = (best['selection_score'], model, best['threshold'], curve)
    _, model, threshold, curve = best_model
    weights = model.coef_[0].astype(np.float32)
    scale = float(np.max(np.abs(weights)) / 127)
    quantized = np.clip(np.rint(weights / scale), -127, 127).astype(np.int8)
    probabilities = 1 / (1 + np.exp(-np.clip(valid_matrix @ (quantized.astype(np.float32) * scale) + model.intercept_[0], -50, 50)))
    quantized_threshold, quantized_curve = tune_threshold(validation, offsets, unpack(probabilities, offsets), criterion)
    destination = (RUNS / 'candidate-models' if args.profile == 'balanced' else ROOT / 'docs' / 'models') / f'linear{suffix}'
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'weights.int8').write_bytes(quantized.tobytes())
    write_json(destination / 'config.json', {'model': 'Nurture hashed character-context logistic regression', 'dimensions': 65536, 'scale': scale, 'bias': float(model.intercept_[0]), 'threshold': quantized_threshold['threshold'], 'hash': 'FNV-1a UTF-8 unsigned32', 'version': 1})
    np.savez(RUNS / f'linear{suffix}.npz', weights=weights, bias=model.intercept_, threshold=threshold)
    write_json(REPORTS / f'linear{suffix}-training.json', {'seed': SEED, 'profile': args.profile, 'selection_criterion': criterion, 'train_tokens': len(labels), 'positive_tokens': int(sum(labels)), 'candidates': candidates, 'validation_curve': curve, 'quantized_validation_curve': quantized_curve, 'threshold': threshold, 'quantized_threshold': quantized_threshold['threshold'], 'training_seconds': time.perf_counter() - started, 'sklearn': importlib.metadata.version('scikit-learn')})
    print(json.dumps({'candidates': candidates, 'quantized_validation': quantized_threshold, 'seconds': time.perf_counter() - started}, indent=2))


if __name__ == '__main__':
    main()
