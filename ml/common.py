import hashlib
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'ml' / 'data'
RUNS = ROOT / 'ml' / 'runs'
REPORTS = ROOT / 'docs' / 'reports'
SEED = 1729
TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[^\s]", re.UNICODE)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def load_split(split, profile='original'):
    if profile not in ('original', 'balanced'):
        raise ValueError('Unknown dataset profile.')
    path = DATA / ('balanced' if profile == 'balanced' else '') / f'{split}.jsonl'
    manifest = json.loads((REPORTS / ('balanced-dataset.json' if profile == 'balanced' else 'dataset.json')).read_text())
    if digest(path) != manifest['splits'][split]['sha256']:
        raise ValueError(f'Dataset checksum mismatch for {split}; run prepare.py from verified source files.')
    return [json.loads(line) for line in path.read_text().splitlines()]


def token_spans(text):
    return [(match.group(), match.start(), match.end()) for match in TOKEN_PATTERN.finditer(text)]


def fnv(value, size=65536):
    result = 2166136261
    for byte in value.encode('utf-8'):
        result = ((result ^ byte) * 16777619) & 0xffffffff
    return result % size


def features(tokens, index):
    word = tokens[index][0].lower()
    previous = tokens[index - 1][0].lower() if index else '<BOS>'
    following = tokens[index + 1][0].lower() if index + 1 < len(tokens) else '<EOS>'
    values = [f'w={word}', f'p={previous}', f'n={following}', f'pw={previous}|{word}', f'wn={word}|{following}']
    wrapped = '^' + word + '$'
    for width in (2, 3, 4):
        values.extend(f'c={wrapped[start:start + width]}' for start in range(len(wrapped) - width + 1))
    return sorted({fnv(value) for value in values})


def character_metrics(rows, predictions):
    if not rows or len(rows) != len(predictions):
        raise ValueError('Metrics require a nonempty aligned prediction for every example.')
    scores = []
    document_counts = []
    true_positive = false_positive = false_negative = 0
    empty_gold = empty_gold_flagged = exact = 0
    for row, prediction in zip(rows, predictions):
        gold = set(row['spans'])
        predicted = set(prediction)
        overlap = len(gold & predicted)
        document_counts.append((overlap, len(predicted - gold), len(gold - predicted)))
        scores.append(2 * overlap / (len(gold) + len(predicted)) if gold or predicted else 1.0)
        true_positive += overlap
        false_positive += len(predicted - gold)
        false_negative += len(gold - predicted)
        exact += gold == predicted
        if not gold:
            empty_gold += 1
            empty_gold_flagged += bool(predicted)
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    random = np.random.default_rng(SEED)
    array = np.array(scores)
    counts = np.array(document_counts)
    boot, precision_boot, recall_boot = [], [], []
    for _ in range(1000):
        indices = random.integers(0, len(array), len(array))
        boot.append(float(np.mean(array[indices])))
        positive, extra, missed = np.sum(counts[indices], axis=0)
        precision_boot.append(positive / max(1, positive + extra))
        recall_boot.append(positive / max(1, positive + missed))
    return {
        'examples': len(rows), 'macro_character_f1': float(np.mean(scores)),
        'macro_character_f1_ci95': np.quantile(boot, [0.025, 0.975]).tolist(),
        'micro_character_precision': precision, 'micro_character_recall': recall,
        'micro_character_precision_ci95': np.quantile(precision_boot, [0.025, 0.975]).tolist(),
        'micro_character_recall_ci95': np.quantile(recall_boot, [0.025, 0.975]).tolist(),
        'confidence_interval_method': '1000 bootstrap resamples of documents, fixed seed; descriptive for this benchmark, not a deployment guarantee.',
        'micro_character_f1': 2 * precision * recall / max(1e-12, precision + recall),
        'exact_character_set_match': exact / len(rows),
        'true_positive_characters': true_positive, 'false_positive_characters': false_positive,
        'false_negative_characters': false_negative, 'empty_gold_examples': empty_gold,
        'empty_gold_examples_flagged': empty_gold_flagged,
        'empty_gold_prediction_rate': empty_gold_flagged / empty_gold if empty_gold else None,
        'empty_gold_caveat': 'No annotated span does not establish that a comment is benign. Some entire comments are toxic without attributable spans.',
    }


def predict_characters(offsets, probabilities, threshold):
    predicted = set()
    for (start, end), probability in zip(offsets, probabilities):
        if probability >= threshold and end > start:
            predicted.update(range(int(start), int(end)))
    return sorted(predicted)


def tune_threshold(rows, offsets, probabilities, criterion='macro_character_f1'):
    if criterion not in ('macro_character_f1', 'balanced_context'):
        raise ValueError('Unknown threshold-selection criterion.')
    results = []
    for threshold in np.arange(0.10, 0.96, 0.025):
        predictions = [predict_characters(span, score, threshold) for span, score in zip(offsets, probabilities)]
        scores = []
        for row, prediction in zip(rows, predictions):
            gold, predicted = set(row['spans']), set(prediction)
            scores.append(2 * len(gold & predicted) / (len(gold) + len(predicted)) if gold or predicted else 1.0)
        entry = {'threshold': round(float(threshold), 3), 'macro_character_f1': float(np.mean(scores))}
        if criterion == 'balanced_context':
            span_scores = [score for row, score in zip(rows, scores) if row.get('source') != 'Civil Comments zero-rated negative']
            benign_scores = [score for row, score in zip(rows, scores) if row.get('source') == 'Civil Comments zero-rated negative']
            if not span_scores or not benign_scores:
                raise ValueError('Balanced-context selection requires both original span and rated-benign validation examples.')
            entry['span_subset_macro_f1'] = float(np.mean(span_scores))
            entry['rated_benign_acceptance'] = float(np.mean(benign_scores))
            entry['selection_score'] = (entry['span_subset_macro_f1'] + entry['rated_benign_acceptance']) / 2
        else:
            entry['selection_score'] = entry['macro_character_f1']
        results.append(entry)
    best = max(results, key=lambda item: (item['selection_score'], item['threshold']))
    return best, results
