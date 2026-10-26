"""Compare frozen served classifiers on exactly the cascade pilot's sampled test rows."""

import argparse
import json
import random

import numpy as np

from common import ROOT, digest, load_split, write_json
from evaluate import neural_scores, summarize
from train_cascade import local_tokenizer
from train_linear import matrix, unpack
from train_neural import Chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True)
    parser.add_argument('--output', default='cascade-reference')
    args = parser.parse_args()
    source = ROOT / args.report
    experiment = json.loads(source.read_text())
    protocol = experiment['args']
    rows = load_split('test', protocol['profile'])
    random.Random(protocol['seed']).shuffle(rows)
    if protocol['limit']:
        rows = rows[:protocol['limit']]
    report = {'cascade_report_sha256': digest(source), 'examples': len(rows), 'models': {},
              'scope': 'Frozen served models, same sampled documents. Different training budgets; not an optimizer comparison. No throughput claim.'}
    suffix = '-balanced' if protocol['profile'] == 'balanced' else ''
    directory = ROOT / f'docs/models/linear{suffix}'
    config = json.loads((directory / 'config.json').read_text())
    weights = np.frombuffer((directory / 'weights.int8').read_bytes(), dtype=np.int8).astype(np.float32) * config['scale']
    sparse, _, offsets = matrix(rows)
    probabilities = unpack(1 / (1 + np.exp(-np.clip(sparse @ weights + config['bias'], -50, 50))), offsets)
    report['models'][f'linear{suffix}'] = summarize(rows, offsets, probabilities, config['threshold'])
    directory = ROOT / f'docs/models/bert-mini{suffix}'
    config = json.loads((directory / 'inference.json').read_text())
    offsets, probabilities = neural_scores(directory / 'model.int8.onnx', Chunks(rows, local_tokenizer()), rows)
    report['models'][f'bert-mini{suffix}'] = summarize(rows, offsets, probabilities, config['threshold'])
    write_json(ROOT / f'docs/reports/{args.output}.json', report)


if __name__ == '__main__':
    main()
