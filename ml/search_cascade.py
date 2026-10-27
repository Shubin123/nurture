"""Validation-only successive-halving search; never reads the test split."""

import argparse
from dataclasses import asdict
import math
import random
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from cascade import Cascade, CascadeConfig, calibrate, select_gates
from common import ROOT, RUNS, SEED, digest, load_split, write_json
from train_cascade import collect, local_tokenizer, metrics
from train_neural import Chunks


def partition_validation(rows, seed):
    if len(rows) < 8:
        raise ValueError('At least eight validation documents are required.')
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    middle, last = len(rows) // 2, 3 * len(rows) // 4
    return [[rows[index] for index in part] for part in (indices[:middle], indices[middle:last], indices[last:])]


def successive_halving(identifiers, budgets, evaluate, factor=2):
    if not identifiers or len(set(identifiers)) != len(identifiers) or factor < 2 or not budgets or budgets[0] < 1 or any(right <= left for left, right in zip(budgets, budgets[1:])):
        raise ValueError('Need unique trials and strictly increasing positive budgets.')
    active = sorted(identifiers)
    history = []
    for budget in budgets:
        entries = []
        for identifier in active:
            result = evaluate(identifier, budget)
            if not math.isfinite(result['objective']):
                raise ValueError('Search objective must be finite.')
            entries.append({'trial': identifier, 'budget': budget, **result})
        entries.sort(key=lambda entry: (-entry['objective'], entry['trial']))
        history.extend(entries)
        active = [entry['trial'] for entry in entries[:max(1, math.ceil(len(entries) / factor))]]
    return active[0], history


def definitions():
    return {
        'coarse-euclidean': {'slots': (2, 4, 8), 'codes': 64, 'code_width': 0, 'metric': 'euclidean'},
        'fine-euclidean': {'slots': (8, 32, 128), 'codes': 64, 'code_width': 0, 'metric': 'euclidean'},
        'compressed-cosine': {'slots': (8, 16, 32), 'codes': 128, 'code_width': 16, 'metric': 'cosine'},
        'fine-cosine': {'slots': (8, 32, 128), 'codes': 128, 'code_width': 16, 'metric': 'cosine'},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experimental-cascade', required=True, action='store_true')
    parser.add_argument('--profile', choices=('original', 'balanced'), default='original')
    parser.add_argument('--train-limit', type=int, default=4096)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--budgets', type=int, nargs='+', default=[2, 6, 12])
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--output', default='cascade-kernel-space')
    args = parser.parse_args()
    if args.train_limit < 0 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error('Invalid training limit, batch size or learning rate.')
    torch.set_num_threads(4)
    tokenizer = local_tokenizer()
    train_rows = load_split('train', args.profile)
    random.Random(args.seed).shuffle(train_rows)
    if args.train_limit:
        train_rows = train_rows[:args.train_limit]
    selection_rows, temperature_rows, gate_rows = partition_validation(load_split('validation', args.profile), args.seed)
    chunks = Chunks(train_rows, tokenizer)
    selection = Chunks(selection_rows, tokenizer)
    temperature = Chunks(temperature_rows, tokenizer)
    gates = Chunks(gate_rows, tokenizer)
    labels = torch.stack([item['labels'] for item in chunks.items])
    positives, negatives = int(labels.eq(1).sum()), int(labels.eq(0).sum())
    if not positives or not negatives:
        raise ValueError('Both span classes must occur in training.')
    class_weight = torch.tensor([1.0, min(20.0, math.sqrt(negatives / positives))])
    device = torch.device('cpu')
    trials = {}
    for identifier, values in definitions().items():
        torch.manual_seed(args.seed)
        config = CascadeConfig(vocabulary=len(tokenizer), commitment=0.05, lookup='dot', **values)
        model = Cascade(config)
        first = next(iter(DataLoader(chunks, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))))
        model.initialize_codebooks(first['input_ids'], first['attention_mask'])
        trials[identifier] = {'model': model, 'optimizer': torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01), 'epochs': 0, 'seconds': 0.0, 'steps': 0}
    destination = RUNS / args.output
    destination.mkdir(parents=True, exist_ok=True)

    def evaluate(identifier, budget):
        trial = trials[identifier]
        model, optimizer = trial['model'], trial['optimizer']
        started = time.perf_counter()
        usage = [torch.zeros(model.config.codes, dtype=torch.long) for _ in model.heads]
        for epoch in range(trial['epochs'], budget):
            model.train()
            loader = DataLoader(chunks, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed + epoch))
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                scores, _, commitment, codes = model(batch['input_ids'], batch['attention_mask'], reconstruct=False)
                labels = batch['labels']
                loss = sum(weight * F.cross_entropy(values.flatten(0, 1), labels.flatten(), weight=class_weight, ignore_index=-100)
                           for weight, values in zip((0.1, 0.2, 0.7), scores)) + 0.1 * commitment
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                trial['steps'] += 1
                for counts, values in zip(usage, codes):
                    counts += torch.bincount(values.detach(), minlength=model.config.codes)
        trial['seconds'] += time.perf_counter() - started
        trial['epochs'] = budget
        scores, labels = collect(model, selection, device, args.batch_size)
        valid = labels.ne(-100)
        result = metrics(labels[valid], scores[-1].argmax(-1)[valid])
        entry = {'objective': result['f1'], 'token_metrics': result, 'training_seconds': trial['seconds'], 'training_steps': trial['steps'],
                 'codebooks_last_increment': [{'used': int(counts.gt(0).sum()), 'histogram': counts.tolist(),
                                               'perplexity': float((-(counts.float() / counts.sum()).clamp_min(1e-12).log().mul(counts.float() / counts.sum()).sum()).exp())} for counts in usage]}
        print({'trial': identifier, 'epoch': budget, 'positive_f1': result['f1'], 'seconds': trial['seconds']}, flush=True)
        return entry

    winner, history = successive_halving(list(trials), args.budgets, evaluate)
    model = trials[winner]['model']
    scores, labels = collect(model, temperature, device, args.batch_size)
    temperatures = calibrate(scores, labels, labels.ne(-100))
    scores, labels = collect(model, gates, device, args.batch_size)
    thresholds = select_gates(scores, labels, labels.ne(-100), temperatures)
    checkpoint = destination / 'selected.pt'
    torch.save({'config': asdict(model.config), 'state_dict': model.state_dict(), 'temperatures': temperatures, 'thresholds': thresholds}, checkpoint)
    selection = {'args': vars(args), 'selected': winner, 'selected_config': asdict(model.config), 'temperatures': temperatures, 'thresholds': thresholds,
                 'checkpoint_sha256': digest(checkpoint), 'history': history, 'test_used': False, 'selection_metric': 'final-head positive-token F1 at fixed decision boundary 0.5; overlapping chunks included',
                 'training_documents': len(train_rows), 'training_positive_tokens': positives, 'training_negative_tokens': negatives, 'class_weights': class_weight.tolist(),
                 'validation_partition_documents': [len(selection_rows), len(temperature_rows), len(gate_rows)],
                 'validation_partition_hashes': [[row['sha256_normalized'] for row in rows] for rows in (selection_rows, temperature_rows, gate_rows)],
                 'epoch_work_used': sum(trial['epochs'] for trial in trials.values()), 'full_grid_epoch_work': len(trials) * args.budgets[-1],
                 'source_sha256': {name: digest(ROOT / 'ml' / name) for name in ('cascade.py', 'vq_search.py', 'search_cascade.py')},
                 'dataset_manifest_sha256': digest(ROOT / 'docs/reports' / ('balanced-dataset.json' if args.profile == 'balanced' else 'dataset.json')),
                 'limitations': ['Supervised end-to-end repair, not a matched-budget comparison to the earlier frozen reconstruction encoder.',
                                 'One seed; candidates differ in dimension, code count, and resolution. This is a configuration search, not a factorial ablation.',
                                 'No optimizer search or browser export. The test split is not loaded by this program.']}
    write_json(ROOT / f'docs/reports/{args.output}-selection.json', selection)


if __name__ == '__main__':
    main()
