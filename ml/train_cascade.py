"""Opt-in native experiment. No downloads: uses the committed WordPiece vocabulary."""

import argparse
from dataclasses import asdict
import platform
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from cascade import Cascade, CascadeConfig, calibrate, optimizers, select_gates
from common import ROOT, RUNS, SEED, character_metrics, digest, load_split, predict_characters, write_json
from train_neural import Chunks


def local_tokenizer():
    vocabulary = (ROOT / 'docs/models/vocab.txt').read_text().splitlines()
    tokenizer = BertTokenizerFast(vocab={token: index for index, token in enumerate(vocabulary)}, do_lower_case=True)
    if len(tokenizer) != len(vocabulary) or len(vocabulary) < 10000:
        raise ValueError('The complete committed WordPiece vocabulary is required.')
    return tokenizer


def synchronize(device):
    if device.type == 'mps':
        torch.mps.synchronize()


def metrics(labels, predictions):
    result = {'accuracy': float((labels == predictions).float().mean()), 'per_class': {}}
    for label, name in enumerate(('O', 'TOXIC')):
        positive = int(((labels == label) & (predictions == label)).sum())
        predicted = int((predictions == label).sum())
        support = int((labels == label).sum())
        precision, recall = positive / max(1, predicted), positive / max(1, support)
        result['per_class'][name] = {'precision': precision, 'recall': recall, 'f1': 2 * precision * recall / max(1e-12, precision + recall), 'support': support}
    result['macro_f1'] = sum(value['f1'] for value in result['per_class'].values()) / 2
    result.update({key: result['per_class']['TOXIC'][key] for key in ('precision', 'recall', 'f1')})
    return result


@torch.inference_mode()
def collect(model, chunks, device, batch_size):
    model.eval()
    outputs = [[] for _ in model.heads]
    labels = []
    for batch in DataLoader(chunks, batch_size=batch_size):
        scores, _, _, _ = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), reconstruct=False)
        for bucket, values in zip(outputs, scores):
            bucket.append(values.cpu())
        labels.append(batch['labels'])
    return [torch.cat(bucket) for bucket in outputs], torch.cat(labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experimental-cascade', action='store_true', required=True)
    parser.add_argument('--optimizer', choices=('adamw', 'muon'), default='adamw')
    parser.add_argument('--profile', choices=('original', 'balanced'), default='balanced')
    parser.add_argument('--pretrain-epochs', type=int, default=2)
    parser.add_argument('--head-epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0, help='Document cap per split; diagnostic only.')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--fractions', type=float, nargs='+', default=[1.0, 0.25, 0.1])
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--levels', type=int, choices=(3, 4), default=3)
    parser.add_argument('--target-accuracy', type=float, default=0.95)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--muon-rate', type=float, default=0.02)
    parser.add_argument('--output', default='cascade-experiment')
    args = parser.parse_args()
    if args.batch_size < 1 or args.pretrain_epochs < 1 or args.head_epochs < 1 or args.limit < 0 or any(not 0 < value <= 1 for value in args.fractions):
        parser.error('Positive epochs/batch size and label fractions in (0, 1] required.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    tokenizer = local_tokenizer()
    rows = {name: load_split(name, args.profile) for name in ('train', 'validation', 'test')}
    for name, values in rows.items():
        random.Random(args.seed).shuffle(values)
        if args.limit:
            rows[name] = values[:args.limit]
    validation = rows['validation']
    if len(validation) < 4:
        parser.error('At least four validation documents are required.')
    midpoint = len(validation) // 2
    train = Chunks(rows['train'], tokenizer)
    temperature_chunks = Chunks(validation[:midpoint], tokenizer)
    gate_chunks = Chunks(validation[midpoint:], tokenizer)
    test = Chunks(rows['test'], tokenizer)
    config = CascadeConfig(vocabulary=len(tokenizer), slots=(2, 4, 8) if args.levels == 3 else (2, 4, 8, 16))
    model = Cascade(config).to(device)
    optimizer_list = optimizers(model, args.optimizer, args.learning_rate, args.muon_rate)
    history = []
    started = time.perf_counter()
    for epoch in range(args.pretrain_epochs):
        model.train()
        usage = [torch.zeros(config.codes, dtype=torch.long) for _ in config.slots]
        losses = []
        reconstructed = reconstruction_tokens = 0
        loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed + epoch))
        for batch in loader:
            for optimizer in optimizer_list:
                optimizer.zero_grad(set_to_none=True)
            tokens, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            cutoff = args.levels if random.random() >= 0.5 else random.randint(1, args.levels - 1)
            _, reconstruction, commitment, codes = model(tokens, mask, max_levels=cutoff)
            loss = F.cross_entropy(reconstruction[mask.bool()], tokens[mask.bool()]) + commitment
            reconstructed += int((reconstruction.detach().argmax(-1)[mask.bool()] == tokens[mask.bool()]).sum())
            reconstruction_tokens += int(mask.sum())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for optimizer in optimizer_list:
                optimizer.step()
            losses.append(float(loss.detach()))
            for counts, indices in zip(usage, codes):
                counts += torch.bincount(indices.detach().cpu(), minlength=config.codes)
        entry = {'stage': 'pretrain', 'epoch': epoch + 1, 'loss': float(np.mean(losses)), 'elapsed_seconds': time.perf_counter() - started,
                 'training_reconstruction_token_accuracy': reconstructed / reconstruction_tokens,
                 'codebooks': [{'used': int((counts > 0).sum()), 'histogram': counts.tolist(), 'perplexity': float((-(counts.float() / counts.sum()).clamp_min(1e-12).log().mul(counts.float() / counts.sum()).sum()).exp())} for counts in usage]}
        history.append(entry)
        print({key: value for key, value in entry.items() if key != 'codebooks'}, flush=True)
    synchronize(device)
    pretrain_seconds = time.perf_counter() - started
    encoder_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    results = []
    destination = RUNS / args.output
    destination.mkdir(parents=True, exist_ok=True)
    for fraction in args.fractions:
        torch.manual_seed(args.seed)
        model = Cascade(config).to(device)
        model.load_state_dict(encoder_state)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith('heads.'))
        optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3, weight_decay=0.01)
        count = max(1, int(len(rows['train']) * fraction))
        labelled = Chunks(rows['train'][:count], tokenizer)
        head_started = time.perf_counter()
        target_seconds = None
        for epoch in range(args.head_epochs):
            model.train()
            for batch in DataLoader(labelled, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed + epoch)):
                optimizer.zero_grad(set_to_none=True)
                scores, _, _, _ = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), reconstruct=False)
                labels = batch['labels'].to(device)
                loss = sum(F.cross_entropy(values.flatten(0, 1), labels.flatten(), ignore_index=-100) for values in scores) / len(scores)
                loss.backward()
                optimizer.step()
            scores, labels = collect(model, temperature_chunks, device, args.batch_size)
            valid = labels.ne(-100)
            accuracy = metrics(labels[valid], scores[-1].argmax(-1)[valid])['accuracy']
            elapsed = time.perf_counter() - head_started
            if target_seconds is None and accuracy >= args.target_accuracy:
                target_seconds = pretrain_seconds + elapsed
            history.append({'stage': 'heads', 'fraction': fraction, 'epoch': epoch + 1, 'validation_accuracy': accuracy, 'elapsed_seconds': elapsed})
        temperatures = calibrate(scores, labels, valid)
        gate_scores, gate_labels = collect(model, gate_chunks, device, args.batch_size)
        thresholds = select_gates(gate_scores, gate_labels, gate_labels.ne(-100), temperatures)
        model.eval()
        buckets = [{} for _ in rows['test']]
        all_labels, all_predictions, full_predictions, all_exits, durations, full_durations = [], [], [], [], [], []
        cursor = 0
        with torch.inference_mode():
            first = next(iter(DataLoader(test, batch_size=args.batch_size)))
            for _ in range(3):
                model.predict(first['input_ids'].to(device), first['attention_mask'].to(device), temperatures, thresholds)
            for batch in DataLoader(test, batch_size=args.batch_size):
                tokens, attention = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                synchronize(device)
                tick = time.perf_counter()
                probabilities, exits = model.predict(tokens, attention, temperatures, thresholds)
                synchronize(device)
                durations.append(time.perf_counter() - tick)
                synchronize(device)
                tick = time.perf_counter()
                full, _ = model.predict(tokens, attention, temperatures, [1.0] * (args.levels - 1) + [0.0])
                synchronize(device)
                full_durations.append(time.perf_counter() - tick)
                probabilities, exits = probabilities.cpu(), exits.cpu()
                mask = batch['labels'].ne(-100)
                all_labels.append(batch['labels'][mask])
                all_predictions.append(probabilities.argmax(-1)[mask])
                full_predictions.append(full.cpu().argmax(-1)[mask])
                all_exits.extend(exits.tolist())
                for values in probabilities[:, :, 1]:
                    source = test.encoding['overflow_to_sample_mapping'][cursor]
                    for offset, value in zip(test.encoding['offset_mapping'][cursor], values):
                        if offset[1] <= offset[0]:
                            continue
                        key = tuple(offset)
                        total, repeats = buckets[source].get(key, (0.0, 0))
                        buckets[source][key] = (total + float(value), repeats + 1)
                    cursor += 1
        predictions = [predict_characters(list(bucket), [total / repeats for total, repeats in bucket.values()], 0.5) for bucket in buckets]
        result = {'label_fraction': fraction, 'labelled_documents': count, 'temperatures': temperatures, 'thresholds': thresholds,
                  'token_metrics': metrics(torch.cat(all_labels), torch.cat(all_predictions)),
                  'forced_final_token_metrics': metrics(torch.cat(all_labels), torch.cat(full_predictions)),
                  'character_metrics': character_metrics(rows['test'], predictions),
                  'exit_fractions': [all_exits.count(level) / len(all_exits) for level in range(args.levels)],
                  'inference_seconds': sum(durations), 'processed_labelled_tokens_per_second': sum(len(values) for values in all_labels) / sum(durations),
                  'forced_final_inference_seconds': sum(full_durations), 'forced_final_over_cascade_speed_ratio': sum(full_durations) / sum(durations),
                  'time_to_target_accuracy_seconds': target_seconds, 'head_training_and_calibration_seconds': time.perf_counter() - head_started}
        results.append(result)
        torch.save({'config': asdict(config), 'state_dict': model.cpu().state_dict(), 'temperatures': temperatures, 'thresholds': thresholds}, destination / f'{fraction:g}.pt')
        model.to(device)
        print(result, flush=True)
    report = {'experimental': True, 'args': vars(args), 'config': asdict(config), 'torch': torch.__version__, 'platform': platform.platform(),
              'vocabulary_sha256': digest(ROOT / 'docs/models/vocab.txt'), 'parameters': sum(parameter.numel() for parameter in model.parameters()),
              'inference_parameters': sum(parameter.numel() for name, parameter in model.named_parameters() if not name.startswith('decoder.')),
              'source_sha256': {name: digest(ROOT / 'ml' / name) for name in ('cascade.py', 'train_cascade.py')},
              'dataset_manifest_sha256': digest(ROOT / 'docs/reports' / ('balanced-dataset.json' if args.profile == 'balanced' else 'dataset.json')),
              'pretrain_seconds': pretrain_seconds, 'history': history, 'results': results,
              'limitations': ['Native model-only timing, excludes tokenization, transfers and aggregation; not comparable to browser end-to-end throughput.',
                              'Token metrics count overlapping chunks. Character metrics merge offsets once per document.',
                              'Temperature and gate partitions are document-disjoint halves of validation; test is never calibrated.',
                              'Exit error constraint is empirical chunk exact-label error, not a safety or precision guarantee.',
                              'Encoder runs in full before exits; only later quantizers and heads are skipped.',
                              'Label fractions share one frozen encoder trained without span labels; head optimizer is always AdamW.',
                              'Target is final-head validation token accuracy, majority-class sensitive; null means not reached.']}
    write_json(ROOT / 'docs/reports' / f'{args.output}.json', report)


if __name__ == '__main__':
    main()
