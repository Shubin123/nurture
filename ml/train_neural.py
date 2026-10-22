import argparse
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer

from common import REPORTS, ROOT, RUNS, SEED, load_split, tune_threshold, write_json

MODEL = 'google/bert_uncased_L-2_H-128_A-2'
REVISION = '30b0a37ccaaa32f332884b96992754e246e48c5f'
MINI_MODEL = 'google/bert_uncased_L-4_H-256_A-4'
MINI_REVISION = '387825ce42dbb39b87911cdf8e383ee3b25184f8'
MAX_LENGTH = 128
STRIDE = 32


class Chunks(Dataset):
    def __init__(self, rows, tokenizer):
        self.encoding = tokenizer([row['text'] for row in rows], max_length=MAX_LENGTH, truncation=True, padding='max_length', stride=STRIDE, return_overflowing_tokens=True, return_offsets_mapping=True)
        self.items = []
        for index, source in enumerate(self.encoding['overflow_to_sample_mapping']):
            gold = set(rows[source]['spans'])
            labels = []
            for start, end in self.encoding['offset_mapping'][index]:
                labels.append(-100 if end <= start else int(sum(position in gold for position in range(start, end)) >= (end - start) / 2))
            self.items.append({key: torch.tensor(self.encoding[key][index]) for key in ('input_ids', 'attention_mask', 'token_type_ids')})
            self.items[-1]['labels'] = torch.tensor(labels)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def predict(model, chunks, rows, device, batch_size=64):
    buckets = [{} for _ in rows]
    cursor = 0
    model.eval()
    with torch.inference_mode():
        for batch in DataLoader(chunks, batch_size=batch_size, shuffle=False):
            batch.pop('labels')
            scores = torch.softmax(model(**{key: value.to(device) for key, value in batch.items()}).logits, -1)[:, :, 1].cpu().numpy()
            for values in scores:
                source = chunks.encoding['overflow_to_sample_mapping'][cursor]
                for offset, value in zip(chunks.encoding['offset_mapping'][cursor], values):
                    if offset[1] <= offset[0]:
                        continue
                    key = tuple(offset)
                    total, count = buckets[source].get(key, (0, 0))
                    buckets[source][key] = (total + float(value), count + 1)
                cursor += 1
    offsets, probabilities = [], []
    for bucket in buckets:
        ordered = sorted(bucket)
        offsets.append(ordered)
        probabilities.append([bucket[key][0] / bucket[key][1] for key in ordered])
    return offsets, probabilities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--size', choices=['tiny', 'mini'], default='tiny')
    args = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model_id, revision = (MODEL, REVISION) if args.size == 'tiny' else (MINI_MODEL, MINI_REVISION)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    train, validation = load_split('train'), load_split('validation')
    train_chunks, valid_chunks = Chunks(train, tokenizer), Chunks(validation, tokenizer)
    model = AutoModelForTokenClassification.from_pretrained(model_id, revision=revision, num_labels=2, id2label={0: 'O', 1: 'TOXIC'}, label2id={'O': 0, 'TOXIC': 1}, attn_implementation='eager').to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    loader = DataLoader(train_chunks, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    total_steps = len(loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / max(1, total_steps * 0.1), max(0.0, (total_steps - step) / max(1, total_steps * 0.9))))
    history = []
    best_score = -1
    started = time.perf_counter()
    destination = RUNS / f'bert-{args.size}'
    destination.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(destination)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for step, batch in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            loss = model(**{key: value.to(device) for key, value in batch.items()}).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
            if (step + 1) % 100 == 0:
                print(json.dumps({'epoch': epoch + 1, 'step': step + 1, 'steps': len(loader), 'loss': float(np.mean(losses[-100:]))}), flush=True)
        offsets, probabilities = predict(model, valid_chunks, validation, device)
        best, curve = tune_threshold(validation, offsets, probabilities)
        entry = {'epoch': epoch + 1, 'loss': float(np.mean(losses)), **best, 'elapsed_seconds': time.perf_counter() - started}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if best['macro_character_f1'] > best_score:
            best_score = best['macro_character_f1']
            model.save_pretrained(destination)
            write_json(destination / 'selection.json', {'epoch': epoch + 1, **best, 'threshold_curve': curve})
        write_json(REPORTS / f'bert-{args.size}-training.json', {'seed': SEED, 'base_model': model_id, 'revision': revision, 'train_examples': len(train), 'train_chunks': len(train_chunks), 'validation_chunks': len(valid_chunks), 'max_length': MAX_LENGTH, 'stride': STRIDE, 'device': str(device), 'batch_size': args.batch_size, 'learning_rate': 1e-4, 'epochs': args.epochs, 'history': history, 'parameters': sum(parameter.numel() for parameter in model.parameters())})


if __name__ == '__main__':
    main()
