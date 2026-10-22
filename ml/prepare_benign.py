import hashlib
import heapq
import json
import urllib.request

import pyarrow.parquet as parquet

from common import DATA, REPORTS, digest, load_split, write_json

REVISION = 'f2970eb3a55777454c94069077cc8d9b5866312d'
FILES = {
    'train': ('train-00000-of-00002.parquet', 'c20f01c3aecbdd942886cacb0ee67e995df33bc712a622fde75927ed3d6ccefe', 10000),
    'validation': ('validation-00000-of-00001.parquet', '2e0eb65474e7e1290df8689fc93eea783160c898c09ae15382c0a9662367bd03', 2000),
    'test': ('test-00000-of-00001.parquet', '410f803ff161e7c4ba005db7de5b444cac413f4861bf056b0fd69d670d75aa08', 5000),
}
LABELS = ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit']


def main():
    destination = DATA / 'balanced'
    raw = DATA / 'civil-comments'
    destination.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    original = {split: load_split(split) for split in FILES}
    seen = {row['sha256_normalized'] for rows in original.values() for row in rows}
    manifest = {
        'status': 'Prepared for next training experiment; not used by the currently deployed models.',
        'original_dataset': json.loads((REPORTS / 'dataset.json').read_text()),
        'additional_source': 'google/civil_comments', 'revision': REVISION, 'license': 'CC0-1.0',
        'sampling': 'Lowest normalized text SHA-256 among 20–4000-character posts with all seven human-rated toxicity sub-scores exactly zero. Only the first official train shard is sampled. Existing Toxic Spans rows and selected cross-split duplicates are excluded.',
        'limitations': ['Whole-post negative labels are inferred from human-rated zero scores, not independently annotated character spans.', 'Human ratings and sampling can be biased; this is not universal safety ground truth.', 'Author-disjointness is not established.'],
        'splits': {}, 'files': {},
    }
    for split in ('test', 'validation', 'train'):
        filename, expected_hash, limit = FILES[split]
        path = raw / filename
        if not path.exists():
            temporary = path.with_suffix('.download')
            print(f'Downloading pinned Civil Comments {split} shard...', flush=True)
            with urllib.request.urlopen(f'https://huggingface.co/datasets/google/civil_comments/resolve/{REVISION}/data/{filename}') as response, temporary.open('wb') as target:
                while block := response.read(1024 * 1024):
                    target.write(block)
            temporary.replace(path)
        if digest(path) != expected_hash:
            raise ValueError(f'Unexpected checksum for {filename}')
        heap = []
        selected = set()
        scanned = eligible = 0
        for batch in parquet.ParquetFile(path).iter_batches(batch_size=8192, columns=['text'] + LABELS):
            for record in batch.to_pylist():
                index = scanned
                scanned += 1
                text = record['text']
                if not isinstance(text, str) or not 20 <= len(text) <= 4000 or any(record[label] != 0.0 for label in LABELS):
                    continue
                normalized = ' '.join(text.lower().split())
                key = hashlib.sha256(normalized.encode()).hexdigest()
                if key in seen or key in selected:
                    continue
                eligible += 1
                row = {'id': f'civil_comments:{split}:{index}', 'sha256_normalized': key, 'text': text, 'spans': [], 'source': 'Civil Comments zero-rated negative'}
                entry = (-int(key, 16), key, row)
                if len(heap) < limit:
                    heapq.heappush(heap, entry)
                    selected.add(key)
                elif entry[0] > heap[0][0]:
                    removed = heapq.heapreplace(heap, entry)
                    selected.remove(removed[1])
                    selected.add(key)
        negatives = [entry[2] for entry in sorted(heap, key=lambda item: item[1])]
        assert len(negatives) == limit, (split, len(negatives))
        seen.update(row['sha256_normalized'] for row in negatives)
        for name, rows in [(split, original[split] + negatives), (split + '-benign', negatives)]:
            output = destination / f'{name}.jsonl'
            output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
            manifest['splits'][name] = {'examples': len(rows), 'sha256': digest(output)}
        manifest['files'][split] = {'filename': filename, 'sha256': expected_hash, 'bytes': path.stat().st_size, 'rows_scanned': scanned, 'eligible_before_reservoir': eligible, 'selected': len(negatives)}
        write_json(REPORTS / 'balanced-dataset.json', manifest)
        print(split, json.dumps(manifest['files'][split]), flush=True)
    print('Prepared disjoint supplemental negative examples. Training/evaluation is the next step; no accuracy claim is made.', flush=True)


if __name__ == '__main__':
    main()
