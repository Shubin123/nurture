import ast
import csv
import hashlib
import io
import json
import urllib.request

from common import DATA, REPORTS, digest, write_json

REVISION = 'beaf170538df8275c3447b2e26070f2f82e55272'
BASE = f'https://raw.githubusercontent.com/ipavlopoulos/toxic_spans/{REVISION}'
CHECKSUMS = {
    'tsd_train.csv': 'b538ea2d64ed7c306ff8f010a25a5286bef6ce61d469042482ba64efc1ddd6cd',
    'tsd_trial.csv': '6bb8e9a16470c0269ac8727fd8bb77df68d8fb27955d9f5ff29ad724505622e8',
    'tsd_test.csv': '74f4dd086667fdad6198539ff9201d1f25f3a94c5169da1e7643a949974bf6b8',
}


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ['tsd_train.csv', 'tsd_trial.csv', 'tsd_test.csv']:
        path = DATA / name
        if not path.exists():
            path.write_bytes(urllib.request.urlopen(f'{BASE}/SemEval2021/data/{name}').read())
        if digest(path) != CHECKSUMS[name]:
            raise ValueError(f'Unexpected dataset checksum: {name}')
        files[name] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    license_path = DATA / 'LICENSE'
    if not license_path.exists():
        license_path.write_bytes(urllib.request.urlopen(f'{BASE}/LICENSE').read())
    seen = set()
    counts = {}
    for split, filename in [('test', 'tsd_test.csv'), ('validation', 'tsd_trial.csv'), ('train', 'tsd_train.csv')]:
        rows = []
        dropped = 0
        for index, row in enumerate(csv.DictReader(io.StringIO((DATA / filename).read_text()))):
            text = row['text']
            normalized = ' '.join(text.lower().split())
            key = hashlib.sha256(normalized.encode()).hexdigest()
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            spans = sorted(set(ast.literal_eval(row['spans'])))
            assert all(isinstance(position, int) and 0 <= position < len(text) for position in spans)
            rows.append({'id': f'{filename}:{index}', 'sha256_normalized': key, 'text': text, 'spans': spans})
        (DATA / f'{split}.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        counts[split] = {'examples': len(rows), 'dropped_duplicate': dropped, 'sha256': digest(DATA / f'{split}.jsonl')}
    manifest = {
        'dataset': 'SemEval-2021 Task 5 Toxic Spans', 'source': 'https://github.com/ipavlopoulos/toxic_spans',
        'revision': REVISION, 'license': 'CC0-1.0', 'license_sha256': digest(license_path), 'raw_files': files,
        'splits': counts, 'validation_source': 'official trial set',
        'deduplication': 'Lowercase and whitespace-normalized SHA-256; test then validation then train priority. No labels used for deduplication.',
        'limitations': ['No author identifiers; author/source-disjoint evaluation cannot be established.', 'Toxic comment domain, not representative of general browsing.', 'Rater spans are subjective; no guarantee of contextual safety or dialect fairness.'],
    }
    write_json(REPORTS / 'dataset.json', manifest)
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    prepare()
