import json

from transformers import AutoTokenizer

from common import RUNS, features, fnv, load_split, token_spans, write_json


def main():
    tokenizer = AutoTokenizer.from_pretrained(RUNS / 'bert-tiny', use_fast=True)
    texts = [row['text'] for row in load_split('validation')]
    texts.extend(['An idiot, not an idiotic café. naïve naïveté.', '中文 English mixed 👩🏽‍💻 text.', 'That is STUPID. The word "stupid" is quoted.', 'e\u0301 and İ and résumé', 'x\x00y\u200bz', '[CLS] text [MASK] [SEP] [UNK] [PAD]'])
    rows = []
    for text in texts:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        tokens = token_spans(text)
        rows.append({'text': text, 'ids': encoded['input_ids'], 'offsets': encoded['offset_mapping'], 'linear_tokens': tokens, 'features': [features(tokens, index) for index in range(len(tokens))]})
    write_json(RUNS / 'tokenizer-fixtures.json', rows)
    print(f'Prepared {len(rows)} tokenizer conformance fixtures.')


if __name__ == '__main__':
    main()
