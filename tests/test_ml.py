import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ml'))
from common import character_metrics, features, fnv, load_split, predict_characters, token_spans, tune_threshold


class MetricsTests(unittest.TestCase):
    def test_character_f1_and_empty_gold_semantics(self):
        rows = [{'spans': [1, 2]}, {'spans': []}]
        result = character_metrics(rows, [[2, 3], []])
        self.assertEqual(result['macro_character_f1'], 0.75)
        self.assertEqual(result['micro_character_precision'], 0.5)
        self.assertEqual(result['empty_gold_examples'], 1)
        self.assertNotIn('benign_false_positive_rate', result)

    def test_metrics_reject_missing_predictions(self):
        with self.assertRaises(ValueError):
            character_metrics([{'spans': []}], [])

    def test_offsets_are_exact_and_threshold_is_inclusive(self):
        self.assertEqual(predict_characters([(0, 0), (3, 6), (7, 9)], [1, 0.5, 0.49], 0.5), [3, 4, 5])

    def test_hash_and_feature_stability(self):
        self.assertEqual(fnv('hello'), 11435)
        tokens = token_spans("An idiot's café!")
        self.assertEqual([item[0] for item in tokens], ['An', "idiot's", 'caf', 'é', '!'])
        self.assertEqual(features(tokens, 1), sorted(set(features(tokens, 1))))

    def test_splits_are_disjoint(self):
        splits = [load_split(name) for name in ('train', 'validation', 'test')]
        identifiers = [{row['sha256_normalized'] for row in rows} for rows in splits]
        for index, left in enumerate(identifiers):
            for right in identifiers[index + 1:]:
                self.assertFalse(left & right)

    def test_balanced_selection_uses_equal_subset_weight_not_raw_counts(self):
        rows = [{'spans': [0]}] + [{'spans': [], 'source': 'Civil Comments zero-rated negative'} for _ in range(20)]
        offsets = [[(0, 1)]] * len(rows)
        probabilities = [[0.9]] + [[0.1]] * 20
        best, _ = tune_threshold(rows, offsets, probabilities, 'balanced_context')
        self.assertEqual(best['selection_score'], 1)
        self.assertEqual(best['span_subset_macro_f1'], 1)
        self.assertEqual(best['rated_benign_acceptance'], 1)

    @unittest.skipUnless((Path(__file__).resolve().parents[1] / 'ml/data/balanced/train.jsonl').exists(), 'Supplemental balanced dataset not prepared')
    def test_balanced_data_remains_disjoint_and_preserves_original_splits(self):
        original = {name: load_split(name) for name in ('train', 'validation', 'test')}
        balanced = {name: load_split(name, 'balanced') for name in original}
        original_keys = {row['sha256_normalized'] for rows in original.values() for row in rows}
        keys = {name: {row['sha256_normalized'] for row in rows} for name, rows in balanced.items()}
        self.assertFalse(keys['train'] & keys['validation'])
        self.assertFalse(keys['train'] & keys['test'])
        self.assertFalse(keys['validation'] & keys['test'])
        for name, rows in balanced.items():
            self.assertTrue({row['id'] for row in original[name]}.issubset({row['id'] for row in rows}))
            for row in rows:
                if row.get('source') == 'Civil Comments zero-rated negative':
                    self.assertNotIn(row['sha256_normalized'], original_keys)
                    self.assertEqual(row['spans'], [])


if __name__ == '__main__':
    unittest.main()
