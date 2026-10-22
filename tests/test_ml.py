import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ml'))
from common import character_metrics, features, fnv, load_split, predict_characters, token_spans


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


if __name__ == '__main__':
    unittest.main()
