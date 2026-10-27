import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ml'))
from vq_search import KERNELS, nearest_codes
from cascade import Cascade, CascadeConfig


class LookupTests(unittest.TestCase):
    def test_evaluator_measures_active_and_final_paths(self):
        from evaluate_cascade_search import measure

        class TinyChunks(torch.utils.data.Dataset):
            encoding = {'overflow_to_sample_mapping': [0, 1, 1]}

            def __len__(self):
                return 3

            def __getitem__(self, index):
                return {'input_ids': torch.arange(1, 17), 'attention_mask': torch.ones(16, dtype=torch.long), 'labels': torch.zeros(16, dtype=torch.long)}

        model = Cascade(CascadeConfig(vocabulary=32, length=16, width=16, slots=(1, 2, 4), codes=8, lookup='dot'))
        result, scores, full_scores = measure(model, TinyChunks(), 2, [1, 1, 1], [0, 0, 0], 1729)
        self.assertEqual(result['chunk_exit_fractions'], [1, 0, 0])
        self.assertEqual(result['document_max_exit_fractions'], [1, 0, 0])
        self.assertEqual(len(scores), 3)
        self.assertEqual(len(full_scores), 3)
        self.assertEqual(result['lookup_reference_max_probability_difference'], 0)

    def test_score_merge_counts_overlapping_offsets_once(self):
        from evaluate_cascade_search import merge_scores
        from types import SimpleNamespace
        chunks = SimpleNamespace(encoding={'overflow_to_sample_mapping': [0, 0], 'offset_mapping': [[(0, 1), (1, 2)], [(1, 2), (2, 3)]]})
        result = merge_scores(chunks, [{'spans': [1]}], [[0.1, 0.8], [0.6, 0.1]])
        self.assertEqual(result['true_positive_characters'], 1)
        self.assertEqual(result['micro_character_f1'], 1)

    def test_successive_halving_resumes_only_promoted_trials(self):
        from search_cascade import successive_halving
        calls = []

        def evaluate(identifier, budget):
            calls.append((identifier, budget))
            return {'objective': {'best': 0.8, 'middle': 0.4, 'last': 0.1}[identifier]}

        winner, history = successive_halving(['last', 'middle', 'best'], [1, 3, 9], evaluate)
        self.assertEqual(winner, 'best')
        self.assertEqual(len(history), 6)
        self.assertNotIn(('last', 3), calls)
        self.assertEqual(calls[-1], ('best', 9))

    def test_validation_partitions_are_disjoint_and_reproducible(self):
        from search_cascade import partition_validation
        rows = [{'id': index} for index in range(29)]
        groups = partition_validation(rows, 1729)
        identifiers = [{row['id'] for row in group} for group in groups]
        self.assertEqual(set.union(*identifiers), set(range(29)))
        self.assertFalse(identifiers[0] & identifiers[1] or identifiers[0] & identifiers[2] or identifiers[1] & identifiers[2])
        self.assertEqual(groups, partition_validation(rows, 1729))

    def test_invalid_search_budget(self):
        from search_cascade import successive_halving
        with self.assertRaises(ValueError):
            successive_halving(['trial'], [3, 2], lambda identifier, budget: {'objective': 0})

    def test_kernels_match_direct_euclidean_search(self):
        generator = torch.Generator().manual_seed(1729)
        values = torch.randn(3, 9, 16, generator=generator)
        codes = torch.randn(37, 16, generator=generator)
        expected = (values.double().unsqueeze(-2) - codes.double()).square().sum(-1).argmin(-1)
        for kernel in KERNELS:
            torch.testing.assert_close(nearest_codes(values, codes, kernel=kernel, tile_size=7), expected)

    def test_ties_choose_first_code_across_tiles(self):
        values = torch.tensor([[1.0, 2.0], [0.0, 0.0]])
        codes = values[[0, 1, 0, 1]]
        for kernel in KERNELS:
            self.assertEqual(nearest_codes(values, codes, kernel=kernel, tile_size=1).tolist(), [0, 1])

    def test_cosine_matches_direct_dot_product(self):
        torch.manual_seed(17)
        values, codes = torch.randn(12, 7), torch.randn(31, 7)
        expected = (torch.nn.functional.normalize(values, dim=-1) @ torch.nn.functional.normalize(codes, dim=-1).T).argmax(-1)
        for kernel in KERNELS:
            torch.testing.assert_close(nearest_codes(values, codes, metric='cosine', kernel=kernel, tile_size=3), expected)

    def test_zero_cosine_queries_choose_first_code(self):
        torch.manual_seed(17)
        values, codes = torch.zeros(3, 7), torch.randn(31, 7)
        for kernel in KERNELS:
            self.assertEqual(nearest_codes(values, codes, metric='cosine', kernel=kernel, tile_size=3).tolist(), [0, 0, 0])

    def test_lookup_does_not_retain_autograd_graph(self):
        indices = nearest_codes(torch.randn(4, 8, requires_grad=True), torch.randn(9, 8, requires_grad=True))
        self.assertFalse(indices.requires_grad)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            nearest_codes(torch.zeros(2, 4), torch.zeros(0, 4))
        with self.assertRaises(ValueError):
            nearest_codes(torch.zeros(2, 4), torch.zeros(3, 4), tile_size=0)

    def test_low_dimensional_cosine_gradients_and_initialization(self):
        torch.manual_seed(17)
        model = Cascade(CascadeConfig(vocabulary=32, length=16, width=16, slots=(2, 4, 16), codes=8, code_width=4, metric='cosine', lookup='dot'))
        tokens = torch.randint(1, 32, (3, 16))
        mask = torch.ones_like(tokens)
        model.initialize_codebooks(tokens, mask)
        scores, _, loss, codes = model(tokens, mask, reconstruct=False)
        (scores[-1].square().mean() + loss).backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.code_in[-1].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.code_out[-1].weight.grad.abs().sum()), 0)
        self.assertEqual(codes[-1].shape, (48,))


if __name__ == '__main__':
    unittest.main()
