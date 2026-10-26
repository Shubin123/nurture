import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ml'))
from cascade import Cascade, CascadeConfig, calibrate, optimizers, select_gates


class CascadeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1729)
        torch.set_num_threads(1)
        self.model = Cascade(CascadeConfig(vocabulary=32, length=16, width=16, slots=(1, 2, 4), codes=8)).eval()
        self.tokens = torch.randint(1, 32, (4, 16))
        self.mask = torch.ones_like(self.tokens)
        self.mask[-1, 8:] = 0

    def test_shapes_losses_and_codebook_gradients(self):
        logits, reconstruction, loss, codes = self.model(self.tokens, self.mask)
        self.assertEqual(reconstruction.shape, (4, 16, 32))
        self.assertEqual(len(logits), 3)
        self.assertEqual([len(values) for values in codes], [4, 7, 14])
        (loss + reconstruction.square().mean()).backward()
        self.assertTrue(torch.isfinite(loss))
        for codebook in self.model.codebooks:
            self.assertGreater(float(codebook.weight.grad.abs().sum()), 0)

    def test_forced_final_matches_forward(self):
        with torch.inference_mode():
            scores, _, _, _ = self.model(self.tokens, self.mask)
            probabilities, exits = self.model.predict(self.tokens, self.mask, [1, 1, 1], [1, 1, 0])
        self.assertEqual(exits.tolist(), [2] * 4)
        torch.testing.assert_close(probabilities, scores[-1].softmax(-1))

    def test_early_exit_skips_later_quantization(self):
        calls = []
        handles = [head.register_forward_hook(lambda module, inputs, output: calls.append(module)) for head in self.model.heads]
        _, exits = self.model.predict(self.tokens, self.mask, [1, 1, 1], [0, 0, 0])
        self.assertEqual(exits.tolist(), [0] * 4)
        self.assertEqual(calls, [self.model.heads[0]])
        for handle in handles:
            handle.remove()

    def test_fine_codebook_cannot_change_coarse_predictions(self):
        with torch.inference_mode():
            before = self.model(self.tokens, self.mask, reconstruct=False)[0][0]
            self.model.codebooks[-1].weight.add_(100)
            after = self.model(self.tokens, self.mask, reconstruct=False)[0][0]
        torch.testing.assert_close(before, after)

    def test_mixed_exits_shrink_active_batch(self):
        class Head(torch.nn.Module):
            def __init__(self, first):
                super().__init__()
                self.first = first
                self.batch_sizes = []

            def forward(self, values):
                self.batch_sizes.append(len(values))
                scores = values.new_zeros((*values.shape[:2], 2))
                if self.first:
                    scores[0, :, 0] = 20
                else:
                    scores[:, :, 1] = 20
                return scores

        heads = [Head(True), Head(False), Head(False)]
        self.model.heads = torch.nn.ModuleList(heads)
        probabilities, exits = self.model.predict(self.tokens, self.mask, [1, 1, 1], [0.9, 0.9, 0])
        self.assertEqual(exits.tolist(), [0, 1, 1, 1])
        self.assertEqual([head.batch_sizes for head in heads], [[4], [3], []])
        self.assertEqual(probabilities.argmax(-1)[:, 0].tolist(), [0, 1, 1, 1])

    def test_disabled_gate_does_not_exit_saturated_probabilities(self):
        for head in self.model.heads:
            with torch.no_grad():
                head[-1].weight.zero_()
                head[-1].bias.copy_(torch.tensor([1000.0, -1000.0]))
        _, exits = self.model.predict(self.tokens, self.mask, [1, 1, 1], [1, 1, 0])
        self.assertEqual(exits.tolist(), [2] * 4)

    def test_nested_dropout_excludes_fine_codebooks(self):
        scores, reconstruction, loss, codes = self.model(self.tokens, self.mask, max_levels=1)
        self.assertEqual(len(scores), 1)
        self.assertEqual(len(codes), 1)
        (loss + reconstruction.square().mean()).backward()
        self.assertIsNone(self.model.codebooks[-1].weight.grad)

    def test_optimizer_partition_is_complete_and_disjoint(self):
        optimizer_list = optimizers(self.model, 'muon')
        groups = [[parameter for group in optimizer.param_groups for parameter in group['params']] for optimizer in optimizer_list]
        self.assertEqual({id(parameter) for group in groups for parameter in group}, {id(parameter) for parameter in self.model.parameters()})
        self.assertFalse({id(parameter) for parameter in groups[0]} & {id(parameter) for parameter in groups[1]})
        allowed = {id(parameter) for name, parameter in self.model.named_parameters() if name.startswith('encoder.') and parameter.ndim == 2}
        self.assertEqual({id(parameter) for parameter in groups[0]}, allowed)
        _, reconstruction, loss, _ = self.model(self.tokens, self.mask)
        (loss + reconstruction.square().mean()).backward()
        for optimizer in optimizer_list:
            optimizer.step()

    def test_calibration_refuses_confident_wrong_exits(self):
        labels = torch.zeros((40, 2), dtype=torch.long)
        mask = torch.ones_like(labels, dtype=torch.bool)
        wrong = torch.tensor([0.0, 10.0]).expand(40, 2, 2)
        right = wrong.flip(-1)
        temperatures = calibrate([wrong, right, right], labels, mask)
        self.assertEqual(len(temperatures), 3)
        thresholds = select_gates([wrong, right, right], labels, mask, [1, 1, 1])
        self.assertEqual(thresholds[0], 1)
        self.assertLess(thresholds[1], 1)

    def test_invalid_schedule_and_calibration(self):
        with self.assertRaises(ValueError):
            CascadeConfig(slots=(2, 2, 4))
        with self.assertRaises(ValueError):
            self.model.predict(self.tokens, self.mask, [0, 1, 1], [1, 1, 0])

    def test_committed_tokenizer_does_not_fall_back_to_special_tokens(self):
        from train_cascade import local_tokenizer
        tokenizer = local_tokenizer()
        self.assertEqual(len(tokenizer), 30522)
        self.assertNotIn(tokenizer.unk_token_id, tokenizer('hello world')['input_ids'])


if __name__ == '__main__':
    unittest.main()
