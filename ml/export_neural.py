import argparse
import json
import shutil

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForTokenClassification

from common import REPORTS, ROOT, RUNS, digest, write_json


class Export(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', choices=['tiny', 'mini'], default='tiny')
    parser.add_argument('--profile', choices=['original', 'balanced'], default='original')
    args = parser.parse_args()
    suffix = '-balanced' if args.profile == 'balanced' else ''
    torch.set_num_threads(4)
    source = RUNS / f'bert-{args.size}{suffix}'
    destination = (RUNS / 'candidate-models' if args.profile == 'balanced' else ROOT / 'docs' / 'models') / f'bert-{args.size}{suffix}'
    destination.mkdir(parents=True, exist_ok=True)
    model = AutoModelForTokenClassification.from_pretrained(source, attn_implementation='eager').eval()
    wrapper = Export(model).eval()
    ids = torch.tensor([[101] + [2023] * 126 + [102]], dtype=torch.int64)
    mask = torch.ones_like(ids)
    float_path = RUNS / f'bert-{args.size}{suffix}-fp32.onnx'
    torch.onnx.export(wrapper, (ids, mask), str(float_path), input_names=['input_ids', 'attention_mask'], output_names=['logits'], dynamic_axes={'input_ids': {0: 'batch', 1: 'sequence'}, 'attention_mask': {0: 'batch', 1: 'sequence'}, 'logits': {0: 'batch', 1: 'sequence'}}, opset_version=17, dynamo=False)
    onnx.checker.check_model(str(float_path))
    quantized_path = destination / 'model.int8.onnx'
    quantize_dynamic(str(float_path), str(quantized_path), weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=['MatMul', 'Gather'])
    onnx.checker.check_model(str(quantized_path))
    vocabulary = json.loads((source / 'tokenizer.json').read_text())['model']['vocab']
    (destination.parent / 'vocab.txt').write_text('\n'.join(sorted(vocabulary, key=vocabulary.get)) + '\n')
    for filename in ('config.json',):
        if (source / filename).exists():
            shutil.copy(source / filename, destination / filename)
    selection = json.loads((source / 'selection.json').read_text())
    write_json(destination / 'inference.json', {'max_length': 128, 'stride': 32, 'batch_size': 8, 'threshold': selection['threshold'], 'positive_class': 1, 'quantization': 'dynamic int8, per-channel MatMul and Gather weights', 'experimental': True})
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    float_session = ort.InferenceSession(str(float_path), options, providers=['CPUExecutionProvider'])
    quantized_session = ort.InferenceSession(str(quantized_path), options, providers=['CPUExecutionProvider'])
    feed = {'input_ids': ids.numpy(), 'attention_mask': mask.numpy()}
    with torch.inference_mode():
        reference = wrapper(ids, mask).numpy()
    floating = float_session.run(None, feed)[0]
    quantized = quantized_session.run(None, feed)[0]
    assert np.max(np.abs(reference - floating)) < 1e-4, 'ONNX export does not match PyTorch.'
    write_json(REPORTS / f'bert-{args.size}{suffix}-export.json', {'fp32_bytes': float_path.stat().st_size, 'int8_bytes': quantized_path.stat().st_size, 'int8_sha256': digest(quantized_path), 'torch_onnx_max_abs_logit_difference': float(np.max(np.abs(reference - floating))), 'quantized_max_abs_logit_difference_smoke': float(np.max(np.abs(reference - quantized))), 'quantized_accuracy_requires_full_evaluation': True})
    print((REPORTS / f'bert-{args.size}{suffix}-export.json').read_text())


if __name__ == '__main__':
    main()
