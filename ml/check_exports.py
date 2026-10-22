import json

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForTokenClassification

from common import REPORTS, RUNS, write_json

torch.set_num_threads(2)
random = np.random.default_rng(1729)
report = {}
for size in ('tiny', 'mini'):
    model = AutoModelForTokenClassification.from_pretrained(RUNS / f'bert-{size}', attn_implementation='eager').eval()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(RUNS / f'bert-{size}-fp32.onnx'), options, providers=['CPUExecutionProvider'])
    cases = []
    for batch, sequence in ((1, 17), (3, 128), (8, 128)):
        ids = random.integers(999, 20000, (batch, sequence), dtype=np.int64)
        masks = np.ones_like(ids)
        for index in range(batch):
            valid = max(3, sequence - index * 5 - 3)
            ids[index, valid:] = 0
            masks[index, valid:] = 0
        with torch.inference_mode():
            reference = model(input_ids=torch.tensor(ids), attention_mask=torch.tensor(masks)).logits.numpy()
        output = session.run(None, {'input_ids': ids, 'attention_mask': masks})[0]
        error = float(np.max(np.abs(reference - output)))
        assert error < 1e-4, (size, batch, sequence, error)
        cases.append({'batch': batch, 'sequence': sequence, 'mixed_padding': True, 'max_abs_logit_difference': error})
    report[size] = cases
write_json(REPORTS / 'export-conformance.json', report)
print(json.dumps(report, indent=2))
