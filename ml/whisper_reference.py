"""Independent reference for the browser Whisper runtime (ml/whisper-runtime.js, ml/asr-worker.js).

Features: Hugging Face WhisperFeatureExtractor. Tokens: greedy decoding of the same quantized ONNX files
with Python ONNX Runtime. Word timings: measured pauses in the audio itself (RMS below 0.01 for at least
80 ms), which every word boundary must respect.
"""
import json
import sys
import wave

import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor

MODEL = "docs/models/whisper-tiny-en"
path = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/speech.wav"
with wave.open(path) as source:
    assert source.getframerate() == 16000 and source.getnchannels() == 1
    audio = np.frombuffer(source.readframes(source.getnframes()), dtype=np.int16).astype(np.float32) / 32768

extractor = WhisperFeatureExtractor(feature_size=80, sampling_rate=16000, hop_length=160, chunk_length=30, n_fft=400)
features = extractor(audio, sampling_rate=16000, return_tensors="np").input_features.astype(np.float32)

generation = json.load(open(f"{MODEL}/generation_config.json"))
eot = generation["eos_token_id"]
suppressed = set(generation["suppress_tokens"])
encoder = ort.InferenceSession(f"{MODEL}/onnx/encoder_model_quantized.onnx")
decoder = ort.InferenceSession(f"{MODEL}/onnx/decoder_model_merged_quantized.onnx")
hidden = encoder.run(None, {"input_features": features})[0]
names = [output.name for output in decoder.get_outputs()]
empty = {f"past_key_values.{layer}.{kind}.{part}": np.zeros((1, 6, 0, 64), np.float32) for layer in range(4) for kind in ("decoder", "encoder") for part in ("key", "value")}
feeds = {"input_ids": np.array([[generation["decoder_start_token_id"], generation["no_timestamps_token_id"]]], np.int64), "encoder_hidden_states": hidden, "use_cache_branch": np.array([False]), **empty}
tokens = []
encoder_cache = {}
for step in range(min(224, int(np.ceil(len(audio) / 16000 * 8)) + 8)):
    outputs = dict(zip(names, decoder.run(None, feeds)))
    if step == 0:  # cached steps return empty encoder presents; keep the first ones
        encoder_cache = {f"past_key_values.{layer}.encoder.{part}": outputs[f"present.{layer}.encoder.{part}"] for layer in range(4) for part in ("key", "value")}
    logits = outputs["logits"][0, -1, : eot + 1].copy()
    logits[[token for token in suppressed if token <= eot]] = -np.inf
    if step == 0:
        logits[generation["begin_suppress_tokens"]] = -np.inf
    token = int(np.argmax(logits))
    if token == eot:
        break
    tokens.append(token)
    feeds = {"input_ids": np.array([[token]], np.int64), "encoder_hidden_states": hidden, "use_cache_branch": np.array([True])}
    feeds.update({f"past_key_values.{layer}.decoder.{part}": outputs[f"present.{layer}.decoder.{part}"] for layer in range(4) for part in ("key", "value")})
    feeds.update(encoder_cache)

# GPT-2 byte-level decoding from the committed vocabulary; nothing is downloaded.
printable = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
codes = printable + [256 + i for i in range(256 - len(printable))]
byte_of = dict(zip(map(chr, codes), printable + [b for b in range(256) if b not in printable]))
vocabulary = {index: token for token, index in json.load(open(f"{MODEL}/vocab.json")).items()}
decode = lambda ids: bytes(byte_of[char] for i in ids for char in vocabulary[i]).decode("utf-8", "replace")
rms = np.array([np.sqrt(np.mean(audio[i : i + 320] ** 2)) for i in range(0, len(audio) - 320, 160)])
pauses, start = [], None
for index, quiet in enumerate(rms < 0.01):
    if quiet and start is None:
        start = index
    if not quiet and start is not None:
        if index - start >= 8 and start > 0:
            pauses.append([start / 100, index / 100])
        start = None
speech_end = float(np.where(rms >= 0.01)[0][-1] / 100 + 0.02)

json.dump({
    "audio": path,
    "feature_head": features[0, :, :400].round(6).tolist(),
    "feature_frames": 400,
    "tokens": tokens,
    "text": decode(tokens),
    "pauses": pauses,
    "speech_end": speech_end,
}, open("ml/runs/whisper-reference.json", "w"))
print(decode(tokens), pauses, speech_end)
