"""Experimental text-only residual-VQ span classifier; never loaded by the website."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class CascadeConfig:
    vocabulary: int = 30522
    length: int = 128
    width: int = 64
    slots: tuple = (2, 4, 8)
    codes: int = 64
    blocks: int = 1

    def __post_init__(self):
        self.slots = tuple(self.slots)
        if len(self.slots) not in (3, 4) or sorted(set(self.slots)) != list(self.slots):
            raise ValueError('Use three or four strictly increasing scales.')
        if self.slots[0] < 1 or self.slots[-1] > self.length or self.width % 4:
            raise ValueError('Invalid slot schedule or attention width.')


def resize(values, length):
    return F.interpolate(values.transpose(1, 2), size=length, mode='linear', align_corners=False).transpose(1, 2)


class Cascade(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocabulary, config.width, padding_idx=0)
        self.position = nn.Embedding(config.length, config.width)
        block = nn.TransformerEncoderLayer(config.width, 4, config.width * 2, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(block, config.blocks, enable_nested_tensor=False)
        self.codebooks = nn.ModuleList([nn.Embedding(config.codes, config.width) for _ in config.slots])
        for codebook in self.codebooks:
            nn.init.normal_(codebook.weight, std=0.1)
        self.heads = nn.ModuleList([nn.Sequential(nn.LayerNorm(config.width), nn.Linear(config.width, 2)) for _ in config.slots])
        self.decoder = nn.Sequential(nn.LayerNorm(config.width), nn.Linear(config.width, config.vocabulary))

    def encode(self, tokens, mask):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.encoder(self.embedding(tokens) + self.position(positions), src_key_padding_mask=~mask.bool())
        return hidden * mask.unsqueeze(-1)

    def quantize(self, residual, mask, level):
        slots = self.config.slots[level]
        weights = F.adaptive_avg_pool1d(mask.float().unsqueeze(1), slots).transpose(1, 2)
        pooled = F.adaptive_avg_pool1d((residual * mask.unsqueeze(-1)).transpose(1, 2), slots).transpose(1, 2)
        pooled = pooled / weights.clamp_min(1e-6)
        codebook = self.codebooks[level].weight
        distances = pooled.square().sum(-1, keepdim=True) + codebook.square().sum(-1) - 2 * pooled @ codebook.T
        indices = distances.argmin(-1)
        quantized = F.embedding(indices, codebook)
        valid = weights.gt(0).expand_as(pooled)
        loss = ((quantized - pooled.detach()).square() + 0.25 * (pooled - quantized.detach()).square())[valid].mean()
        straight_through = pooled + (quantized - pooled).detach()
        return resize(straight_through * weights.gt(0), residual.shape[1]), loss, indices[weights.squeeze(-1).gt(0)]

    def forward(self, tokens, mask, reconstruct=True, max_levels=None):
        hidden = self.encode(tokens, mask)
        cumulative = torch.zeros_like(hidden)
        logits, losses, codes = [], [], []
        levels = len(self.heads) if max_levels is None else max_levels
        if not 1 <= levels <= len(self.heads):
            raise ValueError('Invalid hierarchy cutoff.')
        for level in range(levels):
            update, loss, indices = self.quantize(hidden - cumulative.detach(), mask, level)
            cumulative = cumulative + update
            logits.append(self.heads[level](cumulative))
            losses.append(loss)
            codes.append(indices)
        return logits, self.decoder(cumulative) if reconstruct else None, torch.stack(losses).mean(), codes

    @torch.inference_mode()
    def predict(self, tokens, mask, temperatures, thresholds):
        if len(temperatures) != len(self.heads) or len(thresholds) != len(self.heads):
            raise ValueError('Calibration must cover every level.')
        if any(value <= 0 for value in temperatures) or any(not 0 <= value <= 1 for value in thresholds):
            raise ValueError('Invalid calibration.')
        hidden = self.encode(tokens, mask)
        cumulative = torch.zeros_like(hidden)
        active = torch.arange(len(tokens), device=tokens.device)
        output = hidden.new_zeros((*tokens.shape, 2))
        exits = torch.zeros(len(tokens), dtype=torch.long, device=tokens.device)
        for level, head in enumerate(self.heads):
            update, _, _ = self.quantize(hidden - cumulative, mask, level)
            cumulative = cumulative + update
            probabilities = (head(cumulative) / temperatures[level]).softmax(-1)
            confidence = probabilities.amax(-1).masked_fill(~mask.bool(), 1).amin(-1)
            done = (confidence.ge(thresholds[level]) & (thresholds[level] < 1)) if level < len(self.heads) - 1 else torch.ones_like(confidence, dtype=torch.bool)
            output[active[done]] = probabilities[done]
            exits[active[done]] = level
            keep = ~done
            if not keep.any():
                break
            active, hidden, cumulative, mask = active[keep], hidden[keep], cumulative[keep], mask[keep]
        return output, exits


def optimizers(model, kind, learning_rate=3e-4, muon_rate=0.02):
    if kind == 'adamw':
        return [torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)]
    if kind != 'muon':
        raise ValueError('Unknown optimizer.')
    hidden, remaining = [], []
    for name, parameter in model.named_parameters():
        (hidden if name.startswith('encoder.') and parameter.ndim == 2 else remaining).append(parameter)
    return [torch.optim.Muon(hidden, lr=muon_rate, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.01),
            torch.optim.AdamW(remaining, lr=learning_rate, weight_decay=0.01)]


def calibrate(logits, labels, mask):
    """Fit temperatures and gates on separate document partitions upstream."""
    temperatures = []
    for scores in logits:
        candidates = torch.logspace(-1, 1, 41, device=scores.device)
        losses = torch.stack([F.cross_entropy(scores[mask] / value, labels[mask]) for value in candidates])
        temperatures.append(float(candidates[losses.argmin()]))
    return temperatures


def select_gates(logits, labels, mask, temperatures, max_error=0.02, min_exit=32):
    active = torch.ones(len(labels), dtype=torch.bool, device=labels.device)
    thresholds = []
    for level, scores in enumerate(logits[:-1]):
        probabilities = (scores / temperatures[level]).softmax(-1)
        confidence = probabilities.amax(-1).masked_fill(~mask, 1).amin(-1)
        wrong = ((probabilities.argmax(-1) != labels) & mask).any(-1)
        chosen = 1.0
        for candidate in torch.linspace(0.5, 0.999, 100, device=scores.device):
            exits = active & confidence.ge(candidate)
            if int(exits.sum()) >= min_exit and float(wrong[exits].float().mean()) <= max_error:
                chosen = float(candidate)
                break
        thresholds.append(chosen)
        if chosen < 1:
            active &= ~confidence.ge(chosen)
    return thresholds + [0.0]
