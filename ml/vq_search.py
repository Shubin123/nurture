"""Exact codebook lookup kernels. No approximate index or persistent norm cache."""

import torch
from torch.nn import functional as F


KERNELS = ('reference', 'dot', 'tiled', 'cdist')


@torch.no_grad()
def nearest_codes(values, codebook, metric='euclidean', kernel='dot', tile_size=256):
    if metric not in ('euclidean', 'cosine') or kernel not in KERNELS:
        raise ValueError('Unknown distance or lookup kernel.')
    if values.ndim < 2 or codebook.ndim != 2 or values.shape[-1] != codebook.shape[-1] or not len(codebook):
        raise ValueError('Expected vectors and a nonempty aligned codebook.')
    if tile_size < 1:
        raise ValueError('Tile size must be positive.')
    shape = values.shape[:-1]
    dtype = torch.float64 if values.dtype == torch.float64 or codebook.dtype == torch.float64 else torch.float32
    vectors = values.reshape(-1, values.shape[-1]).to(dtype=dtype)
    codes = codebook.to(dtype=dtype)
    if metric == 'cosine':
        vectors, codes = F.normalize(vectors, dim=-1), F.normalize(codes, dim=-1)
    if kernel == 'reference':
        distances = -vectors @ codes.T if metric == 'cosine' else vectors.square().sum(-1, keepdim=True) + codes.square().sum(-1) - 2 * vectors @ codes.T
        indices = distances.argmin(-1)
    elif kernel == 'cdist':
        distances = torch.cdist(vectors, codes, compute_mode='donot_use_mm_for_euclid_dist')
        if metric == 'cosine':
            distances = distances.square() - codes.square().sum(-1)
        indices = distances.argmin(-1)
    elif kernel == 'dot':
        scores = vectors @ codes.T
        if metric == 'euclidean':
            scores -= codes.square().sum(-1) * 0.5
        indices = scores.argmax(-1)
    else:
        best = vectors.new_full((len(vectors),), -torch.inf)
        indices = torch.zeros(len(vectors), device=vectors.device, dtype=torch.long)
        for start in range(0, len(codes), tile_size):
            part = codes[start:start + tile_size]
            scores = vectors @ part.T
            if metric == 'euclidean':
                scores -= part.square().sum(-1) * 0.5
            score, local = scores.max(-1)
            replace = score > best
            indices = torch.where(replace, local + start, indices)
            best = torch.maximum(best, score)
    if metric == 'cosine':
        indices = torch.where(vectors.square().sum(-1).eq(0), torch.zeros_like(indices), indices)
    return indices.reshape(shape)
