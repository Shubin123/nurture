"""Tune exact lookup on synthetic shapes, then confirm on fresh random inputs."""

import argparse
import platform
import random
import time

import numpy as np
import torch

from common import ROOT, SEED, digest, write_json
from vq_search import KERNELS, nearest_codes


def sample_times(values, codes, kernels, repeats, rounds, seed):
    times = {kernel: [] for kernel in kernels}
    randomizer = random.Random(seed)
    for _ in range(rounds):
        order = list(kernels)
        randomizer.shuffle(order)
        for kernel in order:
            for _ in range(5):
                nearest_codes(values, codes, kernel=kernel)
            started = time.perf_counter_ns()
            for _ in range(repeats):
                nearest_codes(values, codes, kernel=kernel)
            times[kernel].append((time.perf_counter_ns() - started) / repeats / 1000)
    return {kernel: {'median_us': float(np.median(samples)), 'round_us': samples} for kernel, samples in times.items()}


def agreements(values, codes):
    reference = nearest_codes(values.double(), codes.double(), kernel='cdist')
    return {kernel: float(nearest_codes(values, codes, kernel=kernel).eq(reference).float().mean()) for kernel in KERNELS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=50)
    parser.add_argument('--rounds', type=int, default=7)
    parser.add_argument('--output', default='vq-kernel-search')
    args = parser.parse_args()
    if args.repeats < 1 or args.rounds < 3:
        parser.error('Use positive repeats and at least three timing rounds.')
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(SEED)
    report = {'seed': SEED, 'torch': torch.__version__, 'platform': platform.platform(), 'threads': 1, 'args': vars(args),
              'source_sha256': {name: digest(ROOT / 'ml' / name) for name in ('vq_search.py', 'benchmark_vq_search.py')},
              'scope': 'CPU float32 lookup only, including norm computation; synthetic inputs, not browser or end-to-end model timing.', 'workloads': []}
    for queries, codes_count, width in ((16, 64, 64), (64, 64, 64), (1024, 128, 16), (1024, 256, 64), (1024, 4096, 64)):
        values = torch.randn(queries, width, generator=generator)
        codes = torch.randn(codes_count, width, generator=generator)
        tuning_agreement = agreements(values, codes)
        tuning = sample_times(values, codes, KERNELS, args.repeats, args.rounds, SEED)
        eligible = [kernel for kernel in KERNELS if tuning_agreement[kernel] == 1]
        selected = min(eligible, key=lambda kernel: tuning[kernel]['median_us']) if eligible else None
        fresh_values = torch.randn(queries, width, generator=generator)
        fresh_codes = torch.randn(codes_count, width, generator=generator)
        confirmation_agreement = agreements(fresh_values, fresh_codes)
        confirmation = sample_times(fresh_values, fresh_codes, KERNELS, args.repeats, args.rounds, SEED + 1)
        report['workloads'].append({'queries': queries, 'codes': codes_count, 'width': width, 'selected_on_tuning': selected,
                                    'tuning_agreement': tuning_agreement, 'confirmation_agreement': confirmation_agreement,
                                    'tuning': tuning, 'confirmation': confirmation,
                                    'confirmed_reference_over_selected_ratio': confirmation['reference']['median_us'] / confirmation[selected]['median_us'] if selected else None,
                                    'full_score_matrix_bytes': queries * codes_count * 4, 'tiled_score_block_bytes': queries * min(codes_count, 256) * 4})
    write_json(ROOT / f'docs/reports/{args.output}.json', report)


if __name__ == '__main__':
    main()
