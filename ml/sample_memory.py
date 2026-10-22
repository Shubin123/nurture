import argparse
import json
import signal
import time

import psutil

parser = argparse.ArgumentParser()
parser.add_argument('pid', type=int)
args = parser.parse_args()
running = True


def stop(signum, frame):
    global running
    running = False


signal.signal(signal.SIGTERM, stop)
samples = []
root = psutil.Process(args.pid)
while running:
    try:
        processes = [root] + root.children(recursive=True)
        rss = sum(process.memory_info().rss for process in processes if process.is_running())
        samples.append(rss)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    time.sleep(0.05)
print(json.dumps({'samples': len(samples), 'interval_seconds': 0.05, 'initial_process_tree_rss_bytes': samples[0] if samples else None, 'sampled_peak_process_tree_rss_bytes': max(samples) if samples else None, 'sampled_incremental_rss_bytes': max(samples) - samples[0] if samples else None, 'scope': 'Whole Chrome process tree RSS; shared pages can be double-counted. This is a process-tree proxy, not model allocation or a portable memory guarantee.'}), flush=True)
