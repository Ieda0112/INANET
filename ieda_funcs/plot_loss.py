#!/usr/bin/env python3
"""
Plot epoch-level avg_train_loss and avg_valid_loss from a training log.
Saves <logname>_epoch_losses.png next to the log file.
Usage:
  python plot_epoch_losses.py path/to/training_YYYYMMDD_HHMMSS.log
"""
import re
import sys
from pathlib import Path
import math

def parse_epoch_lines(path):
    train_pattern = re.compile(r"TRAIN_EPOCH_END\s+epoch:\s*([0-9]+).*avg_train_loss:([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")
    valid_pattern = re.compile(r"VALID_EPOCH_END\s+epoch:\s*([0-9]+).*avg_valid_loss:([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")
    e2train = {}
    e2valid = {}
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for ln in f:
            m = train_pattern.search(ln)
            if m:
                ep = int(m.group(1))
                val = float(m.group(2))
                e2train[ep] = val
            m2 = valid_pattern.search(ln)
            if m2:
                ep = int(m2.group(1))
                val = float(m2.group(2))
                e2valid[ep] = val
    return e2train, e2valid


def main():
    import argparse
    p = None
    parser = argparse.ArgumentParser(description='Plot epoch avg train/valid loss from training log')
    parser.add_argument('logfile', help='path to training log (.log)')
    parser.add_argument('--series', choices=['train', 'valid', 'both'], default='both',
                        help='which series to plot (default: both)')
    parser.add_argument('--xstep', type=int, default=5, help='x-axis tick step in epochs (default 5)')
    args = parser.parse_args()
    p = Path(args.logfile)
    if not p.exists():
        print(f"Log file not found: {p}")
        sys.exit(2)

    e2train, e2valid = parse_epoch_lines(p)
    if not e2train and not e2valid:
        print("No epoch lines found in the log.")
        sys.exit(0)

    epochs = sorted(set(list(e2train.keys()) + list(e2valid.keys())))
    if not epochs:
        print("No epochs parsed.")
        sys.exit(0)
    min_ep = min(epochs)
    max_ep = max(epochs)

    # build arrays aligned to epochs
    xs = epochs
    train_vals = [e2train.get(e, float('nan')) for e in xs]
    valid_vals = [e2valid.get(e, float('nan')) for e in xs]

    # plotting
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print('matplotlib is required to create the plot. Install with: pip install matplotlib')
        raise

    plt.figure(figsize=(10, 5))
    # choose which series to plot
    to_plot = args.series if 'args' in locals() else 'both'
    if to_plot in ('both', 'train'):
        plt.plot(xs, train_vals, marker='o', linestyle='-', color='blue', label='avg_train_loss')
    if to_plot in ('both', 'valid'):
        plt.plot(xs, valid_vals, marker='o', linestyle='-', color='orange', label='avg_valid_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # If user requested only the train series, keep y-axis autoscaled.
    # Otherwise (both or valid) force y-axis to start from 0.
    if 'args' in locals() and args.series == 'train':
        pass
    else:
        plt.ylim(bottom=0)
    # title: filename without .log
    title = p.name
    if title.endswith('.log'):
        title = title[:-4]
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # set x-axis range
    left = 0
    right = max_ep
    plt.xlim(left, right)
    # show tick labels every xstep epochs (and always include the rightmost epoch)
    step = args.xstep if 'args' in locals() else 5
    ticks = list(range(left, right+1, step))
    if len(ticks) == 0 or ticks[-1] != right:
        # ensure the final epoch is labeled
        ticks.append(right)
    plt.xticks(ticks)

    # save under outputs/training_graph
    from pathlib import Path as _P
    graph_dir = _P('outputs') / 'training_graph'
    graph_dir.mkdir(parents=True, exist_ok=True)
    suffix = ''
    if 'args' in locals():
        if args.series == 'train':
            suffix = '_train'
        elif args.series == 'valid':
            suffix = '_valid'
    out_path = graph_dir / (p.stem + suffix + '.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved epoch losses plot to: {out_path}")

if __name__ == '__main__':
    main()
