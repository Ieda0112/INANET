#!/usr/bin/env python3
"""
Parse a log file and plot loss vs step.
Usage:
  python plot_loss.py [path/to/1.txt] [--smooth N]
Saves `loss_plot.png` in the same folder.
"""
import re
import sys
from pathlib import Path

INPUT_DEFAULT = "2025-10-23_16:49:30,151.txt"

def parse_lines(path):
    pattern = re.compile(r"step:\s*([0-9]+),\s*epoch:\s*([0-9]+),\s*loss:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    steps = []
    losses = []
    epochs = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            m = pattern.search(ln)
            if m:
                step = int(m.group(1))
                epoch = int(m.group(2))
                loss = float(m.group(3))
                steps.append(step)
                epochs.append(epoch)
                losses.append(loss)
    return steps, epochs, losses


def moving_average(x, w):
    if w <= 1:
        return x
    import numpy as np
    arr = np.array(x, dtype=float)
    if len(arr) < w:
        return arr.tolist()
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    ma = (cumsum[w:] - cumsum[:-w]) / float(w)
    # prepend first values to keep same length
    pad = [arr[0]] * (w-1)
    return (pad + ma.tolist())


def main():
    import argparse
    p = argparse.ArgumentParser(description="Plot loss vs step from training log")
    p.add_argument("logfile", nargs="?", default=INPUT_DEFAULT, help="path to log file (default: 1.txt)")
    p.add_argument("--smooth", type=int, default=1, help="moving average window (default 1 = no smoothing)")
    args = p.parse_args()

    path = Path(args.logfile)
    if not path.exists():
        print(f"Log file not found: {path}")
        sys.exit(2)

    steps, epochs, losses = parse_lines(path)
    if not steps:
        print(f"No step/loss lines found in {path}")
        sys.exit(0)

    # Optionally smooth
    if args.smooth and args.smooth > 1:
        try:
            smooth_losses = moving_average(losses, args.smooth)
        except Exception as e:
            print("Error applying smoothing: ", e)
            smooth_losses = losses
    else:
        smooth_losses = losses

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
    except Exception:
        pass
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib is required to create the plot. Install with: pip install matplotlib")
        raise

    # --- Plot: loss vs step (existing) ---
    plt.figure(figsize=(10,4))
    # plt.plot(steps, losses, color='lightgray', linewidth=1, label='raw')
    if args.smooth and args.smooth > 1:
        plt.plot(steps[:len(smooth_losses)], smooth_losses, color='red', linewidth=1, label=f'smoothed(w={args.smooth})')
    else:
        plt.plot(steps, losses, color='blue', linewidth=1, label='loss')
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title(f'Learning Process')
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_step = path.parent / 'loss_plot.png'
    plt.tight_layout()
    plt.savefig(out_step, dpi=150)
    print(f"Saved step plot to: {out_step}  (points plotted: {len(steps)})")

    # --- Plot: loss vs epoch (aggregate averages per epoch) ---
    # build epoch -> list(losses)
    from collections import defaultdict
    e2l = defaultdict(list)
    for e, l in zip(epochs, losses):
        e2l[e].append(l)
    epoch_sorted = sorted(e2l.keys())
    epoch_avg = [sum(e2l[e]) / len(e2l[e]) for e in epoch_sorted]

    plt.figure(figsize=(10,4))
    # # raw epoch points (could be multiple per epoch)
    # plt.scatter(epochs, losses, color='lightgray', s=10, label='raw')
    # averaged per epoch
    plt.plot(epoch_sorted, epoch_avg, color='blue', linewidth=1.5, label='epoch_avg')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(f'Learning Process')
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_epoch = path.parent / 'loss_plot_epoch.png'
    plt.tight_layout()
    plt.savefig(out_epoch, dpi=150)
    print(f"Saved epoch plot to: {out_epoch}  (epochs plotted: {len(epoch_sorted)})")


if __name__ == '__main__':
    main()
