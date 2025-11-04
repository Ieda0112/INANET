#!/usr/bin/env python3
import re, sys
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

log_path = sys.argv[1]
out_path = sys.argv[2] if len(sys.argv)>2 else "loss_plot.png"

train_pat = re.compile(r"step:\s*\d+,\s*epoch:\s*(\d+),\s*loss:\s*([0-9]*\.?[0-9]+)")
val_pat = re.compile(r"(?i)(?:val|validation)[ _-]*loss[:\s]*([0-9]*\.?[0-9]+)")

train_by_epoch = defaultdict(list)
val_by_epoch = {}

with open(log_path, "r") as f:
    for line in f:
        m = train_pat.search(line)
        if m:
            e = int(m.group(1)); loss = float(m.group(2))
            train_by_epoch[e].append(loss)
            continue
        m2 = val_pat.search(line)
        if m2:
            loss = float(m2.group(1))
            me = re.search(r"epoch[:\s]*([0-9]+)", line, re.I)
            if me:
                e = int(me.group(1))
            else:
                e = len(val_by_epoch)+1
            val_by_epoch[e] = loss

if not train_by_epoch:
    print("train loss がログに見つかりません。ログパスを確認してください:", log_path)
    sys.exit(1)

epochs = sorted(train_by_epoch.keys())
train_mean = [np.mean(train_by_epoch[e]) for e in epochs]
train_last = [train_by_epoch[e][-1] for e in epochs]

plt.figure(figsize=(8,5))
plt.plot(epochs, train_mean, label="train_loss_mean", marker='o')
plt.plot(epochs, train_last, label="train_loss_last", linestyle='--', marker='x')
if val_by_epoch:
    v_epochs = sorted(val_by_epoch.keys())
    v_losses = [val_by_epoch[e] for e in v_epochs]
    plt.plot(v_epochs, v_losses, label="val_loss", marker='s')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out_path)
print("Saved:", out_path)