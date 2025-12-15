#!/usr/bin/env python3
"""Plot evaluation metrics vs epochs for one or more eval logs.

使い方:
-------
batch_eval.py で生成した eval_*.log ファイルから、
epoch ごとの評価指標（precision/recall/fmeasure）の推移をグラフ化します。
複数のログファイルを指定すると、1つのグラフに重ねて表示されます。

基本的な使用例:
  python ieda_funcs/plot_eval_metrics.py outputs/.../eval_20251202_134531.log outputs/.../eval_20251203_120000.log --metric fmeasure --output outputs/training_graph/fmeasure_comparison.png

オプション:
  --metric {precision,recall,fmeasure}
                        プロットする指標 (デフォルト: fmeasure)
  --labels LABEL1 LABEL2 ...
                        各ログファイルのラベル (未指定時はファイル名を使用)
  --xstep N            X軸の目盛り間隔 (デフォルト: 1)
  --ymin VALUE         Y軸の下限 (デフォルト: 0.0)
  --ymax VALUE         Y軸の上限 (デフォルト: 自動)
  --output PATH        出力画像ファイルのパス
                       (未指定時は outputs/training_graph/<metric>_overlay_<timestamp>.png)

出力:
  生成されたグラフは指定したパスまたはデフォルトパスに PNG 形式で保存されます。
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

EPOCH_RE = re.compile(r"epoch[_\s]*(\d+)", re.IGNORECASE)
METRIC_RE = re.compile(
    r"(precision|recall|fmeasure)\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)

MetricMap = Dict[int, Dict[str, float]]


def parse_eval_log(path: Path) -> MetricMap:
    """Return a mapping of epoch -> {metric_name: value} for the given eval log."""
    metrics_by_epoch: Dict[int, Dict[str, float]] = {}
    max_epoch = -1
    
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            # 'final' 行の処理
            if line.strip().startswith("final"):
                metric_entries = METRIC_RE.findall(line)
                if metric_entries:
                    # final は最大エポック + 1 として扱う
                    final_epoch = max_epoch + 1
                    metric_dict = metrics_by_epoch.setdefault(final_epoch, {})
                    for name, value in metric_entries:
                        metric_dict[name.lower()] = float(value)
                continue
            
            # 通常の epoch_X_minibatch_Y 行の処理
            epoch_match = EPOCH_RE.search(line)
            if not epoch_match:
                continue
            epoch = int(epoch_match.group(1))
            max_epoch = max(max_epoch, epoch)
            metric_entries = METRIC_RE.findall(line)
            if not metric_entries:
                continue
            metric_dict = metrics_by_epoch.setdefault(epoch, {})
            for name, value in metric_entries:
                metric_dict[name.lower()] = float(value)
    return dict(sorted(metrics_by_epoch.items()))


def extract_series(metric_map: MetricMap, metric: str) -> Tuple[List[int], List[float]]:
    epochs: List[int] = []
    values: List[float] = []
    for epoch, metrics in metric_map.items():
        if metric not in metrics:
            continue
        # エポック番号を +1 してプロット（epoch 0 を epoch 1 の位置に表示）
        epochs.append(epoch + 1)
        values.append(metrics[metric])
    return epochs, values


def default_labels(logs: Iterable[Path]) -> List[str]:
    labels: List[str] = []
    for path in logs:
        labels.append(path.stem)
    return labels


def ensure_output_path(path: Path | None, metric: str, log_paths: List[Path]) -> Path:
    target_dir = Path("outputs") / "eval_graph"
    target_dir.mkdir(parents=True, exist_ok=True)
    if path is not None:
        if path.is_dir():
            raise ValueError("--output must be a file path, not a directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    
    # 最初のログファイル名からベース名を取得
    # eval_20251208_111429.log -> eval_20251208_111429
    base_name = log_paths[0].stem
    
    metric_abbrev = {"precision": "P", "recall": "R", "fmeasure": "F1"}
    abbrev = metric_abbrev.get(metric, metric)
    
    # eval_20251208_111429_F1.png のような形式
    return target_dir / f"{base_name}_{abbrev}.png"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot metric progression (default: fmeasure) across epochs from one or more eval logs."
        )
    )
    parser.add_argument("logs", nargs="+", help="Paths to eval_*.log files")
    parser.add_argument(
        "--metric",
        choices=["precision", "recall", "fmeasure"],
        default="fmeasure",
        help="Metric to plot (default: fmeasure)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Optional custom labels for each log (must match number of logs)",
    )
    parser.add_argument(
        "--xstep",
        type=int,
        default=5,
        help="Epoch spacing between x-axis ticks (default: 5)",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=0.0,
        help="Lower bound for y-axis (default: 0.0)",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=1.0,
        help="Upper bound for y-axis (default: 1.0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination image path (default: outputs/training_graph/<metric>_overlay_<timestamp>.png)",
    )
    args = parser.parse_args()

    log_paths = [Path(p).expanduser().resolve() for p in args.logs]
    for path in log_paths:
        if not path.exists():
            parser.error(f"Eval log not found: {path}")

    if args.labels is not None and len(args.labels) != len(log_paths):
        parser.error("Number of --labels entries must match number of logs")

    series_labels = args.labels if args.labels is not None else default_labels(log_paths)

    metric_maps = [parse_eval_log(p) for p in log_paths]
    series_data: List[Tuple[List[int], List[float]]] = []
    for idx, metric_map in enumerate(metric_maps):
        epochs, values = extract_series(metric_map, args.metric)
        if not epochs:
            parser.error(f"No '{args.metric}' data found in {log_paths[idx]}")
        series_data.append((epochs, values))

    try:
        import matplotlib
        matplotlib.use("Agg")  # use headless backend
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - matplotlib import failure
        parser.error(f"matplotlib is required for plotting: {exc}")

    plt.figure(figsize=(10, 5))
    for (epochs, values), label in zip(series_data, series_labels):
        plt.plot(epochs, values, marker="o", linestyle="-", label=label)
    plt.xlabel("epoch")
    plt.ylabel(args.metric)
    plt.title(f"{args.metric} vs epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()

    if args.ymin is not None:
        if args.ymax is not None and args.ymax <= args.ymin:
            parser.error("--ymax must be greater than --ymin")
        plt.ylim(bottom=args.ymin)
    if args.ymax is not None:
        plt.ylim(top=args.ymax)
    
    # Y軸の罫線を0.2刻みに設定
    import numpy as np
    y_ticks = np.arange(args.ymin, args.ymax + 0.01, 0.2)
    plt.yticks(y_ticks)

    if series_data:
        max_epoch = max(epoch for epochs, _ in series_data for epoch in epochs)
    else:
        parser.error("No data available to plot")

    plt.xlim(left=0, right=max_epoch)
    if args.xstep > 0:
        ticks = list(range(0, max_epoch + 1, args.xstep))
        if ticks[-1] != max_epoch:
            ticks.append(max_epoch)
        plt.xticks(ticks)

    out_path = ensure_output_path(args.output, args.metric, log_paths)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
