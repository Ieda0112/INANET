#!/usr/bin/env python3
"""Plot evaluation metrics from batch_eval.py log files.

複数の eval_*.log ファイルを読み込み、epoch ごとの fmeasure を
1 つのグラフにプロットします。

使い方:
-------
  python scripts/plot_metrics.py eval_20251202_134531.log eval_20251203_120000.log \\
    --output metrics.png \\
    --metric fmeasure \\
    --title "F-measure progression"

オプション:
  --output PATH     : 出力画像ファイルのパス (デフォルト: metrics.png)
  --metric NAME     : プロットする指標 (precision/recall/fmeasure, デフォルト: fmeasure)
  --title TEXT      : グラフのタイトル
  --xlabel TEXT     : X軸ラベル (デフォルト: Epoch)
  --ylabel TEXT     : Y軸ラベル (デフォルト: 指定したメトリクス名)
  --figsize W H     : 図のサイズ (デフォルト: 10 6)
  --dpi N           : 出力解像度 (デフォルト: 150)
  --no-legend       : 凡例を非表示
  --grid            : グリッド線を表示
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI なし環境でも動作するように


# ログ行のパターン: epoch_X_minibatch_Y Precision:... Recall:... fmeasure:...
LINE_PATTERN = re.compile(
    r"epoch_(\d+)_minibatch_\d+\s+"
    r"(?:Precision:([0-9.]+)\(\d+\)\s*)?"
    r"(?:Recall:([0-9.]+)\(\d+\)\s*)?"
    r"(?:fmeasure:([0-9.]+)\(\d+\))?"
)


def parse_log_file(path: Path) -> Dict[str, List[Tuple[int, float]]]:
    """
    ログファイルを読み込み、{metric_name: [(epoch, value), ...]} を返す。
    """
    metrics: Dict[str, List[Tuple[int, float]]] = {
        "precision": [],
        "recall": [],
        "fmeasure": [],
    }
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            match = LINE_PATTERN.match(line.strip())
            if not match:
                continue
            epoch = int(match.group(1))
            precision_str = match.group(2)
            recall_str = match.group(3)
            fmeasure_str = match.group(4)
            if precision_str:
                metrics["precision"].append((epoch, float(precision_str)))
            if recall_str:
                metrics["recall"].append((epoch, float(recall_str)))
            if fmeasure_str:
                metrics["fmeasure"].append((epoch, float(fmeasure_str)))
    return metrics


def plot_metrics(
    log_paths: List[Path],
    metric: str,
    output_path: Path,
    title: str | None,
    xlabel: str,
    ylabel: str | None,
    figsize: Tuple[float, float],
    dpi: int,
    show_legend: bool,
    show_grid: bool,
) -> None:
    """
    複数のログファイルから指定メトリクスをプロットし、画像ファイルに保存。
    """
    fig, ax = plt.subplots(figsize=figsize)

    for log_path in log_paths:
        data = parse_log_file(log_path)
        if metric not in data or not data[metric]:
            print(f"警告: {log_path.name} に {metric} データが見つかりませんでした")
            continue
        epochs, values = zip(*sorted(data[metric]))
        label = log_path.stem  # ファイル名 (拡張子なし) をラベルに使用
        ax.plot(epochs, values, marker="o", label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or metric.capitalize())
    if title:
        ax.set_title(title)
    if show_legend and len(log_paths) > 1:
        ax.legend()
    if show_grid:
        ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"グラフを保存しました: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="batch_eval.py で生成したログファイルをグラフ化します。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "log_files",
        nargs="+",
        type=Path,
        help="eval_*.log ファイルのパス (複数指定可)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("metrics.png"),
        help="出力画像ファイルのパス",
    )
    parser.add_argument(
        "--metric",
        choices=["precision", "recall", "fmeasure"],
        default="fmeasure",
        help="プロットする指標",
    )
    parser.add_argument(
        "--title",
        type=str,
        help="グラフのタイトル",
    )
    parser.add_argument(
        "--xlabel",
        type=str,
        default="Epoch",
        help="X軸のラベル",
    )
    parser.add_argument(
        "--ylabel",
        type=str,
        help="Y軸のラベル (未指定時は metric 名を使用)",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[10.0, 6.0],
        metavar=("WIDTH", "HEIGHT"),
        help="図のサイズ (インチ)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="出力解像度 (DPI)",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="凡例を非表示にする",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="グリッド線を表示する",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_paths = [p.resolve() for p in args.log_files]
    for path in log_paths:
        if not path.exists():
            raise FileNotFoundError(f"ログファイルが見つかりません: {path}")

    plot_metrics(
        log_paths=log_paths,
        metric=args.metric,
        output_path=args.output,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        show_legend=not args.no_legend,
        show_grid=args.grid,
    )


if __name__ == "__main__":
    main()
