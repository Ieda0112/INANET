#!/usr/bin/env python3
"""Utility to evaluate every checkpoint inside a directory and log metrics.

使い方:
-------
指定したモデルディレクトリ内のすべてのチェックポイントを評価し、
各チェックポイントの Precision/Recall/Fmeasure をログファイルに記録します。

基本的な使用例:
  python scripts/batch_eval.py experiments/seg_detector/COO_resnet50_deform_thre_INA.yaml outputs/workspace/INANET_ieda/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss/model/20251208_111429/ --box_thresh 0.6 --polygon

オプション:
  --limit N         : 最初の N 件のみ評価
  --reverse         : 新しいチェックポイントから順に処理
  --skip-final      : 'final' という名前のチェックポイントを除外
  --dry-run         : 実際には eval.py を実行せず、コマンドのみ表示
  --append          : 既存ログファイルに追記 (デフォルトは上書き)
  --strict          : メトリクス抽出失敗時に即座に停止

出力形式:
  生成されるログファイルの各行は次の形式になります:
    epoch_0_minibatch_0 Precision:0.859977(654) Recall:0.465278(654) fmeasure:0.603851
    epoch_3_minibatch_3000 Precision:0.845182(654) Recall:0.473926(654) fmeasure:0.607123
    ...

eval.py に渡す追加オプション (--box_thresh, --polygon など) は、
コマンドライン引数の末尾に自由に追加できます。

グラフ化:
--------
生成されたログファイルを可視化するには、scripts/plot_metrics.py を使用してください:
  python scripts/plot_metrics.py eval_20251202_134531.log eval_20251203_120000.log --output metrics.png
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

METRIC_PATTERN = re.compile(r"(precision|recall|fmeasure)\s*:\s*([0-9]*\.?[0-9]+)\s*\((\d+)\)", re.IGNORECASE)
MODEL_NAME_PATTERN = re.compile(r"model_epoch_(\d+)_minibatch_(\d+)")

MetricDict = Dict[str, Tuple[float, int]]


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="指定フォルダ内のすべてのモデル重みで eval.py を実行し、評価指標をログにまとめます。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("exp", help="experiments/ 以下の YAML など eval.py に渡す設定ファイル")
    parser.add_argument("model_dir", help="ModelSaver が生成したチェックポイント群が格納されたディレクトリ")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="eval.py を実行する Python 実行ファイル",
    )
    parser.add_argument(
        "--eval-script",
        type=Path,
        help="eval.py へのパス (未指定ならリポジトリ直下の eval.py を使用)",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        help="評価結果を書き出すログファイルのパス (未指定なら model_dir 内に eval_<dir>.log を作成)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="既存ログファイルに追記する (デフォルトは上書き)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="評価するチェックポイント数の上限 (未指定なら全件)",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="新しい (番号の大きい) チェックポイントから処理する",
    )
    parser.add_argument(
        "--skip-final",
        action="store_true",
        help="final という名前のチェックポイントをスキップする",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="コマンドを表示するだけで実行しない",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="メトリクス抽出に失敗したら例外を投げて停止する (デフォルトは警告のみ)",
    )

    return parser.parse_known_args()


def resolve_eval_script(path: Optional[Path]) -> Path:
    if path is not None:
        return path.resolve()
    return Path(__file__).resolve().parents[1] / "eval.py"


def iter_checkpoints(model_dir: Path, include_final: bool = True) -> Iterable[Path]:
    candidates: List[Path] = []
    for entry in sorted(model_dir.iterdir()):
        if not entry.is_file():
            continue
        if entry.name == "final" and include_final:
            candidates.append(entry)
        elif entry.name.startswith("model_"):
            candidates.append(entry)
    def sort_key(path: Path):
        match = MODEL_NAME_PATTERN.match(path.name)
        if match:
            return int(match.group(1)), int(match.group(2))
        if path.name == "final":
            return (sys.maxsize, sys.maxsize)
        return (sys.maxsize, path.name)
    return sorted(candidates, key=sort_key)


def parse_metrics(output: str) -> MetricDict:
    metrics: MetricDict = {}
    for match in METRIC_PATTERN.finditer(output):
        name, value, count = match.groups()
        metrics[name.lower()] = (float(value), int(count))
    return metrics


def format_metrics(metrics: MetricDict) -> str:
    labels = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("fmeasure", "fmeasure"),
    ]
    parts = []
    for key, label in labels:
        if key in metrics:
            value, count = metrics[key]
            parts.append(f"{label}:{value:.6f}({count})")
    return " ".join(parts) if parts else "(no metrics)"


def checkpoint_display_name(path: Path) -> str:
    if path.name.startswith("model_"):
        return path.name[len("model_"):]
    return path.name


def get_final_epoch(checkpoints: List[Path]) -> int:
    """チェックポイントリストから最後のエポック数を取得し、最も近い10の倍数を返す"""
    max_epoch = 0
    for checkpoint in checkpoints:
        if checkpoint.name.startswith("model_"):
            match = MODEL_NAME_PATTERN.match(checkpoint.name)
            if match:
                epoch = int(match.group(1))
                max_epoch = max(max_epoch, epoch)
    # 最も近い10の倍数に丸める（切り上げ）
    return ((max_epoch + 9) // 10) * 10


def run_eval(
    python_bin: str,
    eval_script: Path,
    exp_path: str,
    checkpoint_path: Path,
    extra_args: List[str],
) -> Tuple[int, str, str]:
    cmd = [python_bin, str(eval_script), exp_path, *extra_args, "--resume", str(checkpoint_path)]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    combined = stdout + "\n" + stderr if stderr else stdout
    return completed.returncode, combined, stdout


def ensure_log_path(log_path: Optional[Path], model_dir: Path) -> Path:
    if log_path is not None:
        if log_path.is_dir():
            raise ValueError("--log-path にはファイル名を指定してください")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        return log_path
    # デフォルト出力先を outputs/eval_models に変更
    target_dir = Path("outputs") / "eval_models"
    target_dir.mkdir(parents=True, exist_ok=True)
    default_name = f"eval_{model_dir.name}.log"
    return (target_dir / default_name)


def main() -> None:
    args, extra_eval_args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.exists() or not model_dir.is_dir():
        raise FileNotFoundError(f"model_dir が見つかりません: {model_dir}")

    eval_script = resolve_eval_script(args.eval_script)
    if not eval_script.exists():
        raise FileNotFoundError(f"eval.py が見つかりません: {eval_script}")

    checkpoints = list(
        iter_checkpoints(model_dir, include_final=not args.skip_final)
    )
    if args.reverse:
        checkpoints.reverse()
    if args.limit is not None:
        checkpoints = checkpoints[: args.limit]

    if not checkpoints:
        raise RuntimeError(f"{model_dir} に評価可能なチェックポイントが見つかりませんでした")

    log_path = ensure_log_path(args.log_path, model_dir)
    log_mode = "a" if args.append else "w"
    
    # final チェックポイント用のエポック数を計算
    final_epoch = get_final_epoch(checkpoints)

    with log_path.open(log_mode, encoding="utf-8") as writer:
        for idx, checkpoint in enumerate(checkpoints, start=1):
            label = checkpoint_display_name(checkpoint)
            print(f"[{idx}/{len(checkpoints)}] Evaluating {label} ...")
            
            # ログに書き込む際のラベル（finalの場合はエポック数を付与）
            log_label = f"epoch_{final_epoch}_minibatch_final" if checkpoint.name == "final" else label
            
            if args.dry_run:
                writer.write(f"{log_label} DRY-RUN\n")
                continue
            returncode, output, _ = run_eval(
                args.python,
                eval_script,
                args.exp,
                checkpoint,
                extra_eval_args,
            )
            metrics = parse_metrics(output)
            if returncode != 0:
                writer.write(f"{log_label} ERROR(code={returncode})\n")
                if args.strict:
                    raise RuntimeError(f"eval.py が失敗しました: {output}")
                continue
            if not metrics:
                writer.write(f"{log_label} NO_METRICS\n")
                if args.strict:
                    raise RuntimeError(
                        f"メトリクスを抽出できませんでした\n----- output -----\n{output}"
                    )
                continue
            writer.write(f"{log_label} {format_metrics(metrics)}\n")
            writer.flush()

    print(f"完了: {log_path}")


if __name__ == "__main__":
    main()
