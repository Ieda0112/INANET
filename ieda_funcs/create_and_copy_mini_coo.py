#!/usr/bin/env python3
"""create_and_copy_mini_coo.py

Create small COO dataset lists (train/valid/test) and optionally copy the
corresponding images and GTs into an output mini dataset directory.

日本語:
既存の `datasets/COO` にある list ファイル（train_list.txt, valid_list.txt, test_list.txt）
から指定個数を取り出して `out_root` に書き出し、必要なら画像と GT をコピーします。

主な機能:
- サンプリング方法: shuffle（ランダムサンプリング）または head（先頭から）
- 再現性のための seed 指定
- コピーはドライラン / 実行 / 上書き制御 可能
- 画像／GT 検索順は引数で指定可能

例:
    python3 ieda_funcs/create_and_copy_mini_coo.py --mode head --train 420 --valid 90 --test 90 \
            --src-root datasets/COO --out-root datasets/mini_COO --dry-run

画像・アノテーションをコピーする例（日本語）:

- ドライランで実行予定を確認する（コピーは行わない）:
    python3 ieda_funcs/create_and_copy_mini_coo.py --mode head --train 420 --valid 90 --test 90 \
            --src-root datasets/COO --out-root datasets/mini_COO --dry-run --show-each

- 実際に画像と GT（アノテーション）をコピーする（既定では既存ファイルを上書きします）:
    python3 ieda_funcs/create_and_copy_mini_coo.py --mode head --train 420 --valid 90 --test 90 \
            --src-root datasets/COO --out-root datasets/mini_COO --copy

-- 既存ファイルを上書きしないでコピーする（上書きを無効にする例）:
    python3 ieda_funcs/create_and_copy_mini_coo.py --mode head --train 420 --valid 90 --test 90 \
            --src-root datasets/COO --out-root datasets/mini_COO --copy --no-overwrite

補足:
- 画像は `--image-dirs` で指定した順（デフォルトは `test_images, train_images, valid_images`）で検索し、最初に見つかったファイルをコピーします。
- アノテーション（GT）は画像ファイル名に ".txt" を付けたファイル名（例: `AisazuNihaIrarenai_003.jpg.txt`）として `--gt-dirs`（デフォルト: `test_gts, train_gts, valid_gts`）から検索してコピーします。
- 出力先ディレクトリ構成は `--out-root` 以下に `train_list.txt` / `valid_list.txt` / `test_list.txt` と、それぞれ `train_images/` `train_gts/` 等が作られます。
"""
import argparse
import os
import random
import shutil
from typing import List, Tuple


def read_list(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]


def write_list(path: str, items: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for it in items:
            f.write(it + '\n')


def sample_items(items: List[str], k: int, mode: str, seed: int = None) -> List[str]:
    if k <= 0:
        return []
    if mode == 'head':
        return items[:k]
    elif mode == 'shuffle':
        rng = random.Random(seed)
        if k >= len(items):
            out = items.copy()
            rng.shuffle(out)
            return out
        return rng.sample(items, k)
    else:
        raise ValueError('Unknown mode: ' + str(mode))


def find_file(src_root: str, candidate_dirs: List[str], filename: str) -> Tuple[str, str]:
    for d in candidate_dirs:
        p = os.path.join(src_root, d, filename)
        if os.path.isfile(p):
            return p, d
    return None, None


def copy_files_for_list(items: List[str], src_root: str, out_root: str,
                        list_prefix: str, image_dirs: List[str], gt_dirs: List[str],
                        dry_run: bool, overwrite: bool, show_each: bool) -> dict:
    img_out = os.path.join(out_root, f"{list_prefix}_images")
    gt_out = os.path.join(out_root, f"{list_prefix}_gts")
    if not dry_run:
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(gt_out, exist_ok=True)

    summary = {'items': len(items), 'images_found': 0, 'gts_found': 0, 'images_copied': 0, 'gts_copied': 0,
               'missing_images': [], 'missing_gts': []}

    for i, fn in enumerate(items, start=1):
        img_src, _ = find_file(src_root, image_dirs, fn)
        if img_src:
            summary['images_found'] += 1
            dst = os.path.join(img_out, fn)
            if dry_run:
                if show_each:
                    print(f"[DRY] IMG: {img_src} -> {dst}")
            else:
                if not os.path.exists(dst) or overwrite:
                    shutil.copy2(img_src, dst)
                    summary['images_copied'] += 1
                    if show_each:
                        print(f"IMG copied: {img_src} -> {dst}")
        else:
            summary['missing_images'].append(fn)
            if show_each:
                print(f"MISSING IMG: {fn}")

        gt_name = fn + '.txt'
        gt_src, _ = find_file(src_root, gt_dirs, gt_name)
        if gt_src:
            summary['gts_found'] += 1
            dstgt = os.path.join(gt_out, gt_name)
            if dry_run:
                if show_each:
                    print(f"[DRY] GT: {gt_src} -> {dstgt}")
            else:
                if not os.path.exists(dstgt) or overwrite:
                    shutil.copy2(gt_src, dstgt)
                    summary['gts_copied'] += 1
                    if show_each:
                        print(f"GT copied: {gt_src} -> {dstgt}")
        else:
            summary['missing_gts'].append(gt_name)
            if show_each:
                print(f"MISSING GT: {gt_name}")

        # no per-file verbose by default; keep function quiet unless --show-each is used

    return summary


def parse_args():
    src_root = 'datasets/COO'
    out_root = 'datasets/mini_COO'

    p = argparse.ArgumentParser(description='Create reduced COO lists and optionally copy image/GT files')
    p.add_argument('--mode', choices=['head', 'shuffle'], default='shuffle', help='Sampling mode')
    p.add_argument('--seed', type=int, default=0, help='Seed for shuffle mode')
    p.add_argument('--train', type=int, default=140, help='Number of train items')
    p.add_argument('--valid', type=int, default=30, help='Number of valid items')
    p.add_argument('--test', type=int, default=30, help='Number of test items')
    p.add_argument('--src-root', default=src_root, help='Root where original lists and images/gts live')
    p.add_argument('--src-train-list', default=None, help='Path to train_list.txt (default: <src-root>/train_list.txt)')
    p.add_argument('--src-valid-list', default=None, help='Path to valid_list.txt (default: <src-root>/valid_list.txt)')
    p.add_argument('--src-test-list', default=None, help='Path to test_list.txt (default: <src-root>/test_list.txt)')
    p.add_argument('--out-root', default=out_root, help='Output root to write lists and copied files')
    p.add_argument('--copy', action='store_true', help='Actually copy images/GTs; if omitted only lists are written (or dry-run)')
    p.add_argument('--dry-run', action='store_true', help='Do not perform any copying, just show what would be done')
    # make overwrite the default; provide --no-overwrite to disable
    p.add_argument('--no-overwrite', dest='overwrite', action='store_false', help='Do not overwrite existing files when copying')
    p.set_defaults(overwrite=True)
    p.add_argument('--image-dirs', nargs='+', default=['test_images', 'train_images', 'valid_images'],
                   help='Image candidate dirs under src-root (searched in order)')
    p.add_argument('--gt-dirs', nargs='+', default=['test_gts', 'train_gts', 'valid_gts'],
                   help='GT candidate dirs under src-root (searched in order)')
    # verbose flag removed; use --show-each for per-file output
    p.add_argument('--show-each', action='store_true', help='Show each file-level action (very verbose)')
    return p.parse_args()


def main():
    args = parse_args()

    # determine source list paths
    train_list_src = args.src_train_list or os.path.join(args.src_root, 'train_list.txt')
    valid_list_src = args.src_valid_list or os.path.join(args.src_root, 'valid_list.txt')
    test_list_src = args.src_test_list or os.path.join(args.src_root, 'test_list.txt')

    # read available lists
    train_items = read_list(train_list_src) if os.path.isfile(train_list_src) else []
    valid_items = read_list(valid_list_src) if os.path.isfile(valid_list_src) else []
    test_items = read_list(test_list_src) if os.path.isfile(test_list_src) else []

    # source list sizes (quiet by default)

    # sample
    sampled_train = sample_items(train_items, args.train, args.mode, args.seed)
    sampled_valid = sample_items(valid_items, args.valid, args.mode, args.seed)
    sampled_test = sample_items(test_items, args.test, args.mode, args.seed)

    # write lists
    out_train_list = os.path.join(args.out_root, 'train_list.txt')
    out_valid_list = os.path.join(args.out_root, 'valid_list.txt')
    out_test_list = os.path.join(args.out_root, 'test_list.txt')

    if args.dry_run:
        print(f"Will write {len(sampled_train)} train, {len(sampled_valid)} valid, {len(sampled_test)} test to {args.out_root}")

    if not args.dry_run:
        write_list(out_train_list, sampled_train)
        write_list(out_valid_list, sampled_valid)
        write_list(out_test_list, sampled_test)

    # copy step (if requested)
    copy_summary = {}
    if args.copy or args.dry_run:
        print("Starting copy (or dry-run of copy)...")
        # Only copy files when args.copy==True and not in dry-run mode
        actually_copy = args.copy and not args.dry_run
        simulated = args.dry_run and not actually_copy

        s = copy_files_for_list(sampled_train, args.src_root, args.out_root, 'train', args.image_dirs, args.gt_dirs,
                                dry_run=simulated, overwrite=args.overwrite,
                                show_each=args.show_each)
        copy_summary['train'] = s

        s = copy_files_for_list(sampled_valid, args.src_root, args.out_root, 'valid', args.image_dirs, args.gt_dirs,
                                dry_run=simulated, overwrite=args.overwrite,
                                show_each=args.show_each)
        copy_summary['valid'] = s

        s = copy_files_for_list(sampled_test, args.src_root, args.out_root, 'test', args.image_dirs, args.gt_dirs,
                                dry_run=simulated, overwrite=args.overwrite,
                                show_each=args.show_each)
        copy_summary['test'] = s

    # final summary
    print('\n=== SUMMARY ===')
    print(f"Lists written to: {args.out_root} (dry-run={args.dry_run})")
    print(f"Train: {len(sampled_train)}  Valid: {len(sampled_valid)}  Test: {len(sampled_test)}")
    if copy_summary:
        for k, v in copy_summary.items():
            print(f"--- {k} --- items={v['items']} img_found={v['images_found']} img_copied={v['images_copied']} gt_found={v['gts_found']} gt_copied={v['gts_copied']} missing_img={len(v['missing_images'])} missing_gt={len(v['missing_gts'])}")


if __name__ == '__main__':
    main()
