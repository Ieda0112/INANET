#!/usr/bin/env python3
"""
COO_2_total_text.pyとsplit_coo_annos.pyの機能を統合したスクリプト
prepare_coo_dataset.py

1) Convert XML annotations (COO format) → TotalText-style txt files (one per page):
   <title>_<index>.txt with lines: x0,y0,x1,y1,...,xn,yn,label

2) Optionally split generated txt files into train/valid/test (default 70/15/15)
   and write list files.

Usage examples:
    python prepare_coo_dataset.py --xml-dir /path/to/xmls --out-dir COO_annos_4_INANET
    python prepare_coo_dataset.py --xml-dir /path/to/xmls --out-dir COO_annos_4_INANET --split --split-dir datasets/COO --seed 42

Supports a --revert option to move files in split dirs back to source if needed.
"""

import argparse
import glob
import os
import shutil
import random
import xml.etree.ElementTree as ET


def convert_xml_folder(xml_dir, out_dir, ext='txt'):
    os.makedirs(out_dir, exist_ok=True)
    xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))
    converted = []
    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        book_title = root.get('title')
        pages = root.find('pages')
        if pages is None:
            continue
        for page in pages.findall('page'):
            index = page.get('index')
            onos = page.findall('onomatopoeia')
            if not onos:
                continue
            annos = []
            for ono in onos:
                points = []
                i = 0
                while True:
                    x = ono.get(f'x{i}')
                    y = ono.get(f'y{i}')
                    if x is None or y is None:
                        break
                    points.append(f"{x},{y}")
                    i += 1
                text = ono.text if ono.text is not None else "###"
                annos.append(','.join(points) + f",{text}")
            index_str = str(index).zfill(3)
            fname = f"{book_title}_{index_str}.{ext}"
            out_path = os.path.join(out_dir, fname)
            # write file
            with open(out_path, 'w', encoding='utf-8') as f:
                for line in annos:
                    f.write(line + "\n")
            converted.append(fname)
    return converted


def split_files(src_dir, split_dir, ratios=(0.7, 0.15, 0.15), seed=42):
    os.makedirs(split_dir, exist_ok=True)
    train_dir = os.path.join(split_dir, 'train')
    valid_dir = os.path.join(split_dir, 'valid')
    test_dir = os.path.join(split_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    files = [f for f in os.listdir(src_dir) if f.endswith('.txt')]
    files.sort()
    random.seed(seed)
    random.shuffle(files)

    n = len(files)
    n_train = int(n * ratios[0])
    n_valid = int(n * ratios[1])
    n_test = n - n_train - n_valid

    train_files = files[:n_train]
    valid_files = files[n_train:n_train + n_valid]
    test_files = files[n_train + n_valid:]

    for fname in train_files:
        shutil.move(os.path.join(src_dir, fname), os.path.join(train_dir, fname))
    for fname in valid_files:
        shutil.move(os.path.join(src_dir, fname), os.path.join(valid_dir, fname))
    for fname in test_files:
        shutil.move(os.path.join(src_dir, fname), os.path.join(test_dir, fname))

    # write lists
    with open(os.path.join(split_dir, 'train_list.txt'), 'w') as f:
        for fname in train_files:
            f.write(fname + '\n')
    with open(os.path.join(split_dir, 'valid_list.txt'), 'w') as f:
        for fname in valid_files:
            f.write(fname + '\n')
    with open(os.path.join(split_dir, 'test_list.txt'), 'w') as f:
        for fname in test_files:
            f.write(fname + '\n')

    return (train_files, valid_files, test_files)


def revert_split(split_dir, dst_dir):
    for part in ('train', 'valid', 'test'):
        src = os.path.join(split_dir, part)
        if not os.path.isdir(src):
            continue
        for fname in os.listdir(src):
            if fname.endswith('.txt'):
                shutil.move(os.path.join(src, fname), os.path.join(dst_dir, fname))
    print(f"Reverted split files to {dst_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml-dir', required=True, help='Folder containing XML files')
    parser.add_argument('--out-dir', default='COO_annos_4_INANET', help='Folder to write converted txts')
    parser.add_argument('--ext', default='txt', help='Extension for output annotation files')
    parser.add_argument('--split', action='store_true', help='Also split converted files into train/valid/test')
    parser.add_argument('--split-dir', default='datasets/COO', help='Destination base dir for split')
    parser.add_argument('--ratios', nargs=3, type=float, default=(0.7,0.15,0.15), help='Train/valid/test ratios')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffle')
    parser.add_argument('--revert', action='store_true', help='Move files in split back to out-dir')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files (if present)')
    args = parser.parse_args()

    if args.revert:
        revert_split(args.split_dir, args.out_dir)
        return

    converted = convert_xml_folder(args.xml_dir, args.out_dir, ext=args.ext)
    print(f"Converted {len(converted)} files to {args.out_dir}")

    if args.split:
        train_files, valid_files, test_files = split_files(args.out_dir, args.split_dir, ratios=tuple(args.ratios), seed=args.seed)
        print(f"Split done. train={len(train_files)}, valid={len(valid_files)}, test={len(test_files)}")


if __name__ == '__main__':
    main()
