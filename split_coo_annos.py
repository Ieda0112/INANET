import os
import random

src_dir = "COO_annos_4_INANET"  # 振り分け元ディレクトリ
base_split_dir = "COO_annos_split"  # 振り分け先ディレクトリ
train_dir = os.path.join(base_split_dir, "train")
valid_dir = os.path.join(base_split_dir, "valid")
test_dir = os.path.join(base_split_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

files = [f for f in os.listdir(src_dir) if f.endswith('.txt')]
files.sort()
random.seed(42)
random.shuffle(files)

n = len(files)
n_train = int(n * 0.7)
n_valid = int(n * 0.15)
n_test = n - n_train - n_valid

train_files = files[:n_train]
valid_files = files[n_train:n_train+n_valid]
test_files = files[n_train+n_valid:]

for fname in train_files:
    os.rename(os.path.join(src_dir, fname), os.path.join(train_dir, fname))
for fname in valid_files:
    os.rename(os.path.join(src_dir, fname), os.path.join(valid_dir, fname))
for fname in test_files:
    os.rename(os.path.join(src_dir, fname), os.path.join(test_dir, fname))

with open(os.path.join(base_split_dir, "train_files.txt"), "w") as f:
    for fname in train_files:
        f.write(fname + "\n")
with open(os.path.join(base_split_dir, "valid_files.txt"), "w") as f:
    for fname in valid_files:
        f.write(fname + "\n")
with open(os.path.join(base_split_dir, "test_files.txt"), "w") as f:
    for fname in test_files:
        f.write(fname + "\n")
print(f"train: {len(train_files)}, valid: {len(valid_files)}, test: {len(test_files)}")

# # 元に戻すコード
# src_dirs = [train_dir, valid_dir, test_dir]
# dst_dir = "COO_annos_4_INANET"
# os.makedirs(dst_dir, exist_ok=True)

# for src in src_dirs:
#     files = [f for f in os.listdir(src) if f.endswith('.txt')]
#     for fname in files:
#         os.rename(os.path.join(src, fname), os.path.join(dst_dir, fname))
# print(f"split内のファイルを {dst_dir} に戻しました")
