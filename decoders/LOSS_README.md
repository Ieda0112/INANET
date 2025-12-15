# 新しい損失関数の実装

このディレクトリには、Balanced Cross Entropyの代わりに使用できる新しい損失関数が実装されています。

## 実装された損失関数

### 1. Focal Loss (`L1FocalLoss`)
- **ファイル**: `decoders/focal_loss.py` の `FocalLoss` クラス
- **論文**: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
- **特徴**: 
  - クラス不均衡問題に対処
  - 簡単な例のロスを下げ、難しい例に焦点を当てる
  - パラメータ:
    - `alpha`: ポジティブ/ネガティブのバランス係数 (デフォルト: 0.25)
    - `gamma`: フォーカルパラメータ (デフォルト: 2.0)
    - `focal_scale`: 損失のスケール係数 (デフォルト: 5)

### 2. Weighted Cross Entropy Loss (`L1WeightedCELoss`)
- **ファイル**: `decoders/focal_loss.py` の `WeightedCrossEntropyLoss` クラス
- **特徴**:
  - クラス不均衡を重み付けで対処
  - 動的または固定の重み付けをサポート
  - パラメータ:
    - `pos_weight`: ポジティブクラスの重み (None で動的計算、デフォルト: None)
    - `wce_scale`: 損失のスケール係数 (デフォルト: 5)

## 使用方法

### Focal Lossの使用

設定ファイル: `experiments/seg_detector/totaltext_resnet50_deform_thre_INA_focal.yaml`

```bash
python train.py experiments/seg_detector/totaltext_resnet50_deform_thre_INA_focal.yaml
```

YAMLファイル内の設定:
```yaml
loss_class: L1FocalLoss
loss_args:
    eps: 1.0e-6
    l1_scale: 10
    focal_scale: 5
    alpha: 0.25      # アルファ値（0-1）
    gamma: 2.0       # ガンマ値（推奨: 2.0）
```

### Weighted Cross Entropyの使用

設定ファイル: `experiments/seg_detector/totaltext_resnet50_deform_thre_INA_weighted_ce.yaml`

```bash
python train.py experiments/seg_detector/totaltext_resnet50_deform_thre_INA_weighted_ce.yaml
```

YAMLファイル内の設定:
```yaml
loss_class: L1WeightedCELoss
loss_args:
    eps: 1.0e-6
    l1_scale: 10
    wce_scale: 5
    pos_weight: null  # null=動的計算、または数値を指定
```

## パラメータのチューニング

### Focal Loss
- **alpha**: 
  - 0.25 が一般的
  - ポジティブサンプルが少ない場合は大きく設定
- **gamma**: 
  - 2.0 が推奨
  - 大きいほど簡単な例の影響を減らす
  - 範囲: 0.5〜5.0

### Weighted Cross Entropy
- **pos_weight**: 
  - `null` で自動計算（ネガティブ数 / ポジティブ数）
  - 手動設定の場合、1.0より大きい値を推奨

## 損失関数の比較

| 損失関数 | 利点 | 欠点 |
|---------|------|------|
| Balanced CE | シンプル、安定 | 簡単/難しい例を区別しない |
| Focal Loss | 難しい例に焦点、クラス不均衡に強い | ハイパーパラメータ調整が必要 |
| Weighted CE | シンプル、クラス不均衡に対応 | 動的重み付けの場合バッチに依存 |

## トレーニングのモニタリング

損失のメトリクス名:
- Focal Loss使用時: `focal_loss`, `thresh_loss`, `l1_loss`
- Weighted CE使用時: `wce_loss`, `thresh_loss`, `l1_loss`

ログから確認:
```python
# 例: outputs/training_graph/ に保存されるグラフで確認
```

## 実装の詳細

両方の損失関数は以下の特徴を持っています:
1. **Hard Negative Mining**: ネガティブサンプルを選択的に使用
2. **L1 Loss**: 閾値マップの学習用
3. **Dice Loss**: 閾値のバイナリ予測用

全体の損失:
```
Total Loss = Dice Loss + l1_scale * L1 Loss + (focal_scale or wce_scale) * (Focal or Weighted CE Loss)
```
