#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
focal_loss.py - クラス不均衡に対処する損失関数の実装

このモジュールには、テキスト検出のようなクラス不均衡が大きいタスクに適した
2つの損失関数が含まれています：

1. Focal Loss
   - 難しいサンプルに焦点を当て、簡単なサンプルの影響を減らす
   - 物体検出やセグメンテーションで広く使われている
   - パラメータ調整により、学習の焦点を制御可能

2. Weighted Cross Entropy Loss
   - クラスごとに異なる重みを設定
   - シンプルで解釈しやすい
   - 動的重み付けにより、バッチごとに最適化

【使用例】
    # Focal Lossの使用
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    loss = loss_fn(predictions, targets, masks)
    
    # Weighted CEの使用
    loss_fn = WeightedCrossEntropyLoss(pos_weight=None)  # 動的重み付け
    loss = loss_fn(predictions, targets, masks)

【パラメータチューニングのガイド】
    Focal Loss:
        - alpha: 0.25（推奨）、ポジティブクラスをどれだけ重視するか調整
        - gamma: 2.0（推奨）、難しい例に強く焦点を当てる場合は3.0～5.0
    
    Weighted CE:
        - pos_weight: None（自動）が推奨、手動設定なら2.0～10.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    '''
    Focal Loss: クラス不均衡問題に対処するための損失関数
    参考文献: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    
    【概要】
    Focal Lossは、簡単に分類できるサンプル(easy examples)の損失を減らし、
    難しいサンプル(hard examples)に学習の焦点を当てることで、
    クラス不均衡問題を解決します。
    
    【数式】
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    where:
      - p_t: 正解クラスに対するモデルの予測確率
            - α_t: クラスバランス用の重み係数
      - γ: フォーカシングパラメータ（難易度による重み付け）
    
    Shape:
        - Input: :math:`(N, 1, H, W)` - ネットワークの予測値（sigmoidを通した後の確率値）
        - GT: :math:`(N, 1, H, W)` - 正解ラベル（0 or 1）
        - Mask: :math:`(N, H, W)` - 有効領域を示すマスク
        - Output: scalar - バッチ全体の平均損失
    
    Args:
     alpha: クラスバランス係数 (0～1)
         - gt=1（テキスト）でα, gt=0で(1-α)を掛ける
         - 0.25は背景が多いケースで有効（推奨）
         - Default: 0.25

     gamma: フォーカシングパラメータ（難易度による重み付け）
               - γ=0: 通常のCross Entropy Lossと同じ
               - γ=2: 簡単な例の損失を大幅に減らす（推奨値）
               - γが大きいほど、難しい例により焦点を当てる
               - 例: p_t=0.9（簡単）の場合、(1-0.9)^2 = 0.01倍に減衰
               - 例: p_t=0.5（難しい）の場合、(1-0.5)^2 = 0.25倍の減衰
               - Default: 2.0
                        
        eps: ゼロ除算を防ぐための微小値
             - Default: 1e-6
    '''

    def __init__(self, alpha=0.25, gamma=2.0, eps=1e-6):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # クラスバランス係数
        self.gamma = gamma  # フォーカシングパラメータ（難易度による重み付け）
        self.eps = eps  # ゼロ除算防止用の微小値

    def forward(self,
                pred: torch.Tensor,
                gt: torch.Tensor,
                mask: torch.Tensor,
                return_origin=False):
        '''
        Focal Lossの順伝播計算
        
        Args:
            pred: shape :math:`(N, 1, H, W)` - ネットワークの予測値（0～1の確率）
            gt: shape :math:`(N, 1, H, W)` - 正解ラベル（0 or 1）
            mask: shape :math:`(N, H, W)` - 有効ピクセルを示すマスク（1=有効, 0=無効）
            return_origin: Trueの場合、損失マップも返す
            
        Returns:
            balance_loss: スカラー値の損失
            または
            (balance_loss, focal_loss): return_origin=Trueの場合
        '''
        # ステップ1: 基本的なBinary Cross Entropy Lossの計算
        # BCE = -[y*log(p) + (1-y)*log(1-p)]
        bce_loss = F.binary_cross_entropy(
            pred, gt, reduction='none')[:, 0, :, :]
        
        # ステップ2: Focal Weightの計算 - (1 - p_t)^gamma
        # p_t: 正解クラスに対する予測確率
        #   - gt=1の場合: p_t = pred（テキストである確率）
        #   - gt=0の場合: p_t = 1-pred（背景である確率）
        # focal_weight: 簡単な例ほど小さくなる（損失の重みを減らす）
        #   - p_t=0.9（簡単）: (1-0.9)^2 = 0.01 → 損失を1%に減衰
        #   - p_t=0.5（難しい）: (1-0.5)^2 = 0.25 → 損失を25%に減衰
        p_t = torch.where(gt[:, 0, :, :] == 1, pred[:, 0, :, :], 1 - pred[:, 0, :, :])
        focal_weight = (1 - p_t) ** self.gamma
        
        # ステップ3: Focal Weightを適用
        focal_loss = focal_weight * bce_loss
        
        # ステップ4: Alphaバランス係数を適用
        alpha_t = torch.where(
            gt[:, 0, :, :] == 1,
            torch.full_like(bce_loss, self.alpha),
            torch.full_like(bce_loss, 1 - self.alpha)
        )
        focal_loss = focal_loss * alpha_t

        # ステップ5: マスクを適用して有効領域のみの損失を計算
        focal_loss = focal_loss * mask
        valid_pixels = mask.sum()
        
        # ステップ6: 平均損失を計算
        mean_loss = focal_loss.sum() / (valid_pixels + self.eps)

        if return_origin:
            return mean_loss, focal_loss
        return mean_loss


class WeightedCrossEntropyLoss(nn.Module):
    '''
    Weighted Cross Entropy Loss: 動的なクラス重み付けを行う損失関数
    
    【概要】
    クラス不均衡問題に対して、クラスごとに異なる重みを設定することで対処します。
    特にポジティブクラス（テキスト領域）が少ない場合に、ポジティブクラスの損失に
    より大きな重みを与えることで、バランスの取れた学習を実現します。
    
    【数式】
    Weighted BCE = -[y * w_pos * log(p) + (1-y) * log(1-p)]
    
    where:
      - y: 正解ラベル（0 or 1）
      - p: 予測確率
      - w_pos: ポジティブクラスの重み
    
    Shape:
        - Input: :math:`(N, 1, H, W)` - ネットワークの予測値（sigmoidを通した後の確率値）
        - GT: :math:`(N, 1, H, W)` - 正解ラベル（0 or 1）
        - Mask: :math:`(N, H, W)` - 有効領域を示すマスク
        - Output: scalar - バッチ全体の平均損失
    
    Args:
        pos_weight: ポジティブクラス（テキスト領域）の重み
                    - None: 動的に計算（neg_count / pos_count）
                           背景ピクセルが多いほど、テキスト領域の重みが大きくなる
                    - 固定値: 例えば2.0を設定すると、テキスト領域の損失を2倍にする
                    - 推奨: Noneで自動調整、または実験的に1.0～5.0の範囲で調整
                    - Default: None（動的計算）
                        
        eps: ゼロ除算やlog(0)を防ぐための微小値
             - Default: 1e-6
    '''

    def __init__(self, pos_weight=None, eps=1e-6):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.pos_weight = pos_weight  # ポジティブクラスの重み（Noneで動的計算）
        self.eps = eps  # ゼロ除算防止用の微小値

    def forward(self,
                pred: torch.Tensor,
                gt: torch.Tensor,
                mask: torch.Tensor,
                return_origin=False):
        '''
        Weighted Cross Entropy Lossの順伝播計算
        
        Args:
            pred: shape :math:`(N, 1, H, W)` - ネットワークの予測値（0～1の確率）
            gt: shape :math:`(N, 1, H, W)` - 正解ラベル（0 or 1）
            mask: shape :math:`(N, H, W)` - 有効ピクセルを示すマスク（1=有効, 0=無効）
            return_origin: Trueの場合、損失マップも返す
            
        Returns:
            mean_loss: スカラー値の損失
            または
            (mean_loss, loss): return_origin=Trueの場合
        '''
        # ステップ1: ポジティブ・ネガティブピクセルの数を計算
        positive = (gt[:, 0, :, :] * mask).float()
        negative = ((1 - gt[:, 0, :, :]) * mask).float()
        positive_count = positive.sum()
        negative_count = negative.sum()
        
        # ステップ2: ポジティブクラスの重み計算（動的または固定）
        if self.pos_weight is None:
            # 動的計算: pos_weight = ネガティブ総数 / ポジティブ総数
            # 例: 背景9000ピクセル、テキスト1000ピクセル → 重み=9.0
            # これにより、少数クラス（テキスト）の損失を大きくして、バランスを取る
            if positive_count > 0:
                pos_weight = negative_count / (positive_count + self.eps)
            else:
                pos_weight = 1.0
        else:
            # 固定値を使用
            pos_weight = self.pos_weight
        
        # ステップ3: Weighted Binary Cross Entropy Lossの計算
        # ポジティブクラス: -y * log(p) * pos_weight
        # ネガティブクラス: -(1-y) * log(1-p)
        pos_loss = -gt[:, 0, :, :] * torch.log(pred[:, 0, :, :] + self.eps) * pos_weight
        neg_loss = -(1 - gt[:, 0, :, :]) * torch.log(1 - pred[:, 0, :, :] + self.eps)
        loss = pos_loss + neg_loss
        
        # ステップ4: マスクを適用して有効領域のみの損失を計算
        loss = loss * mask
        valid_pixels = mask.sum()
        
        # ステップ5: 平均損失を計算
        mean_loss = loss.sum() / (valid_pixels + self.eps)

        if return_origin:
            return mean_loss, loss
        return mean_loss
