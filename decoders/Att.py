import torch
import torch.nn as nn

class Att(nn.Module):
    def __init__(self, in_channels, factor=32, bias=False):
        """
        in_channels: 入力チャネル数
        factor: 圧縮比率（中間チャネル = in_channels // factor）
        bias: 各畳み込み層にバイアスを持たせるかどうか
        """
        super(Att, self).__init__()
        # 中間チャネル数を決定
        mid_channels = max(1, in_channels // factor)
        
        # グローバル平均プーリング
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 圧縮 → 拡張の 1×1 畳み込み
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x の形: (batch, in_channels, H, W)
        # グローバル平均プーリング → (batch, in_channels, 1, 1)
        y = self.avg_pool(x)
        y = self.conv1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.sigmoid(y)
        # 重みを乗算して出力
        return x * y
