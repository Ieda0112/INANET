import warnings
import numpy as np
import cv2
from shapely.geometry import Polygon
import pyclipper

from concern.config import Configurable, State

class MakeBorderMap(Configurable):
    shrink_ratio = State(default=0.4)
    thresh_min = State(default=0.3)
    thresh_max = State(default=0.7)

    def __init__(self, cmd={}, *args, **kwargs):
        self.load_all(cmd=cmd, **kwargs)
        warnings.simplefilter("ignore")


    def __call__(self, data, *args, **kwargs):
        image = data['image']
        polygons = data['polygons']
        ignore_tags = data['ignore_tags']

        canvas = np.zeros(image.shape[:2], dtype=np.float32)
        mask = np.zeros(image.shape[:2], dtype=np.float32)

        for i in range(len(polygons)):
            if ignore_tags[i]:
                continue
            self.draw_border_map(polygons[i], canvas, mask=mask)
        canvas = canvas * (self.thresh_max - self.thresh_min) + self.thresh_min
        data['thresh_map'] = canvas
        data['thresh_mask'] = mask
        return data

    def draw_border_map(self, polygon, canvas, mask):
        """
        テキスト領域の境界マップを描画する
        境界に近いほど値が大きく(1に近い)、遠いほど小さい(0に近い)マップを作成
        これはDBNetの閾値マップ生成に使われる
        
        Args:
            polygon: テキスト領域のポリゴン座標 [[x1,y1], [x2,y2], ...]
            canvas: 境界マップを描画する出力配列 (画像サイズと同じ)
            mask: テキスト領域のマスク (1=テキスト領域, 0=背景)
        """
        polygon = np.array(polygon)
        assert polygon.ndim == 2
        assert polygon.shape[1] == 2

        # --- 1. ポリゴンを少し膨張させる (shrink_ratioに基づく) ---
        polygon_shape = Polygon(polygon)
        # 膨張距離 = 面積×(1-shrink_ratio²) / 周長
        distance = polygon_shape.area * \
            (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
        
        # pyclipperでポリゴンをdistance分だけ外側に膨張
        subject = [tuple(l) for l in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND,
                        pyclipper.ET_CLOSEDPOLYGON)
        padded_polygon = np.array(padding.Execute(distance)[0])
        
        # 膨張したポリゴン領域をmaskに描画 (この領域内で損失を計算する)
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

        # --- 2. ポリゴンを囲む最小矩形領域を取得 ---
        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        # ポリゴン座標を矩形の左上を原点とする相対座標に変換
        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin

        # --- 3. 距離マップを作成 (各ピクセルからポリゴン境界までの距離) ---
        # メッシュグリッド作成: xs[i,j]=j, ys[i,j]=i
        xs = np.broadcast_to(
            np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(
            np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

        # 各ピクセルから各辺までの距離を計算
        distance_map = np.zeros(
            (polygon.shape[0], height, width), dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]  # 次の頂点 (最後の頂点は最初に戻る)
            # 各ピクセルから辺(i,j)までの距離を計算
            absolute_distance = self.distance(xs, ys, polygon[i], polygon[j])
            # 距離を正規化: distance(膨張距離)で割って[0,1]にクリップ
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        # 全ての辺からの距離の最小値を取る = ポリゴン境界までの最短距離
        distance_map = distance_map.min(axis=0)

        # --- 4. 画像全体のcanvasに距離マップを配置 ---
        # 矩形領域が画像の範囲内に収まるように調整
        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
        
        # 配列のインデックス範囲を計算
        src_ymin = ymin_valid - ymin
        src_ymax = ymax_valid - ymin + 1
        src_xmin = xmin_valid - xmin
        src_xmax = xmax_valid - xmin + 1
        
        # エッジケース: インデックス範囲が不正な場合はスキップ
        if ymax_valid < ymin_valid or xmax_valid < xmin_valid:
            return
        if src_ymax <= src_ymin or src_xmax <= src_xmin:
            return
        if src_ymin < 0 or src_xmin < 0:
            return
        if src_ymax > distance_map.shape[0] or src_xmax > distance_map.shape[1]:
            return
            
        # canvasに1-distance_mapを書き込む (境界に近いほど1、遠いほど0)
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
            1 - distance_map[src_ymin:src_ymax, src_xmin:src_xmax],
            canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])

    def distance(self, xs, ys, point_1, point_2):
        '''
        compute the distance from point to a line
        ys: coordinates in the first axis
        xs: coordinates in the second axis
        point_1, point_2: (x, y), the end of the line
        '''
        height, width = xs.shape[:2]
        square_distance_1 = np.square(
            xs - point_1[0]) + np.square(ys - point_1[1])
        square_distance_2 = np.square(
            xs - point_2[0]) + np.square(ys - point_2[1])
        square_distance = np.square(
            point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

        cosin = (square_distance - square_distance_1 - square_distance_2) / \
            (2 * np.sqrt(square_distance_1 * square_distance_2))
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_distance_1 * square_distance_2 *
                         square_sin / square_distance)

        result[cosin < 0] = np.sqrt(np.fmin(
            square_distance_1, square_distance_2))[cosin < 0]
        # self.extend_line(point_1, point_2, result)
        return result

    def extend_line(self, point_1, point_2, result):
        ex_point_1 = (int(round(point_1[0] + (point_1[0] - point_2[0]) * (1 + self.shrink_ratio))),
                      int(round(point_1[1] + (point_1[1] - point_2[1]) * (1 + self.shrink_ratio))))
        cv2.line(result, tuple(ex_point_1), tuple(point_1),
                 4096.0, 1, lineType=cv2.LINE_AA, shift=0)
        ex_point_2 = (int(round(point_2[0] + (point_2[0] - point_1[0]) * (1 + self.shrink_ratio))),
                      int(round(point_2[1] + (point_2[1] - point_1[1]) * (1 + self.shrink_ratio))))
        cv2.line(result, tuple(ex_point_2), tuple(point_2),
                 4096.0, 1, lineType=cv2.LINE_AA, shift=0)
        return ex_point_1, ex_point_2
