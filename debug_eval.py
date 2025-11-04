#!/usr/bin/env python3
"""
評価時の予測値を可視化するデバッグスクリプト
"""
import torch
import cv2
import numpy as np
from eval import Eval
from experiment import Experiment
from concern.config import Configurable, Config

def debug_eval():
    # 設定を読み込み
    conf = Config()
    experiment_args = conf.compile(conf.load('experiments/seg_detector/totaltext_resnet50_deform_thre_INA.yaml'))['Experiment']
    cmd = {
        'resume': 'workspace/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss/model/model_epoch_315_minibatch_66000',
        'result_dir': './debug_results/',
        'batch_size': 1,
        'polygon': True,
        'box_thresh': 0.01,
        'debug': False,
        'verbose': True,
    }
    experiment_args.update(cmd=cmd)
    experiment = Configurable.construct_class_from_config(experiment_args)
    
    # 初期化
    evaluator = Eval(experiment, experiment_args, cmd=cmd, verbose=True)
    evaluator.init_torch_tensor()
    model = evaluator.init_model()
    evaluator.resume(model, cmd['resume'])
    
    data_loader = list(evaluator.data_loaders.values())[0]
    model.eval()
    
    # 最初のバッチを処理
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            print("=" * 70)
            print(f"DEBUGGING BATCH {batch_idx}")
            print("=" * 70)
            
            print(f"\n[BATCH INFO]")
            print(f"  Image shape: {batch['image'].shape}")
            print(f"  Polygons shape: {batch['polygons'].shape if 'polygons' in batch else 'N/A'}")
            print(f"  Ignore tags shape: {batch['ignore_tags'].shape if 'ignore_tags' in batch else 'N/A'}")
            print(f"  Filename: {batch.get('filename', 'N/A')}")
            
            # モデル出力
            pred = model.forward(batch, training=False)
            print(f"\n[MODEL OUTPUT]")
            print(f"  Type: {type(pred)}")
            if isinstance(pred, dict):
                print(f"  Keys: {pred.keys()}")
                for k, v in pred.items():
                    print(f"    {k}: shape={v.shape}, dtype={v.dtype}, min={v.min():.6f}, max={v.max():.6f}, mean={v.mean():.6f}")
            else:
                print(f"  Shape: {pred.shape}, dtype: {pred.dtype}")
                print(f"  Min: {pred.min():.6f}, Max: {pred.max():.6f}, Mean: {pred.mean():.6f}")
            
            # representer処理
            print(f"\n[REPRESENTER CONFIG]")
            print(f"  dest: {evaluator.structure.representer.dest}")
            print(f"  thresh: {evaluator.structure.representer.thresh}")
            print(f"  box_thresh: {evaluator.structure.representer.box_thresh}")
            
            output = evaluator.structure.representer.represent(batch, pred, is_output_polygon=cmd['polygon'])
            boxes_batch, scores_batch = output
            
            print(f"\n[REPRESENTER OUTPUT]")
            for i, (boxes, scores) in enumerate(zip(boxes_batch, scores_batch)):
                print(f"  Batch {i}:")
                print(f"    Number of detected polygons: {len(boxes)}")
                if len(boxes) > 0:
                    boxes_arr = np.array(boxes)
                    print(f"    Boxes shape: {boxes_arr.shape}")
                    print(f"    Scores: min={np.min(scores):.6f}, max={np.max(scores):.6f}, mean={np.mean(scores):.6f}")
                    print(f"    First few scores: {scores[:min(5, len(scores))]}")
                else:
                    print(f"    ⚠️ No polygons detected!")
            
            # 検証メトリクス
            print(f"\n[VALIDATION METRICS]")
            raw_metric = evaluator.structure.measurer.validate_measure(
                batch, output, is_output_polygon=cmd['polygon'], box_thresh=cmd['box_thresh'])
            print(f"  Raw metric: {raw_metric}")
            if isinstance(raw_metric, list) and len(raw_metric) > 0:
                print(f"  Metric details: {raw_metric[0]}")
            
            if batch_idx >= 2:  # 最初の3バッチを処理
                break

if __name__ == '__main__':
    debug_eval()
