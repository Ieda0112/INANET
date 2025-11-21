import os
from datetime import datetime

import torch

from concern.config import Configurable, State
from concern.signal_monitor import SignalMonitor


class ModelSaver(Configurable):
    dir_path = State()
    save_interval = State(default=1000)
    signal_path = State()

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

        # BUG: signal path should not be global
        self.monitor = SignalMonitor(self.signal_path)
        # 最初の保存時に作る日時付きサブディレクトリを保持
        self._model_subdir = None

    def maybe_save_model(self, model, epoch, step, logger):
        if step % self.save_interval == 0 or self.monitor.get_signal() is not None:
            self.save_model(model, epoch, step)
            logger.report_time('Saving ')
            logger.iter(step)

    def save_model(self, model, epoch=None, step=None):
        if isinstance(model, dict):
            for name, net in model.items():
                checkpoint_name = self.make_checkpoint_name(name, epoch, step)
                self.save_checkpoint(net, checkpoint_name)
        else:
            checkpoint_name = self.make_checkpoint_name('model', epoch, step)
            self.save_checkpoint(model, checkpoint_name)

    def save_checkpoint(self, net, name):
        # 保存先は dir_path/<timestamp>/ にする（初回保存で作成）
        target_dir = self.model_dir()
        target_path = os.path.join(target_dir, name)
        torch.save(net.state_dict(), target_path)

        # 互換性のため、dir_path 以下にも名前付きのシンボリックリンクを作成（例: model/final -> model/<timestamp>/final）
        link_path = os.path.join(self.dir_path, name)
        try:
            if os.path.islink(link_path) or os.path.exists(link_path):
                os.remove(link_path)
            os.symlink(target_path, link_path)
        except Exception:
            # シンボリックリンク作成に失敗しても保存自体は成功しているので無視
            pass

    def model_dir(self):
        if getattr(self, '_model_subdir', None) is not None:
            return self._model_subdir

        # dir_path はすでに experiment の構成で model ディレクトリを指している想定
        os.makedirs(self.dir_path, exist_ok=True)
        time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._model_subdir = os.path.join(self.dir_path, time_str)
        os.makedirs(self._model_subdir, exist_ok=True)
        return self._model_subdir

    def make_checkpoint_name(self, name, epoch=None, step=None):
        if epoch is None or step is None:
            c_name = name + '_latest'
        else:
            c_name = '{}_epoch_{}_minibatch_{}'.format(name, epoch, step)
        return c_name
