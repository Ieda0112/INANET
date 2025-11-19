import os
import datetime

import torch
from tqdm import tqdm

from experiment import Experiment
from data.data_loader import DistributedSampler


class Trainer:
    def __init__(self, experiment: Experiment):
        self.init_device()

        self.experiment = experiment
        self.structure = experiment.structure
        self.logger = experiment.logger
        self.model_saver = experiment.train.model_saver

        # FIXME: Hack the save model path into logger path
        self.model_saver.dir_path = self.logger.save_dir(
            self.model_saver.dir_path)
        self.current_lr = 0

        self.total = 0

        # --- output.log形式のログファイルパス生成 ---
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join('outputs', 'processing_logs')
        log_dir = os.path.join('outputs', 'processing_logs')
        os.makedirs(log_dir, exist_ok=True)
        self.full_log_path = os.path.join(log_dir, f'training_{timestamp}.log')

    def init_device(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def init_model(self):
        model = self.structure.builder.build(
            self.device, self.experiment.distributed, self.experiment.local_rank)
        return model

    def update_learning_rate(self, optimizer, epoch, step):
        lr = self.experiment.train.scheduler.learning_rate.get_learning_rate(
            epoch, step)

        for group in optimizer.param_groups:
            group['lr'] = lr
        self.current_lr = lr

    def train(self):
        self.logger.report_time('Start')
        self.logger.args(self.experiment)
        model = self.init_model()
        train_data_loader = self.experiment.train.data_loader
        if self.experiment.validation:
            validation_loaders = self.experiment.validation.data_loaders

        total_params = sum(p.numel() for p in model.parameters())
        print('参数量为：')
        print(f'{total_params:,} total parameters.')

        self.steps = 0
        if self.experiment.train.checkpoint:
            self.experiment.train.checkpoint.restore_model(
                model, self.device, self.logger)
            epoch, iter_delta = self.experiment.train.checkpoint.restore_counter()
            self.steps = epoch * self.total + iter_delta

        # Init start epoch and iter
        optimizer = self.experiment.train.scheduler.create_optimizer(
            model.parameters())

        self.logger.report_time('Init')

        model.train()
        with open(self.full_log_path, 'a') as f:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            f.write(f'Training {timestamp}\n')
        with open(self.full_log_path, 'a') as f:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            f.write(f'Training {timestamp}\n')
        while True:
            # --- エポック開始 ---

            # --- エポック開始 ---

            self.logger.info('Training epoch ' + str(epoch))
            self.logger.epoch(epoch)
            self.total = len(train_data_loader)

            # --- validation lossもエポックごとに記録 ---
            # # 確認のために挿入しているだけ
            # if self.experiment.validation:
            #     try:
            #         val_metrics, avg_valid_loss = self.validate(validation_loaders, model, epoch, self.steps)
            #         metric_str = ', '.join([f'{k}:{v.avg:.6f}' for k, v in val_metrics.items()])
            #         self.logger.info('Epoch %6d avg_valid_loss: %.6f validation: %s' % (epoch, avg_valid_loss, metric_str))
            #         with open(self.full_log_path, 'a') as f:
            #             metric_tab = '\t'.join([f'{k}:{v.avg:.6f}' for k, v in val_metrics.items()])
            #             f.write(f'VALID_EPOCH_END\tepoch:{epoch}\tstep:{self.steps}\tavg_valid_loss:{avg_valid_loss:.6f}\t{metric_tab}\n')
            #     except Exception as e:
            #         self.logger.info('Epoch-end validation failed: %s' % str(e))

            epoch_train_losses = []
            for batch in train_data_loader:
                self.update_learning_rate(optimizer, epoch, self.steps)

                self.logger.report_time("Data loading")

                # if self.experiment.validation and\
                #         self.steps % self.experiment.validation.interval == 0 and\
                #         self.steps > self.experiment.validation.exempt:
                #     self.validate(validation_loaders, model, epoch, self.steps)
                # if self.experiment.validation and\
                #         self.steps % self.experiment.validation.interval == 0 and\
                #         self.steps > self.experiment.validation.exempt:
                #     self.validate(validation_loaders, model, epoch, self.steps)
                self.logger.report_time('Validating ')
                if self.logger.verbose:
                    torch.cuda.synchronize()

                loss = self.train_step(model, optimizer, batch,
                                      epoch=epoch, step=self.steps)
                epoch_train_losses.append(loss)
                loss = self.train_step(model, optimizer, batch,
                                      epoch=epoch, step=self.steps)
                epoch_train_losses.append(loss)

                if self.logger.verbose:
                    torch.cuda.synchronize()
                self.logger.report_time('Forwarding ')

                self.model_saver.maybe_save_model(
                    model, epoch, self.steps, self.logger)

                self.steps += 1
                self.logger.report_eta(self.steps, self.total, epoch)

            # --- エポック終了時にtrain loss/metricsを記録・保存 ---
            # 最後のステップの情報をログに記録
            if hasattr(self, 'last_step_info'):
                info = self.last_step_info
                if info['is_dict']:
                    line = '\t'.join(info['line'])
                    log_info = '\t'.join(['step:{:6d}', 'epoch:{:3d}', '{}', 'lr:{:.4f}']).format(info['step'], epoch, line, info['lr'])
                    self.logger.info(log_info)
                    with open(self.full_log_path, 'a') as f:
                        f.write(f'TRAIN\tstep:{info["step"]}\tepoch:{epoch}\t{line}\tlr:{info["lr"]:.6f}\n')
                else:
                    self.logger.info('step: %6d, epoch: %3d, loss: %.6f, lr: %f' % (
                        info['step'], epoch, info['loss'], info['lr']))
                    with open(self.full_log_path, 'a') as f:
                        f.write(f'TRAIN\tstep:{info["step"]}\tepoch:{epoch}\tloss:{info["loss"]:.6f}\tlr:{info["lr"]:.6f}\n')
                
                self.logger.add_scalar('loss', info['loss_tensor'], info['step'])
                self.logger.add_scalar('learning_rate', info['lr'], info['step'])
                for name, metric in info['metrics'].items():
                    self.logger.add_scalar(name, metric, info['step'])
                    self.logger.info('%s: %6f' % (name, metric))

            # エポック平均train lossを記録
            print(f"len of epoch train loss:{len(epoch_train_losses)}")
            if len(epoch_train_losses) > 0:
                avg_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            else:
                avg_loss = float('nan')
            print(f"avg train loss:{avg_loss}")
            self.logger.info('Epoch %6d avg_train_loss: %.6f' % (epoch, avg_loss))
            # 最後のステップの情報をログに記録
            if hasattr(self, 'last_step_info'):
                info = self.last_step_info
                if info['is_dict']:
                    line = '\t'.join(info['line'])
                    log_info = '\t'.join(['step:{:6d}', 'epoch:{:3d}', '{}', 'lr:{:.4f}']).format(info['step'], epoch, line, info['lr'])
                    self.logger.info(log_info)
                    with open(self.full_log_path, 'a') as f:
                        f.write(f'TRAIN\tstep:{info["step"]}\tepoch:{epoch}\t{line}\tlr:{info["lr"]:.6f}\n')
                else:
                    self.logger.info('step: %6d, epoch: %3d, loss: %.6f, lr: %f' % (
                        info['step'], epoch, info['loss'], info['lr']))
                    with open(self.full_log_path, 'a') as f:
                        f.write(f'TRAIN\tstep:{info["step"]}\tepoch:{epoch}\tloss:{info["loss"]:.6f}\tlr:{info["lr"]:.6f}\n')
                
                self.logger.add_scalar('loss', info['loss_tensor'], info['step'])
                self.logger.add_scalar('learning_rate', info['lr'], info['step'])
                for name, metric in info['metrics'].items():
                    self.logger.add_scalar(name, metric, info['step'])
                    self.logger.info('%s: %6f' % (name, metric))

            # エポック平均train lossを記録
            print(f"len of epoch train loss:{len(epoch_train_losses)}")
            if len(epoch_train_losses) > 0:
                avg_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            else:
                avg_loss = float('nan')
            print(f"avg train loss:{avg_loss}")
            self.logger.info('Epoch %6d avg_train_loss: %.6f' % (epoch, avg_loss))
            with open(self.full_log_path, 'a') as f:
                f.write(f'TRAIN_EPOCH_END\tepoch:{epoch}\tavg_train_loss:{avg_loss:.6f}\n')
            
                f.write(f'TRAIN_EPOCH_END\tepoch:{epoch}\tavg_train_loss:{avg_loss:.6f}\n')
            
            # --- validation lossもエポックごとに記録 ---
            if self.experiment.validation:
                try:
                    val_metrics, avg_valid_loss = self.validate(validation_loaders, model, epoch, self.steps)
                    metric_str = ', '.join([f'{k}:{v.avg:.6f}' for k, v in val_metrics.items()])
                    self.logger.info('Epoch %6d avg_valid_loss: %.6f validation: %s' % (epoch, avg_valid_loss, metric_str))
                    with open(self.full_log_path, 'a') as f:
                        metric_tab = '\t'.join([f'{k}:{v.avg:.6f}' for k, v in val_metrics.items()])
                        f.write(f'VALID_EPOCH_END\tepoch:{epoch}\tstep:{self.steps}\tavg_valid_loss:{avg_valid_loss:.6f}\t{metric_tab}\n')
                except Exception as e:
                    self.logger.info('Epoch-end validation failed: %s' % str(e))
                try:
                    val_metrics, avg_valid_loss = self.validate(validation_loaders, model, epoch, self.steps)
                    metric_str = ', '.join([f'{k}:{v.avg:.6f}' for k, v in val_metrics.items()])
                    self.logger.info('Epoch %6d avg_valid_loss: %.6f validation: %s' % (epoch, avg_valid_loss, metric_str))
                    with open(self.full_log_path, 'a') as f:
                        metric_tab = '\t'.join([f'{k}:{v.avg:.6f}' for k, v in val_metrics.items()])
                        f.write(f'VALID_EPOCH_END\tepoch:{epoch}\tstep:{self.steps}\tavg_valid_loss:{avg_valid_loss:.6f}\t{metric_tab}\n')
                except Exception as e:
                    self.logger.info('Epoch-end validation failed: %s' % str(e))

            epoch += 1
            if epoch > self.experiment.train.epochs:
                self.model_saver.save_checkpoint(model, 'final')
                if self.experiment.validation:
                    val_metrics, avg_valid_loss = self.validate(validation_loaders, model, epoch, self.steps)
                    val_metrics, avg_valid_loss = self.validate(validation_loaders, model, epoch, self.steps)
                self.logger.info('Training done')
                break
            iter_delta = 0
            # --- エポック終了 ---
            # --- エポック終了 ---


    def train_step(self, model, optimizer, batch, epoch, step, **kwards):
        optimizer.zero_grad()

        results = model.forward(batch, training=True)
        if len(results) == 2:
            l, pred = results
            metrics = {}
        elif len(results) == 3:
            l, pred, metrics = results
        else:
            for res in results:
                print(f"len of result:{len(res)}")
                print(f"result:{res}")

        if isinstance(l, dict):
            line = []
            loss = torch.tensor(0.).cuda()
            for key, l_val in l.items():
                loss += l_val.mean()
                line.append('loss_{0}:{1:.4f}'.format(key, l_val.mean()))
        else:
            loss = l.mean()
        loss.backward()
        optimizer.step()

        # # エポック終了時のログ用に最後のステップ情報を保存
        # if isinstance(l, dict):
        #     line_list = []
        #     for key, l_val in l.items():
        #         line_list.append('loss_{0}:{1:.4f}'.format(key, l_val.mean()))
        #     self.last_step_info = {
        #         'step': step,
        #         'loss': loss.item(),
        #         'loss_tensor': loss,
        #         'lr': self.current_lr,
        #         'is_dict': True,
        #         'line': line_list,
        #         'metrics': {name: metric.mean().item() for name, metric in metrics.items()}
        #     }
        # else:
        #     self.last_step_info = {
        #         'step': step,
        #         'loss': loss.item(),
        #         'loss_tensor': loss,
        #         'lr': self.current_lr,
        #         'is_dict': False,
        #         'metrics': {name: metric.mean().item() for name, metric in metrics.items()}
        #     }

        # if step % self.experiment.logger.log_interval == 0:
        #     if isinstance(l, dict):
        #         line = '\t'.join(line)
        #         log_info = '\t'.join(['step:{:6d}', 'epoch:{:3d}', '{}', 'lr:{:.4f}']).format(step, epoch, line, self.current_lr)
        #         self.logger.info(log_info)
        #         # --- train loss/metricsをログファイルに追記 ---
        #         with open(self.full_log_path, 'a') as f:
        #             f.write(f'TRAIN\tstep:{step}\tepoch:{epoch}\t{line}\tlr:{self.current_lr:.6f}\n')
        #     else:
        #         self.logger.info('step: %6d, epoch: %3d, loss: %.6f, lr: %f' % (
        #             step, epoch, loss.item(), self.current_lr))
        #         with open(self.full_log_path, 'a') as f:
        #             f.write(f'TRAIN\tstep:{step}\tepoch:{epoch}\tloss:{loss.item():.6f}\tlr:{self.current_lr:.6f}\n')
        #     # --- train loss保存タイミングでvalidation lossも保存 ---
        #     # if self.experiment.validation:
        #     #     validation_loaders = self.experiment.validation.data_loaders
        #     #     self.validate(validation_loaders, model, epoch, step)
        #     self.logger.add_scalar('loss', loss, step)
        #     self.logger.add_scalar('learning_rate', self.current_lr, step)
        #     for name, metric in metrics.items():
        #         self.logger.add_scalar(name, metric.mean(), step)
        #         self.logger.info('%s: %6f' % (name, metric.mean()))
        #     self.logger.report_time('Logging')
        return loss.item()

    def validate(self, validation_loaders, model, epoch, step):
        # print("validate: entered")
        all_matircs = {}
        all_valid_losses = []
        model.eval()
        # print(f"validation_loaders.item: {validation_loaders.item}")
        for name, loader in validation_loaders.items():
            print(f"Validating {name} dataset...")
            if self.experiment.validation.visualize:
                metrics, vis_images, valid_losses = self.validate_step(loader, model, True)
                # print(f"valid_losses: {valid_losses}")
                self.logger.images(
                    os.path.join('vis', name), vis_images, step)
            else:
                metrics, vis_images, valid_losses = self.validate_step(loader, model, False)
                # print(f"valid_losses:")
                # for valid_loss in valid_losses:
                #     print(f"{valid_loss}")
            all_valid_losses.extend(valid_losses)
            for _key, metric in metrics.items():
                key = name + '/' + _key
                if key in all_matircs:
                    all_matircs[key].update(metric.val, metric.count)
                else:
                    all_matircs[key] = metric

        for key, metric in all_matircs.items():
            self.logger.info('%s : %f (%d)' % (key, metric.avg, metric.count))
        
        # --- validation lossを計算して表示 ---
        # print(f"all_valid_losses: {all_valid_losses}")
        if len(all_valid_losses) > 0:
            avg_valid_loss = sum(all_valid_losses) / len(all_valid_losses)
            self.logger.info('Validation avg loss: %.6f' % avg_valid_loss)
        else:
            avg_valid_loss = float('nan')
        
        model.train()
        return all_matircs, avg_valid_loss

    def validate_step(self, data_loader, model, visualize=False):
        # print("validate_step: entered")
        raw_metrics = []
        vis_images = dict()
        valid_losses = []
        # Disable grad to reduce memory use during validation
        with torch.no_grad():
            for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
                pred = model.forward(batch, training=False)

                # 損失計算（training=Trueと同じロジックで計算、ただしバックプロパゲーションはしない）
                try:
                    # model.forwardの結果から損失を取得
                    # 通常、training=Falseでは損失は返されないので、training=Trueで損失を計算する
                    results_with_loss = model.forward(batch, training=True)
                    # if isinstance(results_with_loss, (tuple, list)):
                    #     if len(results_with_loss) >= 2:
                    #         l = results_with_loss[0]
                    #         if isinstance(l, dict):
                    #             loss_val = sum([l_val.mean().item() for l_val in l.values()])
                    #         else:
                    #             loss_val = l.mean().item()
                    #         print("valid_losses apend")
                    #         valid_losses.append(loss_val)
                    #         # 
                    if isinstance(results_with_loss, torch.Tensor):
                        # print("results_with_loss is Tensor")
                        loss_val = results_with_loss.mean().item()
                        # print("valid_losses append:", loss_val)
                        valid_losses.append(loss_val)
                    else:
                        print("results_with_loss is not tuple or list")
                except Exception as e:
                    print("validate_step error:", str(e))
                    # 損失計算に失敗した場合は空リストを返す
                    # return {}, {}
                    return {}, {}, []

                output = self.structure.representer.represent(batch, pred)
                raw_metric, interested = self.structure.measurer.validate_measure(batch, output), None
                raw_metrics.append(raw_metric)

                if visualize and self.structure.visualizer:
                    vis_image = self.structure.visualizer.visualize(
                        batch, output, interested)
                    vis_images.update(vis_image)
        metrics = self.structure.measurer.gather_measure(
            raw_metrics, self.logger)
        # print("metrics:", metrics)
        # print("visualization images:", vis_images)
        # return metrics, vis_images
        return metrics, vis_images, valid_losses

    def to_np(self, x):
        return x.cpu().data.numpy()

