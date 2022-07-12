import time
import cv2
import torch
from tqdm import tqdm
import pytorch_warmup as warmup
import numpy as np
import random
import os

from clrnet.models.registry import build_net
from .registry import build_trainer, build_evaluator
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from clrnet.datasets import build_dataloader
from clrnet.utils.recorder import build_recorder
from clrnet.utils.net_utils import save_model, load_network, resume_network
from mmcv.parallel import MMDataParallel


class Runner(object):
    def __init__(self, cfg):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        self.cfg = cfg
        self.recorder = build_recorder(self.cfg)
        self.net = build_net(self.cfg)
        self.net = MMDataParallel(self.net,
                                  device_ids=range(self.cfg.gpus)).cuda()
        self.recorder.logger.info('Network: \n' + str(self.net))
        self.resume()
        self.optimizer = build_optimizer(self.cfg, self.net)
        self.scheduler = build_scheduler(self.cfg, self.optimizer)
        self.metric = 0.
        self.val_loader = None
        self.test_loader = None

    def to_cuda(self, batch):
        for k in batch:
            if not isinstance(batch[k], torch.Tensor):
                continue
            batch[k] = batch[k].cuda()
        return batch

    def resume(self):
        if not self.cfg.load_from and not self.cfg.finetune_from:
            return
        load_network(self.net, self.cfg.load_from, finetune_from=self.cfg.finetune_from, logger=self.recorder.logger)

    def train_epoch(self, epoch, train_loader):
        self.net.train()
        end = time.time()
        max_iter = len(train_loader)
        for i, data in enumerate(train_loader):
            if self.recorder.step >= self.cfg.total_iter:
                break
            date_time = time.time() - end
            self.recorder.step += 1
            """
            img_file_name_dic = data['meta'].data[0]
            is_first = True
            for dic in img_file_name_dic:
                file_list = dic['full_img_path'].split('/')[:-1]
                head = ''
                for f in file_list:
                    head += f + '/'
                img_5 = np.load(os.path.join(head, 'img_5.npy'))
                if is_first:
                    batch_img = img_5
                    is_first = False
                else:
                    batch_img = np.concatenate((batch_img, img_5), axis=0)
            batch_img = torch.tensor(batch_img).float()
            batch_img = batch_img.permute(0, 3, 1, 2)
            data['img'] = batch_img
            """
            data = self.to_cuda(data)
            output = self.net(data)
            self.optimizer.zero_grad()
            loss = output['loss'].sum()
            # print('# generator parameters:', sum(param.numel() for param in self.net.parameters()))
            """
            这里冻结backbone的网络参数进行训练
            """

            for param in self.net.module.backbone.parameters():
                param.requires_grad = False
            for param in self.net.module.neck.parameters():
                param.requires_grad = False
            


            




            # 其余代码一样
            loss.backward()
            self.optimizer.step()
            if not self.cfg.lr_update_by_epoch:
                self.scheduler.step()
            batch_time = time.time() - end
            end = time.time()
            self.recorder.update_loss_stats(output['loss_stats'])
            self.recorder.batch_time.update(batch_time)
            self.recorder.data_time.update(date_time)

            if i % self.cfg.log_interval == 0 or i == max_iter - 1:
                lr = self.optimizer.param_groups[0]['lr']
                self.recorder.lr = lr
                self.recorder.record('train')

    def train(self):
        self.recorder.logger.info('Build train loader...')
        train_loader = build_dataloader(self.cfg.dataset.train,
                                        self.cfg,
                                        is_train=True)

        self.recorder.logger.info('Start training...')
        start_epoch = 0
        if self.cfg.resume_from:
            start_epoch = resume_network(self.cfg.resume_from, self.net,
                                         self.optimizer, self.scheduler,
                                         self.recorder)
        for epoch in range(start_epoch, self.cfg.epochs):
            self.recorder.epoch = epoch
            self.train_epoch(epoch, train_loader)
            if (epoch +
                    1) % self.cfg.save_ep == 0 or epoch == self.cfg.epochs - 1:
                self.save_ckpt()
            if (epoch +
                    1) % self.cfg.eval_ep == 0 or epoch == self.cfg.epochs - 1:
                self.validate()
            if self.recorder.step >= self.cfg.total_iter:
                break
            if self.cfg.lr_update_by_epoch:
                self.scheduler.step()


    def test(self):
        if not self.test_loader:
            self.test_loader = build_dataloader(self.cfg.dataset.test,
                                                self.cfg,
                                                is_train=False)
        self.net.eval()
        predictions = []
        for i, data in enumerate(tqdm(self.test_loader, desc=f'Testing')):
            """
            img_file_name_dic = data['meta'].data[0]
            is_first = True
            for dic in img_file_name_dic:
                file_list = dic['full_img_path'].split('/')[:-1]
                head = ''
                for f in file_list:
                    head += f + '/'
                img_5 = np.load(os.path.join(head, 'img_5.npy'))
                if is_first:
                    batch_img = img_5
                    is_first = False
                else:
                    batch_img = np.concatenate((batch_img, img_5), axis=0)
            batch_img = torch.tensor(batch_img).float()
            batch_img = batch_img.permute(0, 3, 1, 2)
            data['img'] = batch_img
            """
            # print('# generator parameters:', sum(param.numel() for param in self.net.parameters()))
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data)
                output = self.net.module.heads.get_lanes(output)
                # print(str(output))
                predictions.extend(output)
            if self.cfg.view:
                self.test_loader.dataset.view(output, data['meta'])

        metric = self.test_loader.dataset.evaluate(predictions,
                                                   self.cfg.work_dir)
        if metric is not None:
            self.recorder.logger.info('metric: ' + str(metric))
        # output_basedir
        self.draw_prediction(self.cfg.work_dir)

    def draw_prediction(self, output_basedir):
        color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (128, 0, 128), (128, 128,0), (0, 128, 128)]
        with open(os.path.join(output_basedir, 'port_predictions.json')) as infile:
            for line in infile:
                predict = eval(line)
                raw_file = predict['raw_file']
                lanes = predict['lanes']
                # 读取原图
                img = cv2.imread(os.path.join(self.cfg.dataset_path, raw_file))
                img = cv2.resize(img, (1280, 720), cv2.INTER_CUBIC)
                idx = 0
                for lane in lanes:
                    # 保存点的坐标
                    point_list = []
                    y_list = range(160, 720, 10)
                    for i in range(len(y_list)):
                        if lane[i] > 0:
                            point_list.append((lane[i], y_list[i]))
                    for i in range(len(point_list) - 1):
                        start = point_list[i]
                        stop = point_list[i + 1]
                        color = color_list[idx]
                        thick = 3
                        cv2.line(img, start, stop, color, thick)
                    idx += 1
                img_name = raw_file.split('/')[-1]
                save_file = os.path.join(output_basedir, 'result', raw_file.split('/')[-2])
                if not os.path.exists(save_file):
                    os.makedirs(save_file)
                cv2.imwrite(os.path.join(save_file, img_name), img)

        return


    def validate(self):
        if not self.val_loader:
            self.val_loader = build_dataloader(self.cfg.dataset.val,
                                               self.cfg,
                                               is_train=False)
        self.net.eval()
        predictions = []
        for i, data in enumerate(tqdm(self.val_loader, desc=f'Validate')):
            """
            img_file_name_dic = data['meta'].data[0]
            is_first = True
            for dic in img_file_name_dic:
                file_list = dic['full_img_path'].split('/')[:-1]
                head = ''
                for f in file_list:
                    head += f + '/'
                img_5 = np.load(os.path.join(head, 'img_5.npy'))
                if is_first:
                    batch_img = img_5
                    is_first = False
                else:
                    batch_img = np.concatenate((batch_img, img_5), axis=0)
            batch_img = torch.tensor(batch_img).float()
            batch_img = batch_img.permute(0, 3, 1, 2)
            data['img'] = batch_img
            """
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data)
                output = self.net.module.heads.get_lanes(output)
                predictions.extend(output)
            if self.cfg.view:
                self.val_loader.dataset.view(output, data['meta'])
        metric = self.val_loader.dataset.evaluate(predictions,
                                                  self.cfg.work_dir)
        self.recorder.logger.info('metric: ' + str(metric))

    def save_ckpt(self, is_best=False):
        save_model(self.net, self.optimizer, self.scheduler, self.recorder,
                   is_best)

    def show_modules(self):
        print(self.net.module.backbone.parameters())
