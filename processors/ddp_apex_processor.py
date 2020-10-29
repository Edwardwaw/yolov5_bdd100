import os
import yaml
import random
import torch
import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from torch import nn
from torch.cuda import amp
from torch.utils.data.distributed import DistributedSampler
from datasets.coco import COCODataSets
from datasets.BDD100 import BDD100DataSets
from nets.yolov5 import YOLOv5
from losses.yolov5_loss import YOLOv5LossOriginal
from torch.utils.data.dataloader import DataLoader
from utils.yolo_utils import non_max_suppression
from commons.boxs_utils import clip_coords
from commons.model_utils import rand_seed,is_parallel,ModelEMA,freeze_bn,labels_to_class_weights,labels_to_image_weights
from commons.optims_utils import WarmUpCosineDecayMultiStepLRAdjust,split_optimizer

from utils.auto_anchor import check_anchors
from metrics.map import coco_map
from torch.nn.functional import interpolate

rand_seed(1024)


class COCODDPApexProcessor(object):
    def __init__(self, cfg_path):
        with open(cfg_path, 'r') as rf:
            self.cfg = yaml.safe_load(rf)
        self.data_cfg = self.cfg['data']   # dataset params
        self.model_cfg = self.cfg['model']  # model params
        self.optim_cfg = self.cfg['optim']  # optim params
        self.hyper_params = self.cfg['hyper_params']   # other hyper params
        self.val_cfg = self.cfg['val']   # validation hyper params
        print(self.data_cfg)
        print(self.model_cfg)
        print(self.optim_cfg)
        print(self.hyper_params)
        print(self.val_cfg)

        os.environ['CUDA_VISIBLE_DEVICES'] = self.cfg['gpus'] # set avaliable gpu
        dist.init_process_group(backend='nccl',init_method='env://')

        ## load dataset ---------------------------------------------------------------------------------------
        # self.tdata = COCODataSets(img_root=self.data_cfg['train_img_root'],
        #                           annotation_path=self.data_cfg['train_annotation_path'],
        #                           img_size=self.data_cfg['img_size'],
        #                           debug=self.data_cfg['debug'],
        #                           augments=True,
        #                           remove_blank=self.data_cfg['remove_blank'],
        #                           image_weight=self.hyper_params['use_weight_sample']
        #                           )
        self.tdata = BDD100DataSets(img_root=self.data_cfg['train_img_root'],
                                    annotation_path=self.data_cfg['train_annotation_path'],
                                    img_size=self.data_cfg['img_size'],
                                    debug=self.data_cfg['debug'],
                                    augments=True,
                                    remove_blank=self.data_cfg['remove_blank'],
                                    image_weight=self.hyper_params['use_weight_sample']
                                    )
        self.tloader = DataLoader(dataset=self.tdata,
                                  batch_size=self.data_cfg['batch_size'],
                                  num_workers=self.data_cfg['num_workers'],
                                  collate_fn=self.tdata.collate_fn,
                                  sampler=DistributedSampler(dataset=self.tdata, shuffle=True))

        # self.vdata = COCODataSets(img_root=self.data_cfg['val_img_root'],
        #                           annotation_path=self.data_cfg['val_annotation_path'],
        #                           img_size=self.data_cfg['img_size'],
        #                           debug=self.data_cfg['debug'],
        #                           augments=False,
        #                           remove_blank=False
        #                           )
        self.vdata = BDD100DataSets(img_root=self.data_cfg['val_img_root'],
                                    annotation_path=self.data_cfg['val_annotation_path'],
                                    img_size=self.data_cfg['img_size'],
                                    debug=self.data_cfg['debug'],
                                    augments=False,
                                    remove_blank=False
                                    )
        self.vloader = DataLoader(dataset=self.vdata,
                                  batch_size=self.data_cfg['batch_size'],
                                  num_workers=self.data_cfg['num_workers'],
                                  collate_fn=self.vdata.collate_fn,
                                  sampler=DistributedSampler(dataset=self.vdata, shuffle=False))

        print("train_data: ", len(self.tdata), " | ",
              "val_data: ", len(self.vdata), " | ",
              "empty_data: ", self.tdata.empty_images_len)
        print("train_iter: ", len(self.tloader), " | ",
              "val_iter: ", len(self.vloader))

        ### define model -------------------------------------------------------------------------------------
        model=YOLOv5(in_channels=3,
                     num_cls=self.model_cfg['num_cls'],
                     anchors=self.model_cfg['anchors'],
                     strides=self.model_cfg['strides'],
                     scale_name=self.model_cfg['scale_name'])
        ### check anchor -------------------------------------------------------------------------------------
        check_anchors(self.tdata,model,self.hyper_params['anchor_t'],self.data_cfg['img_size'])
        ############------------------------------------------------------------------------------------------
        self.best_map = 0.
        self.best_map50 = 0.

        optimizer=split_optimizer(model,self.optim_cfg)
        local_rank=dist.get_rank()
        self.local_rank=local_rank
        self.device=torch.device('cuda',local_rank)
        model.to(self.device)
        pretrain=self.model_cfg.get('pretrain',None)
        if pretrain:
            pretrained_weights=torch.load(pretrain,map_location=self.device)
            load_info=model.load_state_dict(pretrained_weights['ema'],strict=False)
            print('load info ',load_info)

        # 通过torch1.6自带的api设置混合精度训练
        self.scaler = amp.GradScaler(enabled=True)
        #多卡batchnormalized同步
        if self.optim_cfg['sync_bn']:
            model=nn.SyncBatchNorm.convert_sync_batchnorm(model)
        self.model=nn.parallel.distributed.DistributedDataParallel(model,
                                                                   device_ids=[local_rank],
                                                                   output_device=local_rank)
        self.optimizer=optimizer
        self.ema=ModelEMA(self.model)

        self.creterion=YOLOv5LossOriginal(
            iou_type=self.hyper_params['iou_type'],
            fl_gamma=self.hyper_params['fl_gamma'],
            class_smoothing_eps=self.hyper_params['class_smoothing_eps']
        )

        self.lr_adjuster=WarmUpCosineDecayMultiStepLRAdjust(init_lr=self.optim_cfg['lr'],
                                                            milestones=self.optim_cfg['milestones'],
                                                            warm_up_epoch=self.optim_cfg['warm_up_epoch'],
                                                            iter_per_epoch=len(self.tloader),
                                                            epochs=self.optim_cfg['epochs'],
                                                            cosine_weights=self.optim_cfg['cosine_weights'])
        ## for class-aware weighted sampling ---------------------------------------------------------------------
        self.class_weights=labels_to_class_weights(self.tdata.labels,nc=self.model_cfg['num_cls']).to(self.device) if self.hyper_params['use_weight_sample'] else None
        self.maps=np.zeros(self.model_cfg['num_cls'])  #mAP per class



    def train(self,epoch):
        self.model.train()
        if self.model_cfg['freeze_bn']:
            self.model.apply(freeze_bn)
        if self.local_rank==0:
            pbar=tqdm(self.tloader)
        else:
            pbar=self.tloader

        if self.hyper_params['use_weight_sample']:
            cw=self.class_weights.cpu().numpy()*(1-self.maps)**2   # class weight
            iw=labels_to_image_weights(self.tdata.labels,nc=self.model_cfg['num_cls'],class_weights=cw)  # image weight
            self.tdata.indices = random.choices(range(len(self.tdata)), weights=iw, k=len(self.tdata))  # rand weighted idx


        loss_list = [list(), list(), list(), list()]  # loss_box, loss_obj, loss_cls, loss
        lr = 0
        match_num = 0



        for i, (img_tensor, targets_tensor, _, _) in enumerate(pbar):
            '''
            img_tensor: [bs,3,h,w]
            targets_tensor:  [bs,7] (bs_idx,weights,label_idx,x1,y1,x2,y2), box annotations have been normalized
            '''
            # DDP training
            if len(self.hyper_params['multi_scale']) >= 2:
                target_size = np.random.choice(self.hyper_params['multi_scale'])
                img_tensor = interpolate(img_tensor, mode='bilinear', size=target_size, align_corners=False)
            _, _, h, w = img_tensor.shape
            with torch.no_grad():
                img_tensor = img_tensor.to(self.device)
                # bs_idx,weights,label_idx,x1,y1,x2,y2
                targets_tensor[:, [5, 6]] = targets_tensor[:, [5, 6]] - targets_tensor[:, [3, 4]]
                targets_tensor[:, [3, 4]] = targets_tensor[:, [3, 4]] + targets_tensor[:, [5, 6]] / 2.
                targets_tensor = targets_tensor.to(self.device)

            self.optimizer.zero_grad()
            # Forward
            # 混合精度
            with amp.autocast(True):
                '''
                predicts(list): len=nl, predicts[i].shape=(bs,3,ny,nx,85)
                normalized_anchor(torch.Tensor): shape=[3,3,2]
                '''
                predicts,anchors=self.model(img_tensor)
                # 计算损失，包括分类损失，objectness损失，框的回归损失
                # loss为总损失值，loss_items为一个元组，包含分类损失，objectness损失，框的回归损失和总损失
                total_loss, detail_loss, total_num = self.creterion(predicts, targets_tensor, anchors)   # loss scaled by batch_size

            self.scaler.scale(total_loss).backward()
            match_num+=total_num
            # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.optim_cfg['max_norm'],norm_type=2)
            self.lr_adjuster(self.optimizer, i, epoch)
            lr=self.optimizer.param_groups[0]['lr']
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.ema.update(self.model)

            loss_box, loss_obj, loss_cls, loss = detail_loss
            loss_list[0].append(loss_box.item())
            loss_list[1].append(loss_obj.item())
            loss_list[2].append(loss_cls.item())
            loss_list[3].append(loss.item())

            if self.local_rank==0:
                pbar.set_description(
                    "epoch:{:2d}|match_num:{:4d}|size:{:3d}|loss:{:6.4f}|loss_box:{:6.4f}|loss_obj:{:6.4f}|loss_cls:{:6.4f}|lr:{:8.6f}".format(
                        epoch + 1,
                        int(total_num),
                        h,
                        loss.item(),
                        loss_box.item(),
                        loss_obj.item(),
                        loss_cls.item(),
                        lr
                        ))
        self.ema.update_attr(self.model)
        mean_loss_list = [np.array(item).mean() for item in loss_list]
        print(
            "epoch:{:3d}|match_num:{:4d}|local:{:3d}|loss:{:6.4f}||loss_box:{:6.4f}|loss_obj:{:6.4f}|loss_cls:{:6.4f}|lr:{:8.6f}"
                .format(epoch + 1,
                        match_num,
                        self.local_rank,
                        mean_loss_list[3],
                        mean_loss_list[0],
                        mean_loss_list[1],
                        mean_loss_list[2],
                        lr))


    @torch.no_grad()
    def val(self,epoch):
        if self.local_rank!=0:
            return
        predict_list = list()
        target_list = list()
        # self.model.eval()
        if self.local_rank==0:
            pbar=tqdm(self.vloader)
        else:
            pbar=self.vloader
        path_list,shape_list=[],[]
        for img_tensor, targets_tensor, path, shapes in pbar:
            _, _, h, w = img_tensor.shape
            targets_tensor[:, 3:] = targets_tensor[:, 3:] * torch.tensor(data=[w, h, w, h])
            img_tensor = img_tensor.to(self.device)
            targets_tensor = targets_tensor.to(self.device)
            predicts=self.ema.ema(img_tensor)
            #  predicts after nms is a list(len=bs), its element has shape=[N,6]  6==>(x1, y1, x2, y2, conf, cls)
            predicts=non_max_suppression(predicts,
                                         conf_thresh=self.val_cfg['conf_thresh'],
                                         iou_thresh=self.val_cfg['iou_thresh'],
                                         max_det=self.val_cfg['max_det'])

            for i,predict in enumerate(predicts):
                if predict is not None:
                    clip_coords(predict,(h,w))
                predict_list.append(predict)
                targets_sample=targets_tensor[targets_tensor[:,0]==i][:,2:]
                target_list.append(targets_sample)
                path_list.append(path[i])
                shape_list.append(shapes[i])
            # after above code block, predict_list(len=len(dataset)), its element shape = [n,6]  6==>(x1,y1,x2,y2,score,cls_id)
            #                         target_list(len=len(dataset)), its element shape = [m, 5] 5==>(cls_id,x1,y1,x2,y2)
        mp, mr, map50, map, self.maps = coco_map(predict_list, target_list, path_list, shape_list, self.data_cfg['img_size'])
        print("epoch: {:2d}|local:{:d}|mp:{:6.4f}|mr:{:6.4f}|map50:{:6.4f}|map:{:6.4f}"
              .format(epoch + 1,
                      self.local_rank,
                      mp * 100,
                      mr * 100,
                      map50 * 100,
                      map * 100))

        last_weight_path = os.path.join(self.val_cfg['weight_path'],"{:s}_last.pth".format(self.cfg['model_name']))
        best_map_weight_path = os.path.join(self.val_cfg['weight_path'],"{:s}_best_map.pth".format(self.cfg['model_name']))
        best_map50_weight_path = os.path.join(self.val_cfg['weight_path'],"{:s}_best_map50.pth".format(self.cfg['model_name']))
        # model_static = self.model.module.state_dict() if is_parallel(self.model) else self.model.state_dict()

        ema_static = self.ema.ema.state_dict()
        cpkt = {
            "ema": ema_static,
            "map": map * 100,
            "epoch": epoch,
            "map50": map50 * 100
        }

        if self.local_rank!=0:
            return

        torch.save(cpkt,last_weight_path)
        if map > self.best_map:
            torch.save(cpkt, best_map_weight_path)
            self.best_map = map
        if map50 > self.best_map50:
            torch.save(cpkt, best_map50_weight_path)
            self.best_map50 = map50


    def run(self):
        for epoch in range(self.optim_cfg['epochs']):
            # self.train(epoch)
            if (epoch + 1) % self.val_cfg['interval'] == 0:
                self.val(epoch)
        dist.destroy_process_group()
        torch.cuda.empty_cache()





