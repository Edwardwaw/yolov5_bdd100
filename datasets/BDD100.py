import os
import torch
import cv2 as cv
import numpy as np
from torch.utils.data.dataset import Dataset
from pycocotools.coco import COCO
from commons.boxs_utils import draw_box, xyxy2xywh
from commons.augmentations import Compose, OneOf, ScalePadding, RandNoise, Mosaic, MixUp, RandPerspective, HSV, Identity, LRFlip, RandCutOut



######################################################################################################################
#### coco parameters -------------------------------------------------------------------------------------------------


BDD100_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

BDD100_names = ['person','rider','car','bus','truck','bike','motor','tl_green',
              'tl_red','tl_yellow','tl_none','traffic sign','train']

colors = [(67, 68, 113), (130, 45, 169), (2, 202, 130), (127, 111, 90), (92, 136, 113),
          (33, 250, 7), (238, 92, 104), (0, 151, 197), (134, 9, 145), (253, 181, 88),
          (246, 11, 137), (55, 72, 220), (136, 8, 253)]

######################################################################################################################
### data augmentation hyperparameters --------------------------------------------------------------------------------

default_aug_cfg = {
    'hsv_h': 0.014,
    'hsv_s': 0.68,
    'hsv_v': 0.36,
    'degree': 5,
    'translate': 0.1,
    'scale': (0.6, 1.5),
    'shear': 0.0,
    'beta': (8, 8),
    'pad_val': (103, 116, 123),
}

# rgb_mean = [0.485, 0.456, 0.406]
# rgb_std = [0.229, 0.224, 0.225]
cv.setNumThreads(0)



############## CoCoDataset -------------------------------------------------------------------------------------------


# noinspection PyTypeChecker
class BDD100DataSets(Dataset):
    def __init__(self, img_root, annotation_path,
                 img_size=640,
                 augments=True,
                 use_crowd=True,
                 debug=False,
                 remove_blank=True,
                 aug_cfg=None,
                 image_weight=False
                 ):
        """
        :param img_root: 图片根目录
        :param annotation_path: 标注（json）文件的路径
        :param img_size: 长边的size
        :param augments: 是否进行数据增强
        :param use_crowd: 是否使用crowed的标注
        :param debug: debug模式(少量数据)
        :param remove_blank: 是否过滤掉没有标注的数据
        :param aug_cfg: 数据增强中配置
        """
        super(BDD100DataSets, self).__init__()
        self.coco = COCO(annotation_path)
        self.img_size = img_size
        self.img_root = img_root
        self.use_crowd = use_crowd
        self.remove_blank = remove_blank
        self.image_weight = image_weight
        self.data_len = len(self.coco.imgs.keys())
        self.img_paths = [None] * self.data_len
        self.shapes = [None] * self.data_len
        self.img_IDs= [None] * self.data_len

        # [label_weights, label_index, x1, y1, x2, y2]
        self.labels = [np.zeros((0, 6), dtype=np.float32)] * self.data_len

        self.augments = augments
        if aug_cfg is None:
            aug_cfg = default_aug_cfg
        self.aug_cfg = aug_cfg
        self.debug = debug
        self.empty_images_len = 0

        valid_len = self.__load_data()
        if valid_len != self.data_len:
            print("valid data len: ", valid_len)
            self.data_len = valid_len
            self.img_paths = self.img_paths[:valid_len]
            self.shapes = self.shapes[:valid_len]
            self.labels = self.labels[:valid_len]
            self.img_IDs = self.img_IDs[:valid_len]
        if self.debug:
            assert debug <= valid_len, "not enough data to debug"
            print("debug")
            self.img_paths = self.img_paths[:debug]
            self.shapes = self.shapes[:debug]
            self.labels = self.labels[:debug]
            self.img_IDs = self.img_IDs[:debug]
        self.transform = None
        self.set_transform()


    def __load_data(self):
        index = 0
        for img_id in self.coco.imgs.keys():
            file_name = self.coco.imgs[img_id]['file_name']
            width, height = self.coco.imgs[img_id]['width'], self.coco.imgs[img_id]['height']
            file_path = os.path.join(self.img_root, file_name)
            if not os.path.exists(file_path):
                print("img {:s} is not exist".format(file_path))
                continue

            assert width > 1 and height > 1, "invalid width or heights"

            anns = self.coco.imgToAnns[img_id]
            label_list = list()
            for ann in anns:
                category_id, box, iscrowd = ann['category_id'], ann['bbox'], ann['iscrowd']
                label_id = BDD100_ids.index(category_id)
                assert label_id >= 0, 'error label_id'
                if not self.use_crowd and iscrowd == 1:
                    continue
                x1, y1 = box[:2]
                x2, y2 = x1 + box[2], y1 + box[3]
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                if x2 - x1 < 1 or y2 - y1 < 1:
                    print("not a valid box ", box)
                    continue
                if x1 < 0 or x2 > width or y1 < 0 or y2 > height:
                    print("warning box ", box)
                # note: one box's label is (class_weight, label_id, x1, y1, x2, y2), shape=[5,]
                label_list.append((1., label_id, x1, y1, x2, y2))
            if self.remove_blank:
                if len(label_list) < 1:
                    self.empty_images_len += 1
                    continue
            if label_list:
                self.labels[index] = np.array(label_list, dtype=np.float32)

            self.img_paths[index] = file_path
            self.shapes[index] = (width, height)
            self.img_IDs[index] = img_id
            index += 1
        return index


    def __getitem__(self, item):
        if self.image_weight:
            item=self.indices[item]
        img_path, label = self.img_paths[item], self.labels[item]
        img_ID=self.img_IDs[item]
        img = cv.imread(img_path)
        # in transform function, [x1,y1,x2,y2] are unnormalized coords.
        img, label = self.transform(img, label)
        if self.debug:
            import uuid
            ret_img = draw_box(img, label, colors, BDD100_names)
            cv.imwrite("{:d}_{:s}.jpg".format(item, str(uuid.uuid4()).replace('-', "")), ret_img)
        label_num = len(label)
        # normalized boxes
        if label_num:
            # [weight,label,x1,y1,x2,y2]
            label[:, [3, 5]] /= img.shape[0]  # height
            label[:, [2, 4]] /= img.shape[1]  # width
        img_out = (img[:, :, [2, 1, 0]] / 255.0).transpose(2, 0, 1).astype(np.float32)
        img_out = np.ascontiguousarray(img_out)
        assert not np.any(np.isnan(img_out)), "invalid input"
        labels_out = torch.zeros((label_num, 7))
        if label_num:
            labels_out[:, 1:] = torch.from_numpy(label)
        return torch.from_numpy(img_out).float(), labels_out, img_ID, self.shapes[item]


    def set_transform(self):
        if self.augments:
            color_jitter = OneOf(transforms=[Identity(),
                                             HSV(p=1,
                                                 hgain=self.aug_cfg['hsv_h'],
                                                 sgain=self.aug_cfg['hsv_s'],
                                                 vgain=self.aug_cfg['hsv_v']),
                                             RandNoise()
                                             ])
            mosaic = Mosaic(self.img_paths,
                            self.labels,
                            color_gitter=color_jitter,
                            target_size=self.img_size,
                            pad_val=self.aug_cfg['pad_val'],
                            )
            mix_up = MixUp(self.img_paths,
                           self.labels,
                           color_gitter=color_jitter,
                           target_size=self.img_size,
                           pad_val=self.aug_cfg['pad_val'],
                           beta=self.aug_cfg['beta'])
            perspective_transform = RandPerspective(target_size=(self.img_size, self.img_size),
                                                    scale=self.aug_cfg['scale'],
                                                    degree=self.aug_cfg['degree'],
                                                    translate=self.aug_cfg['translate'],
                                                    shear=self.aug_cfg['shear'],
                                                    pad_val=self.aug_cfg['pad_val'])
            basic_transform = Compose(transforms=[
                color_jitter,
                RandCutOut(),
                ScalePadding(target_size=self.img_size, padding_val=self.aug_cfg['pad_val']),
                perspective_transform,
            ])
            aug_mosaic = Mosaic(self.img_paths,
                                self.labels,
                                color_gitter=mix_up,
                                target_size=self.img_size,
                                pad_val=self.aug_cfg['pad_val'],
                                )
            aug_mixup = MixUp(self.img_paths,
                              self.labels,
                              color_gitter=mosaic,
                              target_size=self.img_size,
                              pad_val=self.aug_cfg['pad_val'],
                              beta=self.aug_cfg['beta'])

            self.transform = Compose(transforms=[
                OneOf(transforms=[
                    (0.2, basic_transform),
                    (0.8, mosaic),
                    (0., mix_up),
                    (0., aug_mosaic),
                    (0., aug_mixup)
                ]),
                LRFlip()
            ])
        else:
            self.transform = ScalePadding(target_size=(self.img_size, self.img_size),
                                          padding_val=self.aug_cfg['pad_val'])

    def __len__(self):
        return len(self.img_paths)

    @staticmethod
    def collate_fn(batch):
        """
        :param batch:
        :return: images shape[bs,3,h,w] targets[bs,7] (bs_idx,weights,label_idx,x1,y1,x2,y2)
        """
        img, label, img_IDs, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), img_IDs, shapes




if __name__ == '__main__':
    from torch.utils.data.dataloader import DataLoader

    dataset = BDD100DataSets(img_root="/home/wangchao/github_resposity/BDD100/images/trains",
                           annotation_path="/home/wangchao/github_resposity/BDD100/annotations/bdd100k_labels_images_det_coco_train.json",
                           use_crowd=True,
                           augments=True,
                           debug=False
                           )
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, num_workers=8, collate_fn=dataset.collate_fn)
    for img_tensor, target_tensor, _ , _ in dataloader:
        for weights in target_tensor[:, 1].unique():
            nonzero_index = torch.nonzero((target_tensor[:, 1] == weights), as_tuple=True)
            print(target_tensor[nonzero_index].shape)
        print("=" * 20)

    # for img_tensor, target_tensor, _ in dataloader:
    #     for weights in target_tensor[:, 1].unique():
    #         nonzero_index = torch.nonzero((target_tensor[:, 1] == weights), as_tuple=True)
    #         print(target_tensor[nonzero_index].shape)
    #     print("=" * 20)