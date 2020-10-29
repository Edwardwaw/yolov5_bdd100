import math
import torch
from torch import nn


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BoxSimilarity(object):
    # calculate different kinds of iou including iou giou ciou and diou.
    def __init__(self, iou_type="giou", coord_type="xyxy", eps=1e-9):
        self.iou_type = iou_type
        self.coord_type = coord_type
        self.eps = eps

    def __call__(self, box1, box2):
        """
        :param box1: [num,4] predicts
        :param box2:[num,4] targets
        note: the shape of box1 and box2 must be same.
        :return:
        iou: shape=[num]
        """
        box1_t = box1.T
        box2_t = box2.T

        if self.coord_type == "xyxy":
            b1_x1, b1_y1, b1_x2, b1_y2 = box1_t[0], box1_t[1], box1_t[2], box1_t[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2_t[0], box2_t[1], box2_t[2], box2_t[3]
        elif self.coord_type == "xywh":
            b1_x1, b1_x2 = box1_t[0] - box1_t[2] / 2., box1_t[0] + box1_t[2] / 2.
            b1_y1, b1_y2 = box1_t[1] - box1_t[3] / 2., box1_t[1] + box1_t[3] / 2.
            b2_x1, b2_x2 = box2_t[0] - box2_t[2] / 2., box2_t[0] + box2_t[2] / 2.
            b2_y1, b2_y2 = box2_t[1] - box2_t[3] / 2., box2_t[1] + box2_t[3] / 2.
        elif self.coord_type == "ltrb":
            b1_x1, b1_y1 = 0. - box1_t[0], 0. - box1_t[1]
            b1_x2, b1_y2 = 0. + box1_t[2], 0. + box1_t[3]
            b2_x1, b2_y1 = 0. - box2_t[0], 0. - box2_t[1]
            b2_x2, b2_y2 = 0. + box2_t[2], 0. + box2_t[3]
        else:
            raise NotImplementedError("coord_type only support xyxy, xywh,ltrb")
        inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                     (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
        union_area = w1 * h1 + w2 * h2 - inter_area + self.eps
        iou = inter_area / union_area
        if self.iou_type == "iou":
            return iou

        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        if self.iou_type == "giou":
            c_area = cw * ch + self.eps
            giou = iou - (c_area - union_area) / c_area
            return giou

        diagonal_dis = cw ** 2 + ch ** 2 + self.eps
        center_dis = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                      (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
        if self.iou_type == 'diou':
            diou = iou - center_dis / diagonal_dis
            return diou

        v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
        with torch.no_grad():
            alpha = v / ((1 + self.eps) - iou + v)

        if self.iou_type == "ciou":
            ciou = iou - (center_dis / diagonal_dis + v * alpha)
            return ciou

        raise NotImplementedError("iou_type only support iou,giou,diou,ciou")




class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss



class IOULoss(object):
    def __init__(self, iou_type="giou", coord_type="xyxy"):
        super(IOULoss, self).__init__()
        self.iou_type = iou_type
        self.box_similarity = BoxSimilarity(iou_type, coord_type)

    def __call__(self, predicts, targets):
        similarity = self.box_similarity(predicts, targets)
        if self.iou_type == "iou":
            return -similarity.log()
        else:
            return 1 - similarity