import torch
import cv2
import numpy as np



def draw_box(img, labels, colors, names):
    if labels.shape[0] < 1:
        return img
    ret_img = img.copy()
    for weights, label, x1, y1, x2, y2 in labels:
        cv2.rectangle(ret_img, (int(x1), int(y1)), (int(x2), int(y2)), color=colors[int(label)], thickness=2)
        cv2.putText(ret_img, "{:s}".format(names[int(label)]), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   colors[int(label)], 2)
        # cv.putText(ret_img, "{:.2f}".format(float(weights)), (int(x1), int(y1 + 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5,
        #            colors[int(label)],
        #            1)
    return ret_img




def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2



def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.t())  # shape=[N]
    area2 = box_area(box2.t())  # shape=M]

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)




def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]  #(left,top)

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def calculate_border(original_shape,target_shape):
    '''
    x1x2 = ratio[0] * x1x2 + left
    y1y2= ratio[1] * y1y2 + top

    :param original_shape: [h,w]
    :param target_shape: [h,w] or single int
    :return:
    '''
    if isinstance(target_shape, int):
        new_shape = (target_shape, target_shape)
    else:
        new_shape=target_shape
    r = min(new_shape[0] / original_shape[0], new_shape[1] / original_shape[1])
    ratio = r, r
    new_unpad = int(round(original_shape[0] * r)), int(round(original_shape[1] * r))  #[unpadded_h,unpadded_w]
    dw, dh = new_shape[1] - new_unpad[1], new_shape[0] - new_unpad[0]

    dw /= 2
    dh /= 2

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    return ratio, (left, top)
