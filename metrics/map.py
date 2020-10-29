import torch
import numpy as np
from commons.boxs_utils import box_iou,scale_coords,xyxy2xywh,calculate_border
from pathlib import Path
from datasets.coco import coco_ids
from datasets.BDD100 import BDD100_ids
import json


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls: Predicted object classes (nparray).
        target_cls: True object classes (nparray).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    pr_score = 0.1  # score to evaluate P and R https://github.com/ultralytics/yolov3/issues/898
    s = [unique_classes.shape[0], tp.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 or n_gt == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_gt + 1e-16)  # recall curve
            r[ci] = np.interp(-pr_score, -conf[i], recall[:, 0])  # r at pr_score, negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-pr_score, -conf[i], precision[:, 0])  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

            # Plot
            # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            # ax.plot(recall, precision)
            # ax.set_xlabel('Recall')
            # ax.set_ylabel('Precision')
            # ax.set_xlim(0, 1.01)
            # ax.set_ylim(0, 1.01)
            # fig.tight_layout()
            # fig.savefig('PR_curve.png', dpi=300)

    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')


def coco_map(predicts_list, targets_list,  ID_list, shape_list, net_input_size, save_json=True):
    """
    :param predicts_list(list, len=len(dataset)): per_img predicts_shape [n,6] (x1,y1,x2,y2,score,cls_id)
    :param targets_list(list, len=len(dataset)): per_img targets_shape [m, 5] (cls_id,x1,y1,x2,y2)
    :param ID_list(list, len=len(dataset)):  image path, shape=[w,h]
    :param shapes_list(list, len=len(dataset)): original image shape=[w0,h0], which is used for evaluate mAP
    :return:
    """
    device = targets_list[0].device
    # 设置iou阈值，从0.5~0.95，每间隔0.05取一次
    iouv = torch.linspace(0.5, 0.95, 10).to(device)    # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    stats = list()
    jdict=[]

    for predicts, targets, path, original_shape in zip(predicts_list, targets_list, ID_list, shape_list):
        # 获取第i张图片的标签信息, 包括x1,y1,x2,y2,score,cls_id
        nl = len(targets)
        tcls = targets[:, 0].tolist() if nl else []

        # 如果预测为空，则添加空的信息到stats里
        if predicts is None:
            if nl:
                stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            continue

        # Append to pycocotools JSON dictionary
        if save_json:
            # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
            ##-------------- change it --------------------------------------------------------------------
            # image_id = Path(path).stem  # for coco
            image_id = path  # for BDD
            ##-------------------------------------------------------------------------------------------
            box_json = predicts[:, :4].clone()  # x1y1x2y2
            ratio,pad=calculate_border(original_shape[::-1], net_input_size)  # note: shape=[w,h]
            scale_coords(None, box_json, original_shape[::-1], (ratio,pad))  # to original shape
            box_json = xyxy2xywh(box_json)  # xywh
            box_json[:, :2] -= box_json[:, 2:] / 2  # xy center to top-left corner
            for p, b in zip(predicts.tolist(), box_json.tolist()):
                jdict.append({#'image_id': int(image_id) if image_id.isnumeric() else image_id,  # coco
                              'image_id': image_id,  # BDD100
                              # 'category_id': coco_ids[int(p[5])],  # coco
                              'category_id': BDD100_ids[int(p[5])],   #BDD100
                              'bbox': [round(x, 3) for x in b],
                              'score': round(p[4], 5)})

        # Assign all predictions as incorrect
        # 初始化预测评定，niou为iou阈值的个数
        correct = torch.zeros(predicts.shape[0], niou, dtype=torch.bool, device=device)
        if nl:
            detected = list()   # detected用来存放已检测到的目标
            tcls_tensor = targets[:, 0]
            tbox = targets[:, 1:5]

            # Per target class
            # 对图片中的每个类单独处理
            for cls in torch.unique(tcls_tensor):
                ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)     # prediction indices
                pi = (cls == predicts[:, 5]).nonzero(as_tuple=False).view(-1)   # target indices
                # Search for detections
                if pi.shape[0]:
                    # Prediction to target ious
                    # box_iou计算预测框与标签框的iou值，max(1)选出最大的ious值,i为对应的索引
                    """
                    pred shape[N, 4]
                    tbox shape[M, 4]
                    box_iou shape[N, M]
                    ious shape[N, 1]
                    i shape[N, 1], i里的值属于0~M
                    """
                    ious, i = box_iou(predicts[pi, :4], tbox[ti]).max(1)
                    # Append detections
                    for j in (ious > iouv[0]).nonzero(as_tuple=False):   #选择出iou>0.5的pred_box索引
                        d = ti[i[j]]  # 有对应iou>0.5的pred_box所对应的bbox
                        if d not in detected:
                            detected.append(d)
                            correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn (n=num of iou thresh)
                        if len(detected) == nl:              # all targets already located in image
                            break
        # Append statistics (correct, conf, pcls, tcls)
        stats.append((correct.cpu(), predicts[:, 4].cpu(), predicts[:, 5].cpu(), tcls))

    # Save JSON
    if save_json and len(jdict):
        f = 'detections_val2017_results.json'  # filename
        print('\nCOCO mAP with pycocotools... saving %s...' % f)
        with open(f, 'w') as file:
            json.dump(jdict, file)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            import glob

            ###---------------------------------------------------------------------------------------------------------
            # note: 以下为随着数据集不同而需要修改的项目
            # imgIds = [int(Path(x).stem) for x in ID_list]   # coco
            imgIds = ID_list  # BDD100
            # cocoGt = COCO('/home/wangchao/github_resposity/coco/annotations/instances_val2017.json')  # initialize COCO ground truth api
            cocoGt = COCO('/home/wangchao/public_dataset/BDD100/annotations/bdd100k_labels_images_det_coco_val.json')  # initialize BDD ground truth api
            ##----------------------------------------------------------------------------------------------------------


            cocoDt = cocoGt.loadRes(f)  # initialize COCO pred api
            # 创建评估器
            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.params.imgIds = imgIds  # image IDs to evaluate
            # 评估
            cocoEval.evaluate()
            cocoEval.accumulate()
            # 展示结果
            cocoEval.summarize()
            map, map50 = cocoEval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print('ERROR: pycocotools unable to run: %s' % e)

    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        # change it with dataset
        maps=np.zeros(13)+map
        for i,c in enumerate(ap_class):
            maps[c]=ap[i]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        return mp, mr, map50, map, maps
    else:
        # changed it with dataset
        return 0., 0., 0., 0., np.zeros(13)




