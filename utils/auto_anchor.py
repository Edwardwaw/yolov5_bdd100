import torch
import numpy as np
from tqdm import tqdm
from scipy.cluster.vq import kmeans
from nets.yolov5 import YOLOv5





def kmean_anchors(path, n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """ Creates kmeans-evolved anchors from training dataset

        Arguments:
            path: a loaded dataset
            n: number of anchors
            img_size: image size used for training
            thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
            gen: generations to evolve anchors using genetic algorithm

        Return:
            k: kmeans evolved anchors

        Usage:
            from utils.general import *; _ = kmean_anchors()
    """
    thr = 1. / thr

    def metric(k, wh):  # compute metrics
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        return x, x.max(1)[0]  # x, best_x

    def fitness(k):  # mutation fitness
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # fitness

    def print_results(k):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
        print('thr=%.2f: %.4f best possible recall, %.2f anchors past thr' % (thr, bpr, aat))
        print('n=%g, img_size=%s, metric_all=%.3f/%.3f-mean/best, past_thr=%.3f-mean: ' %
              (n, img_size, x.mean(), best.mean(), x[x > thr].mean()), end='')
        for i, x in enumerate(k):
            print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')  # use in *.cfg
        return k

    dataset = path  # dataset

    # Get label wh
    shapes = img_size * np.array(dataset.shapes) / np.array(dataset.shapes).max(1, keepdims=True)
    # wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # wh
    wh0 = np.concatenate([(l[:, 4:] - l[:, 2:4]) / ds * s for ds, s, l in zip(dataset.shapes, shapes, dataset.labels)])  # wh


    # Filter
    i = (wh0 < 3.0).any(1).sum()
    if i:
        print('WARNING: Extremely small objects found. '
              '%g of %g labels are < 3 pixels in width or height.' % (i, len(wh0)))
    wh = wh0[(wh0 >= 2.0).any(1)]  # filter > 2 pixels

    # Kmeans calculation
    print('Running kmeans for %g anchors on %g points...' % (n, len(wh)))
    s = wh.std(0)  # sigmas for whitening
    k, dist = kmeans(wh / s, n, iter=30)  # points, mean distance
    k *= s
    wh = torch.tensor(wh, dtype=torch.float32)  # filtered
    wh0 = torch.tensor(wh0, dtype=torch.float32)  # unflitered
    k = print_results(k)

    # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.tight_layout()
    # fig.savefig('wh.png', dpi=200)

    # Evolve
    npr = np.random
    f, sh, mp, s = fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), desc='Evolving anchors with Genetic Algorithm')  # progress bar
    for _ in pbar:

        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * npr.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = 'Evolving anchors with Genetic Algorithm: fitness = %.4f' % f
            if verbose:
                print_results(k)

    return print_results(k)




def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.strides[-1] - m.strides[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print('Reversing anchor order')
        # m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)



def check_anchors(dataset, model, thr=4.0, imgsz=640):
    # Check anchor fit to data, recompute if necessary
    print('\nAnalyzing anchors... ', end='')
    m = model.module.head if hasattr(model, 'module') else model.head  # YOLOv5Head()
    shapes = imgsz * np.array(dataset.shapes) / np.array(dataset.shapes).max(1, keepdims=True)
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # augment scale

    wh = torch.tensor(np.concatenate([(l[:, 4:]-l[:, 2:4])/ds * s for ds, s, l in zip(dataset.shapes, shapes * scale, dataset.labels)])).float()  # wh

    def metric(k):  # compute metric
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
        best = x.max(1)[0]  # best_x
        aat = (x > 1. / thr).float().sum(1).mean()  # anchors above threshold
        bpr = (best > 1. / thr).float().mean()  # best possible recall
        return bpr, aat

    bpr, aat = metric(m.anchor_grid.clone().cpu().view(-1, 2))
    print('anchors/target = %.2f, Best Possible Recall (BPR) = %.4f' % (aat, bpr), end='')
    # if bpr < 0.98:  # threshold to recompute
    if True:
        print('. Attempting to generate improved anchors, please wait...' % bpr)
        na = m.anchor_grid.numel() // 2  # number of anchors
        new_anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        new_anchors=np.array(new_anchors,dtype=np.int)
        new_bpr = metric(new_anchors.reshape(-1, 2))[0]
        if new_bpr > bpr:  # replace anchors
            print('new bpr is ',float(new_bpr),', bpr is ',float(bpr))
            new_anchors = torch.tensor(new_anchors, device=m.anchor_grid.device).type_as(m.anchor_grid)
            m.anchor_grid[:] = new_anchors.clone().view_as(m.anchor_grid)  # for inference
            # check_anchor_order(m)
            print('New anchors saved to model. Update model *.yaml to use these anchors in the future.')
        else:
            print('Original anchors better than new anchors. Proceeding with original anchors.')
    print('')  # newline




if __name__ == '__main__':
    from datasets.coco import COCODataSets

    train_annotation_path='/home/wangchao/github_resposity/coco/annotations/instances_train2017.json'
    val_annotation_path='/home/wangchao/github_resposity/coco/annotations/instances_val2017.json'
    train_img_root='/home/wangchao/github_resposity/coco/images/train2017'
    val_img_root='/home/wangchao/github_resposity/coco/images/val2017'
    img_size=640
    batch_size=32
    num_workers=8
    debug=False
    remove_blank=True
    tdata = COCODataSets(img_root=train_img_root,
                          annotation_path=train_annotation_path,
                          img_size=img_size,
                          debug=debug,
                          augments=True,
                          remove_blank=remove_blank
                          )

    # kmean_anchors(tdata, n=9, img_size=img_size, thr=4.0, gen=1000, verbose=False)
    model = YOLOv5(in_channels=3,
                   num_cls=80,
                   anchors=[[10, 13, 16, 30, 33, 23],
                            [30, 61, 62, 45, 59, 119],
                            [116, 90, 156, 198, 373, 326]],
                   strides=[8, 16, 32],
                   scale_name='s')

    check_anchors(tdata,model)






