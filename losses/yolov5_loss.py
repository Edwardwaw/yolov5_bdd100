import torch
from torch import nn
from losses.commons import BoxSimilarity,FocalLoss,smooth_BCE



class YOLOv5Builder(object):
    def __init__(self, ratio_thresh=4., expansion_bias=0.5):
        '''

        :param ratio_thresh: 标签的长h宽w/anchor的长h_a宽w_a阈值, 即h/h_a, w/w_a都要在(1/2.26, 2.26)之间
        :param expansion_bias: anchor匹配时的centrol point位置偏移，能使同一个anchor匹配到更多的样本
        '''
        super(YOLOv5Builder, self).__init__()
        self.ratio_thresh = ratio_thresh
        self.expansion_bias = expansion_bias

    def __call__(self, predicts, targets, input_anchors):
        """
        :param predicts: list(predict) [bs,anchor_per_grid,ny,nx,output(num_cls+5)]
        :param targets: [gt_num,7] (bs_id,weight,label_idx,xc,yc,w,h){normalized w,h}
        :param input_anchors:[layer_num,anchor_per_grid,2(w,h)], which has been transformed into featuremap size
        :return:
        """
        device = targets.device
        num_layer, anchor_per_grid = input_anchors.shape[:2]   # num of predicted layer(3), num of anchors per grid(3)
        num_gt = targets.shape[0]   # targets(num of gt-boxes in this batch)

        target_weights_cls, target_box, target_indices, target_anchors = list(), list(), list(), list()
        gain = torch.ones(8, device=device).float()  # normalized to gridspace gain

        # ai.shape=[na,nt]
        anchor_idx = torch.arange(anchor_per_grid, device=device).float().view(anchor_per_grid, 1).repeat(1, num_gt)
        '''
           在最后一维添加anchor index, shape=[na,nt,8]
           每一列代表一个目标对应的3个anchor的序号0、1、2,
           即为每一个目标添加可能匹配到的3个不同尺度的anchor
        '''
        targets = torch.cat((targets.repeat(anchor_per_grid, 1, 1), anchor_idx[:, :, None]), dim=2)  # append anchor indices,
        # offsets
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1]]  # j,k,l,m
                           # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                           ,device=device).float() * self.expansion_bias

        for i in range(num_layer):  # 分别对3个yolo层进行处理
            anchors = input_anchors[i]   # 得到第i个yolo层的anchor的参数,shape=[3,2],已经调整到对应的featuremap尺度表示
            '''
               gain.shape=8
               这里是将gain张量中的第3到7个元素设置为w,h,w,h.其中w,h就是当前yolo层特征的宽、高
            '''
            gain[3:7] = torch.tensor(predicts[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            # 这里是将每个目标的坐标参数转换成在当前yolo层的特征图的维度上的坐标
            t = targets * gain
            if num_gt:
                # 有gt-box存在
                # Matches
                r = t[:, :, 5:7] / anchors[:, None, :]  # calculate wh ratio, r.shape=[na,nt,2]
                # matching gt boxes anchors
                '''
                r: targets的w, h与当前yolo层的anchor的w, h的比值
                1/r: anchor的w, h与target的w, h的比值

                这里求出了当前yolo层的所有anchor与所有targets的宽、高之间的比值。
                如果一个anchor与一个target的宽、高比值以及target与anchor宽、高的比值小于一定阈值，则将这个anchor分配到对应的target, 
                即j中对应的元素为true（说明：若j[i,t]=True，则说明anchor[i]的box预测是关于target[t]的一个正样本）。

                问：一个target可以分配给多个anchor ？？？
                回答：一个target可能有多个anchor，也可能没有anchor

                j: 3(na)x31(nt), represents which anchor is assigned to target in pi yolo layer
                j的每一列表示一个目标所在格子的3个anchor中，有哪些anchor被分配到了这个目标。因为每个目标所
                在的格子有3个anchor，前面通过计算这3个anchor与这个目标方框宽、高的比值以及它们的倒数，对于
                小于anchor_t这个阈值的元素，j中对应位置值为true，大于等于的为false。这样就完成了所有目标的anchor分配
                '''

                # h/h_a, w/w_a都要在(1/self.ratio_thresh, self.ratio_thresh)之间
                # compare, valid_idx (bool, array), valid_idx.shape=[na,nt]
                valid_idx = torch.max(r, 1. / r).max(2)[0] < self.ratio_thresh

                '''
                   a: shape is 31, an item represents an anchor id which matches a gt box
                      这里每个目标对应的anchor在3个anchor中的序号存放在a中，假设有n个目标，根据前面r,1/r的最大值是否小于阈值来决定一个目标是否会被分配anchor，
                      如果3个anchor与这个目标的r,1/r的值都满足要求，则这个目标将会被分配3个anchor。所以这里a元素的个数不一定等于目标的数量，
                      而是有多少个anchor被分配到这些目标就会有多少个元素。这些被分配的anchor对应的真实标签存放在t中的一行（解析：t=t[j]即为匹配到target的
                      anchor的标注信息，t.shape=[ns,8] ns为匹配到target的anchor的个数， 8==>(image_id,class_score,class_id,x,y,w,h,anchor_id)）。

                   重点：anchor和目标之间是相互配对，可能有目标没有分配anchor，也有anchor没有分配目标
                        (yolov5中anchor和target之间的匹配是多对多的关系)

                   问：如果有k个目标处于同一个格子，那这个格子的anchor并没有限定为3个，最多可能有3k个anchor分配到这k个目标？？？
                      但是特征图的维度是(b,na,h,w,(c+5))，一个格子最多只有3个anchor,这里到底是怎么回事？？？

                   问：多个目标处于同一个格子，anchor到底怎么分配的？？？
                   回答：根据我在github上对作者进行的提问，发现目前对于多个目标处于同一个格子的情况，他的解决办法就是增加anchor数量。
                        issue链接： https://github.com/ultralytics/yolov5/issues/611
                '''

                '''
                t.shape=[ns,8] ns假设是匹配到target的anchor的数量
                将满足条件/匹配到target的anchor对应的信息（对应的target信息和anchor自身的id信息）提取出来。
                t.shape=[ns,8] ns为匹配到target的anchor的个数， 8==>(image_id,class_score,class_id,x,y,w,h,anchor_id)
                '''
                t = t[valid_idx]  # filter

                # offset
                gxy = t[:, 3:5]  # grid xy, shape=[ns,2]  每行表示一个正样本对应真值目标的xy坐标
                gxy_flip = gain[3:5] - gxy   # inverse, [ns,2] 每行表示一个正样本对应真值目标的xy坐标离右下角的距离

                '''
                  (gxy % 1. < g): 以每个格子左上角为原点,水平，垂直方向为x,y轴的坐标系。判断每个目标方框的中心坐标x',y'(即gxy中一行元素)，在它所在的这个格子的坐标系中是否满足0<x'<0.5, 0<y'<0.5。
                  (gxy > 1.): 以输入图片左上角为原点的坐标系中(实质就是图片被划分的网格坐标系，尺寸也就是当前特征图的维度)，判断每个目标方框中心坐标x,y是否大于1
    
                  格子的中心是（x+0.5,x+0.5）
                  j：长度为31(ns)的tensor,每个元素表示每个目标方框的x坐标是否满足0<x'<0.5 且 x>1，满足为true,否则为false. true表示这个目标在网格坐标系中x大于1并且在目标所在格子中处于格子中心的左边
                  k：长度为31(ns)的tensor，每个元素表示每个目标方框的y坐标是否满足0<y'<0.5 且 y>1，满足为true,否则为false. true表示这个目标在网格坐标系中y大于1并且在目标所在格子中处于格子中心的上边
                  l: 长度为31的tensor,实质是x<w-1,且在目标所在格子中心的右边的目标为true，其余为false
                  m: 长度为31的tensor,实质是y<h-1,且在目标所在格子中心的下边的目标为true，其余为false
                '''
                j, k = ((gxy % 1. < self.expansion_bias) & (gxy > 1.)).T
                l, m = ((gxy_flip % 1. < self.expansion_bias) & (gxy_flip > 1.)).T
                '''
                   gain_valid_idx(shape=[5,ns])
                   这里是将前面由j,k,l,m提取出来的目标的真实标签与原来的目标的方框真实标签堆叠起来，增加目标数量
                '''
                gain_valid_idx = torch.stack([torch.ones_like(j), j, k, l, m])   # shape=[5,ns]
                t = t.repeat((5, 1, 1))[gain_valid_idx]    # t.repeat((5,1,1)).shape=[5,ns,7]   t.shape=[nj,7]
                '''
                   off = [[0,0], [1, 0], [0, 1], [-1, 0], [0, -1]]
                   offsets的计算是以当前yolo层的特征图网格为坐标系
                   z（=torch.zeros_like(gxy)[None]）是原始目标的偏移量，为0. 
                   z[.]+off[0], 不移动
                   z[j]+off[0], 将处于格子中心左边的目标向右移动1*g
                   z[k]+off[1], 将处于格子中心上边的目标向下移动1*g
                   z[l]+off[2], 将处于格子中心右边的目标向左移动1*g
                   z[m]+off[3], 将处于格子中心下边的目标向上移动1*g
               
               重点：对于每个分配了anchor的目标，将它们在格子中相对于格子中心的位置分为原点、上、下、左、右
                    然后将这些目标按照这个相对位置的相反方向移动1，类似于目标抖动，增加了目标的数量
                    这么做的理由是什么？？？
                '''
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[gain_valid_idx]    # offsets.shape=[nj,2]
            else:
                t = targets[0]
                offsets = 0

            # Define
            # b是一个长度为nj的tensor,每个元素表示一个目标所处的图片在这个batch中的序号
            # c是一个长度为nj的tensor,每个元素表示一个目标的类别
            # a是一个长度为nj的tensor,每个元素表示一个anchor id
            b, c, a = t[:, [0, 2, 7]].long().T
            w = t[:, 1]  # class_score

            # grid xy 维度：njx2，所有目标的真实xy坐标(在当前特征图的网格坐标系下)
            # grid wh 维度：njx2，所有目标的真实wh值(在当前特征图的网格坐标系下)
            gt_xy = t[:, 3:5]
            gt_wh = t[:, 5:7]
            gij = (gt_xy - offsets).long()  # 匹配到targets的anchor中心所处的格子的左上角坐标xy
            # gi,gj都是长度为nj的一维tensor，表示每个被分配了目标的anchor所在格子的左上角坐标
            gi, gj = gij.T    # grid xy indices

            # target_indices用于定位到被分配了目标的anchor在当前yolo层的特征图上的位置，用b,a,gj,gi可以唯一确定一个anchor在特征图上的位置
            target_indices.append((b, a, gj, gi))   # image, anchor, grid indices
            '''
              tbox中存放的是被分配了目标的anchor对应的它真正需要学习的目标tx,ty,gw,gh,其中
              tx = gx - gi, gx是这个anchor对应的目标方框中心x坐标，gi是匹配到该目标的anchor中心所在格子的左上角x坐标
              ty = gy - gj, gy是这个anchor对应的目标方框中心y坐标，gi是匹配到该目标的anchor中心所在格子的左上角y坐标
              对于x,y坐标，网络需要学习的输出就是，被分配了目标的anchor所对应的目标中心距离它所在格子左上角的偏移量tx,ty

              gwh:对于目标的宽高，网络需要学习的输出就是被分配了目标的anchor所对应的目标的宽、高w,h

              注意：这里的tx,ty,w,h的计算都是将真实标签的坐标缩放到当前yolo层的特征维度上进行计算的
            '''
            target_box.append(torch.cat((gt_xy - gij, gt_wh), dim=1))    # box
            # anchors[a]保存的是被分配了目标的anchor在当前yolo层的w,h值
            target_anchors.append(anchors[a])
            # tcls保存的是被分配了目标的anchor对应目标的真实类别
            target_weights_cls.append((w, c))
        '''
          indices (list,len=3) : one element of indices is (b,a,gj,gi)  b/a/gj/gi.shape=[nj]
          tbox (list,len=3) :   one element of tbox is (dx,dy,w,h)  shape=[nj,4]
          anch (list,len=3) :   one element of anch is anchors( shape=[nj,2])  已经调整到对应的featuremap尺度表示
          tcls (list,len=3) :    one element's shape is [nj]
          '''

        '''
        tcls: indices是一个list,包含三个tensor，每个tensor的元素个数是93, 135, 135，对应3个yolo层中被分配了目标的anchor对应目标的类别
        tbox: 包含三个元素的list，维度是[93x4, 135x4, 135x4], 93, 135, 135 表示3个yolo层各自有93，135，135个anchor被分配了目标，tbox中保存了这些anchor对应目标方框的tx,ty,gw,gh。这是网络需要学习的值
        indices: 包含三个元素的list, 每个元素是一个元组(b,a,gj,gi),代表一个yolo层中分配有目标的anchor在当前yolo层中的特征图里面的位置索引。根据这些索引，可以定位到分配有目标的anchor在当前yolo层
              的特征网格中的位置以及所在格子的左上角坐标
          b: 分配有目标的anchor的目标所在图片在这个batch中的序号
          a: 分配有目标的anchor在3个anchor中的序号,值为0,1,2中的一个
          gj: 分配有目标的anchor在当前yolo层中的特征网格里面，其中心所在格子的左上角y坐标
          gi: 分配有目标的anchor在当前yolo层中的特征网格里面，其中心所在格子的左上角y坐标
        '''
        return target_weights_cls, target_box, target_indices, target_anchors







class YOLOv5LossOriginal(object):
    def __init__(self,
                 ratio_thresh=4.,
                 expansion_bias=0.5,
                 layer_balance=None,
                 cls_pw=1.0,
                 obj_pw=1.0,
                 iou_type="ciou",
                 coord_type="xywh",
                 iou_ratio=1.0,
                 iou_weights=0.05,
                 cls_weights=0.5,
                 obj_weights=1.0,
                 fl_gamma=0.,
                 class_smoothing_eps=0.0):
        '''
        :param ratio_thresh: 标签的长h宽w/anchor的长h_a宽w_a阈值, 即h/h_a, w/w_a都要在(1/2.26, 2.26)之间
        :param expansion_bias: anchor匹配时的centrol point位置偏移，能使同一个anchor匹配到更多的样本
        :param layer_balance: objectness loss(lobj)置信度损失在不同的层有不同的权重系数, [4.0, 1.0, 0.4] if np == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
        :param cls_pw:  cls BCELoss positive_weight/分类BCELoss中正样本的权重
        :param obj_pw:  obj BCELoss positive_weight/ 有无物体BCELoss中正样本的权重
        :param iou_type:
        :param coord_type:
        :param iou_ratio:   which is used to calcu objectness by calcu iou
        :param iou_weights:  box loss gain
        :param cls_weights:  cls loss gain
        :param obj_weights:  obj loss gain (scale with pixels)/有无物体损失的系数
        :param fl_gamma: hyper-parameter gamma in focal loss
        :param class_smoothing_eps: hyper-parameter in class smoothing label
        '''
        super(YOLOv5LossOriginal, self).__init__()
        if layer_balance is None:
            layer_balance = [4.0, 1.0, 0.4]
        self.layer_balance = layer_balance
        self.iou_ratio = iou_ratio
        self.iou_weights = iou_weights
        self.cls_weights = cls_weights
        self.obj_weights = obj_weights
        self.expansion_bias = expansion_bias
        self.box_similarity = BoxSimilarity(iou_type, coord_type)  # calculate iou between pred_bbox and gt_bbox
        self.target_builder = YOLOv5Builder(ratio_thresh, expansion_bias)  # Build targets for compute_loss/ for anchors
        self.cls_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(data=[cls_pw]))  # classfication loss
        self.obj_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(data=[obj_pw]))  # predicted objectness loss
        ### focal loss wraper----------------------------------------------------------------
        self.fl_gamma=fl_gamma
        if fl_gamma>0:   # focal loss gamma
            self.cls_bce=FocalLoss(self.cls_bce,fl_gamma)
            self.obj_bce=FocalLoss(self.obj_bce,fl_gamma)
        # 样本标签平滑
        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=class_smoothing_eps)

    def __call__(self, predicts, targets, input_anchors):
        '''

        :param predicts (list, len=nl, nl=3 here): predicts[i].shape=(bs,3,ny,nx,85)
        :param targets (torch.Tensor shape=[N,7]):  7==>(bs_idx,weights,label_idx,x1,y1,x2,y2), note: [x1,y1,x2,y2] has been normalized between 0 and 1.
        :param input_anchors (torch.Tensor shape=[3,3,2]): 2==>[w,h] has been normalized to cooresponding featuremap size.
        :return:
        loss
        '''
        device = targets.device
        if self.fl_gamma==0.0:
            if self.cls_bce.pos_weight.device != device:
                self.cls_bce.to(device)
            if self.obj_bce.pos_weight.device != device:
                self.obj_bce.to(device)
        else:
            if self.cls_bce.loss_fcn.pos_weight.device != device:
                self.cls_bce.to(device)
            if self.obj_bce.loss_fcn.pos_weight.device != device:
                self.obj_bce.to(device)

        # 分类损失 边框预测损失 置信度损失
        loss_cls, loss_box, loss_obj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

        '''
           targets_indices (list,len=3) : one element of indices is (b,a,gj,gi) (which consist of 4 list,shape=[nj])
           targets_box (list,len=3) :   one element of tbox is (x,y,w,h)  shape=[nj,4]
           targets_anchors (list,len=3) :   one element of anch is anchors( shape=[nj,2])  已经调整到对应的featuremap尺度表示
           targets_weights_cls (list,len=3) :    one element's shape is [nj]
        '''
        targets_weights_cls, targets_box, targets_indices, targets_anchors = self.target_builder(predicts, targets, input_anchors)

        num_cls = predicts[0].shape[-1] - 5
        num_bs = predicts[0].shape[0]
        match_num = 0  # number of targets
        layer_num = len(predicts)   # number of outputs, layer_num=3 here

        for i, pi in enumerate(predicts):    # layer index, layer predictions
            '''
            这里b,a,gj,gi代表的是第i层yolo层与真实方框对应的那些anchor的下标，
            b就是这些anchor所在的图片在这个batch中的序号
            a就是这些anchor在3个anchor中的序号
            gj,gi就是这个anchor所在格子左上角坐标。
            通过b,a,gj,gi可以定位到yolo层中分配有目标的anchor的位置以及它们所在格子左上角坐标
            '''
            b, a, gj, gi = targets_indices[i]   # image, anchor, gridy, gridx
            target_obj = torch.zeros_like(pi[..., 0], device=device)    # target obj, which is used to calcu objectness
            n = len(b)    # number of targets, n=nj

            if n:
                match_num += n   # cumulative targets
                ps = pi[b, a, gj, gi]   # prediction subset corresponding to targets, shape=[nj,85]

                # regression
                '''
                   xy=2*sigmoid(xy_pred)-0.5
                   wh=(2*sigmoid(wh_pred))**2*anchor_wh
    
                   在training的时候，Detect层没有将输出值转换为预测值，而是在计算loss的时候才将输出值转换为预测值。
                   inference的时候则是在Detect中直接转换为预测值，并且inference中计算出来bx,by会加上cx,cy并转换到输入图片的尺寸上，training的时候没有
                '''
                '''
                why xy=2*sigmoid(xy_pred)-0.5 ? because gt_xy~(-0.5,1.5)
                why wh=(2*sigmoid(wh_pred))**2*anchor_wh? 直接预测在当前尺度的特征图上的bbox
                '''

                pxy = ps[:, :2].sigmoid() * 2. - self.expansion_bias
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * targets_anchors[i]
                # pbox 当前尺度的特征图上预测的bbox
                pbox = torch.cat((pxy, pwh), dim=1).to(device)   # predicted box
                box_sim = self.box_similarity(pbox, targets_box[i])  # iou(prediction, target)
                loss_box += (1.0 - box_sim).mean()   # iou loss

                # Objectness/ 真值置信度/confidence score=(1.0 - model.gr) + model.gr * iou
                # 此时得到了与target对应的那些anchor的confidence的真实标签，这里计算confidence的真实目标采用了与yolov1相似的方法，计算真实方框与预测方框的iou作为是否含有目标的优化目标，只不过这里采用的是giou。其余的anchor的confidence为0
                # 问：用预测值方框与真实方框的giou作为confidence的目标值，这种操作为什么可以学习到是否有目标？？？预测出来的结果很差的话giou算出出来很小，那最终学习到的confidence值不会很小吗？？？
                target_obj[b, a, gj, gi] = \
                    (1.0 - self.iou_ratio) + self.iou_ratio * box_sim.detach().clamp(0).type(target_obj.dtype)

                if num_cls > 1:  # cls loss (only if multiple classes)
                    # t = torch.zeros_like(ps[:, 5:], device=device)  # targets, shape=[nj,80]
                    # note: use label smoothing here
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets, shape=[nj,80]
                    t[range(n), targets_weights_cls[i][1]] = self.cp
                    loss_cls += self.cls_bce(ps[:, 5:], t)  # BCE
            loss_obj += self.obj_bce(pi[..., 4], target_obj) * self.layer_balance[i]   # obj loss

        s = 3 / layer_num   # output count scaling
        cls_weights = self.cls_weights * (num_cls / 80.)
        loss_box *= self.iou_weights * s
        loss_obj *= self.obj_weights * s
        loss_cls *= cls_weights * s
        loss = loss_box + loss_obj + loss_cls
        return loss * num_bs, torch.cat((loss_box, loss_obj, loss_cls, loss)).detach(), match_num


