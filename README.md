# Yolov5 

yolov5: a pytorch implement version  based on liangheming's YOLO_series. we add autonomous anchor cluster  and coco map computation to this repo. 

## requirement

```
tqdm
pyyaml
numpy
opencv-python
pycocotools
torch >= 1.6
torchvision >=0.7.0
```

## result

we train yolov5s on BDD100 dataset. Finally, yolov5s in this repo achieves 54.0 mAP.

```python
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  all | maxDets=100 ] = 0.540
 Average Precision  (AP) @[ IoU=0.50    | area=  all | maxDets=100 ] = 0.722
 Average Precision  (AP) @[ IoU=0.75    | area=  all | maxDets=100 ] = 0.608
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.369
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.636
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.798
 Average Recall   (AR) @[ IoU=0.50:0.95 | area=  all | maxDets=  1 ] = 0.348
 Average Recall   (AR) @[ IoU=0.50:0.95 | area=  all | maxDets= 10 ] = 0.679
 Average Recall   (AR) @[ IoU=0.50:0.95 | area=  all | maxDets=100 ] = 0.708
 Average Recall   (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.611
 Average Recall   (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.801
 Average Recall   (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.901
```



## training

for now we support coco dataset and BDD100 dataset

```PYTHON
 python -m torch.distributed.launch --nproc_per_node=4 main.py
```

## tricks

-  Color Jitter
-  Perspective Transform
-  Mosaic Augment
-  MixUp Augment
-  IOU GIOU DIOU CIOU
-  Warming UP
-  Cosine Lr Decay
-  EMA(Exponential Moving Average)
-  Mixed Precision Training
-  Sync Batch Normalize
-  PANet(neck)
-  autonomous anchor cluster