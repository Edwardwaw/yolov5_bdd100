from processors.ddp_apex_processor import COCODDPApexProcessor
# from processors.ddp_apex_processor_singlegpu import COCODDPApexProcessor


# python -m torch.distributed.launch --nproc_per_node=4 main.py
if __name__ == '__main__':
    # processor = COCODDPApexProcessor(cfg_path="config/coco_yolov5.yaml")
    processor = COCODDPApexProcessor(cfg_path="config/BDD_yolov5.yaml")
    processor.run()

