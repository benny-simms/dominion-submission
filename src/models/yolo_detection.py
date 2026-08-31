import os
from typing import Any, Dict, List, Mapping, Optional

import torch
from torch import Tensor
from torch.nn.modules import Module

os.environ["YOLO_VERBOSE"] = "0"
import ultralytics
from ultralytics.models.yolo.model import YOLO

from src.models.utils import overlapping_detections_merging, overlapping_detections_suppression

YOLO_SAFE_GLOBALS = [
    torch.nn.modules.activation.SiLU,
    torch.nn.modules.batchnorm.BatchNorm2d,
    torch.nn.modules.container.ModuleList,
    torch.nn.modules.container.Sequential,
    torch.nn.modules.conv.Conv2d,
    torch.nn.modules.pooling.MaxPool2d,
    torch.nn.modules.upsampling.Upsample,
    ultralytics.nn.modules.block.Bottleneck,
    ultralytics.nn.modules.block.C2f,
    ultralytics.nn.modules.block.C3k2,
    ultralytics.nn.modules.block.DFL,
    ultralytics.nn.modules.block.SPPF,
    ultralytics.nn.modules.conv.Concat,
    ultralytics.nn.modules.conv.Conv,
    ultralytics.nn.modules.head.Detect,
    ultralytics.nn.tasks.DetectionModel
]

class YOLODetection(Module):
    "Wrapper around the ultralytics YOLO model. This model can only be used for inference and will crash in training"

    def __init__(
        self,
        model_path: str,
        class_config: Optional[Dict[int, str]],
        box_iou_threshold: float = 0.7,
        box_ioa_threshold: float = 0.25,
        suppress_overlapping_bboxes: bool = True,
        merge_overlapping_bboxes: bool = False,
        single_class_detection: bool = False
    ):
        super().__init__()

        if suppress_overlapping_bboxes and merge_overlapping_bboxes:
            raise ValueError(f"Choose only one strategy for overlapping bounding boxes.")

        self.class_config = class_config
        self.single_class_detection = single_class_detection

        self.training = False
        self.box_iou_threshold = box_iou_threshold
        self.box_ioa_threshold = box_ioa_threshold
        self.suppress_overlapping_bboxes = suppress_overlapping_bboxes
        self.merge_overlapping_bboxes = merge_overlapping_bboxes

        with torch.serialization.safe_globals(YOLO_SAFE_GLOBALS):
            self.model = YOLO(model_path, verbose=False)

    def __call__(self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None):
        with torch.no_grad():
            return self.forward(images, targets)

    def train(self, mode: bool = True) -> Module:
        self.training = mode
        self.model.model.train(mode)

        return self

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False) -> None:
        self.model.load_state_dict(state_dict, strict, assign)

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        return self.model.state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)

    def forward(self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None):
        images = [image.unsqueeze(2).repeat(1, 1, 3) if len(image.shape) == 2 else image for image in images]
        images = [image.permute(2, 0, 1) if image.shape[2] == 3 else image for image in images]
        images = torch.cat([image.unsqueeze(0) for image in images])
        res = self.model(images, verbose=False, iou=self.box_iou_threshold)

        num_images = len(res)
        detections = []

        # post-process detections and coerce output into xyxy format
        for idx in range(num_images):
            boxes = res[idx].boxes.xyxy.clone()
            scores = res[idx].boxes.conf.clone()
            labels = res[idx].boxes.cls.to(torch.int32).clone()

            if self.class_config is not None:
                # subset detections not in the class config, useful for pretrained models
                valid_label_idx = torch.nonzero(
                    torch.tensor([val in self.class_config.keys() for val in labels.tolist()], dtype=torch.bool)
                ).squeeze(-1)
                boxes = boxes[valid_label_idx, :]
                scores = scores[valid_label_idx]
                labels = labels[valid_label_idx]

            if self.suppress_overlapping_bboxes:
                boxes, scores, labels = overlapping_detections_suppression(boxes, scores, labels, self.box_ioa_threshold)
            if self.merge_overlapping_bboxes:
                boxes, scores, labels = overlapping_detections_merging(boxes, scores, labels, self.box_ioa_threshold)


            if self.single_class_detection:
                labels = torch.ones(len(labels), dtype=int)

            detections.append({"boxes": boxes, "labels": labels, "scores": scores})

        return {"detections": detections}


if __name__ == "__main__":
    import cv2

    from src.utils.data_manipulations import data_to_tensor
    import torchvision.transforms.functional as F

    model = YOLODetection(
        model_path="./models/yolo26-detection/yolo26n.pt", class_config={0: "person"})

    test_img_path = "./test_img.png"
    img = cv2.imread(test_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1)
    img = F.resize(img, (1024, 1024))
    img = img.unsqueeze(0)

    img = data_to_tensor(img)

    out = model(img)
    print(out)

    out = model
