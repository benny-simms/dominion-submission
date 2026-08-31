from typing import Tuple

import torch
from torch import Tensor
from torchvision.ops.boxes import box_area


def calculate_ioa(boxes: Tensor) -> Tensor:
    lt = torch.max(boxes[:, None, :2], boxes[:, :2])
    rb = torch.min(boxes[:, None, 2:], boxes[:, 2:])

    intersecting_rectangles = (rb - lt).clamp(min=0)

    intersections = intersecting_rectangles[:, :, 0] * intersecting_rectangles[:, :, 1]
    areas = box_area(boxes)

    ioa = intersections / areas
    ioa.fill_diagonal_(0.0)

    return ioa

def encompassing_bbox(bbox_1: Tensor, bbox_2: Tensor) -> Tensor:
    bboxes = torch.cat((bbox_1[None, :], bbox_2[None, :]), dim=0)

    return torch.cat((torch.min(bboxes, dim=0)[0][:2], torch.max(bboxes, dim=0)[0][2:]), dim=0)

def overlapping_detections_merging(
    boxes: Tensor, scores: Tensor, labels: Tensor, box_ioa_threshold: float = 0.25
) -> Tuple[Tensor, Tensor, Tensor]:
    ioa = calculate_ioa(boxes)
    mask = ioa > box_ioa_threshold

    while torch.any(mask):
        mask_indices = torch.argwhere(mask)
        unique_indices = torch.unique(mask_indices)
        ix_min = unique_indices[torch.argmin(scores[unique_indices])]
        ix_max = torch.argmax(ioa[ix_min, :])

        scores[ix_max] = (scores[ix_min] + scores[ix_max]) / 2.0
        scores = torch.cat((scores[:ix_min], scores[ix_min + 1 :]), dim=0)
        labels = torch.cat((labels[:ix_min], labels[ix_min + 1 :]), dim=0)

        boxes[ix_max] = encompassing_bbox(boxes[ix_max], boxes[ix_min])
        boxes = torch.cat((boxes[:ix_min], boxes[ix_min + 1 :]), dim=0)

        ioa = calculate_ioa(boxes)
        mask = ioa > box_ioa_threshold

    return boxes, scores, labels


def overlapping_detections_suppression(
    boxes: Tensor, scores: Tensor, labels: Tensor, box_ioa_threshold: float = 0.25
) -> Tuple[Tensor, Tensor, Tensor]:
    ioa = calculate_ioa(boxes)

    ix_to_remove = torch.zeros((ioa.shape[0],), dtype=torch.bool)
    mask = ioa.sum(dim=0) > box_ioa_threshold
    while torch.any(mask):
        ix = torch.argwhere(mask)[torch.argmin(scores[mask])]
        ix_to_remove[ix] = True

        ioa[ix, :] = 0
        ioa[:, ix] = 0
        mask = ioa.sum(dim=0) > box_ioa_threshold

    ix_to_keep = torch.logical_not(ix_to_remove)

    boxes = boxes[ix_to_keep]
    scores = scores[ix_to_keep]
    labels = labels[ix_to_keep]

    return boxes, scores, labels
