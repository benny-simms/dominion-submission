import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from src.utils.data_manipulations import data_to_tensor

Numeric = int | float

@dataclass
class DetectionPrediction:
    bbox: Tensor
    label: int
    correct: bool
    confidence: float


def mean_average_precision(
        detections: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]
) -> Tuple[float, float, float]:
    metric = MeanAveragePrecision()

    metric.update(data_to_tensor(detections), data_to_tensor(ground_truth))
    mAP_values = metric.compute()

    return mAP_values["map"].item(), mAP_values["map_50"].item(), mAP_values["map_75"].item()

def intersection_over_union(gt_bbox, detection_bbox) -> float:
    x1 = max(gt_bbox[0], detection_bbox[0])
    y1 = max(gt_bbox[1], detection_bbox[1])
    x2 = min(gt_bbox[2], detection_bbox[2])
    y2 = min(gt_bbox[3], detection_bbox[3])

    intersection_width = x2 - x1 + 1
    intersection_height = y2 - y1 + 1

    # reject if non-overlapping
    if intersection_width <= 0 or intersection_height <= 0:
        return 0

    intersection_area = intersection_width * intersection_height
    gt_box_area = (gt_bbox[2] - gt_bbox[0] + 1) * (gt_bbox[3] - gt_bbox[1] + 1)
    det_box_area = (detection_bbox[2] - detection_bbox[0] + 1) * (detection_bbox[3] - detection_bbox[1] + 1)
    iou = intersection_area / float(gt_box_area + det_box_area - intersection_area)

    return iou

def match_detections_and_ground_truths(detections: np.ndarray, ground_truths: np.ndarray, iou_threshold=0.75):
    detections = sorted(detections, key=lambda x: x[0], reverse=True)

    gt_matched = [False] * len(ground_truths)

    prediction_objects = []

    for det in detections:
        score = det[0]
        det_box = det[1:]
        det_cls = int(det[-1]) if len(det) > 5 else 1

        best_iou = 0
        best_gt_idx = None

        for gt_idx, gt in enumerate(ground_truths):
            iou = intersection_over_union(gt, det_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        corr = False
        if best_iou > iou_threshold:
            if not gt_matched[best_gt_idx]:
                corr = True
                gt_matched[best_gt_idx] = True

        prediction_objects.append(
            DetectionPrediction(
                bbox=torch.tensor(det_box, dtype=torch.float32),
                label=det_cls,
                correct=corr,
                confidence=score
            )
        )

    return prediction_objects

import numpy as np

def precision_recall(
        predictions: List[List[DetectionPrediction]], n_ground_truth: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_predictions = [pred for img in predictions for pred in img]

    all_predictions = sorted(all_predictions, key=lambda p: p.confidence, reverse=True)

    TP = np.array([1 if p.correct else 0 for p in all_predictions])
    FP = 1 - TP

    tp_cum = np.cumsum(TP)
    fp_cum = np.cumsum(FP)

    recall = tp_cum / n_ground_truth
    precision = tp_cum / (tp_cum + fp_cum)

    confidences = np.array([p.confidence for p in all_predictions])

    return precision, recall, confidences

def eval_detections(
        detections: List[Dict[str, Any]], ground_truths: List[Dict[str, Any]], beta: float = 1.0
) -> Dict[str, Any]:
    det_bboxes = [b["boxes"] for b in detections]
    det_scores = [b["scores"] for b in detections]
    detections = []
    for ex in range(len(det_bboxes)):
        ex_bb = []
        for idx, bb in enumerate(det_bboxes[ex]):
            ex_bb.append(np.insert(bb, 0, det_scores[ex][idx]))
        detections.append(ex_bb)

    ground_truths = [gt["boxes"] for gt in ground_truths]
    n_ground_truth = sum([len(gt) for gt in ground_truths])

    all_predictions = []
    for det, gt in zip(detections, ground_truths):
        predictions = match_detections_and_ground_truths(np.asarray(det), np.asarray(gt))
        all_predictions.append(predictions)

    precision, recall, confidences = precision_recall(all_predictions, n_ground_truth)
    f_scores = f_score(precision, recall, beta)


    metrics = {"precision": precision.tolist(), "recall": recall.tolist(),
               "confidences": confidences.tolist(), "f_scores": f_scores.tolist()}
    outputs = {"predictions": all_predictions, "metrics": metrics}

    return outputs

def f_score(precision: np.ndarray, recall: np.ndarray, beta: float = 1.0) -> np.ndarray:
    return (1 + beta ** 2) * precision * recall / ((beta ** 2) * precision * recall + np.finfo(np.float32).eps)

def plot_precision_recall_curve(precision: np.ndarray, recall: np.ndarray, show: bool= False) -> plt.Figure:
    figure, axes = plt.subplots()
    axes.plot(recall, precision, color="red")
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    axes.set_title("Precision-Recall Curve Plot")
    axes.set_xlabel("Recall")
    axes.set_ylabel("Precision")
    figure.tight_layout()

    if show:
        plt.show()
    return figure

def image_report_metrics(predictions: List[DetectionPrediction], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """
    Writes an individual report file for each image, detailing
    true and false positive/negative predictions
    """
    metrics = {}
    metrics["n_ground_truth"] = len(ground_truth["labels"])
    metrics["n_detections"] = len(predictions)

    TP = len([pred for pred in predictions if pred.correct])
    FP = len([pred for pred in predictions if not pred.correct])
    FN = len(ground_truth["labels"]) - TP

    precision = TP / (TP + FP + sys.float_info.min)
    recall = TP / (TP + FN + sys.float_info.min)

    metrics["true_positives"] = TP
    metrics["false_positives"] = FP
    metrics["false_negatives"] = FN
    metrics["precision"] = precision
    metrics["recall"] = recall

    true_positive_areas = [((pred.bbox[2] - pred.bbox[0]) * (pred.bbox[3] - pred.bbox[1]))
                            for pred in predictions if pred.correct]
    #metrics["average_tp_area"] = sum(true_positive_areas) / len(true_positive_areas)

    return metrics

