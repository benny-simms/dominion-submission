from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps, patches

from src.utils.metrics import DetectionPrediction


def draw_detections(
    image: np.ndarray,
    predictions: Optional[List[DetectionPrediction]] = None,
    ground_truth: Optional[List[Any]] = None,
    font_size: float = 5.0,
    figure_id: Optional[int] = None,
    figure_size_inches: Optional[Tuple[float, float]] = None,
    show: bool = True
):
    """
    Draws detections over a static image
    -----------
    image: input image in H X W X C format
    predictions: list of DetectionPrediction objects detailing the bounding box, confidence, label and correctness
    ground_truth: list of ground_truth objects to overlay an image
    font_size: size of label/confidences
    """
    figure, axes = plt.subplots(num=figure_id)

    figure.patch.set_facecolor("black")
    axes.set_facecolor("black")

    if figure_size_inches is not None:
        figure.set_size_inches(figure_size_inches)

    axes.imshow(image, cmap="gray" if image.ndim == 2 else None)
    axes.set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if predictions is not None:
        for pred in predictions:
            color = "lightyellow" if pred.correct else "lightcoral"

            bbox = pred.bbox.reshape(4)
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                linewidth=2, edgecolor=color, facecolor=color, alpha=0.2
            )
            axes.add_patch(rect)

            label, confidence = pred.label, pred.confidence

            plt.text(bbox[0], bbox[3] + 20, f"Conf: {confidence:2.2f}", color=color, fontsize=font_size)

    if ground_truth is not None:
        cmap = colormaps["tab10"]
        for bbox, label in zip(ground_truth["boxes"], ground_truth["labels"]):
            color = cmap(label)

            rect = patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                linewidth=2, edgecolor=color, facecolor=color, alpha=0.2
            )
            axes.add_patch(rect)

            #plt.text(bbox[0], bbox[3] + 20, f"{label}", color=color, fontsize=font_size)

    if show:
        plt.show()

    return figure
