import json
import os
import yaml
from shutil import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc
from torch import device as Device
from torch.utils.data.dataloader import DataLoader

from src.utils.factories import initialize_dataset, initialize_model
from src.utils.metrics import (
    eval_detections,
    image_report_metrics,
    mean_average_precision,
    plot_precision_recall_curve
)
from src.utils.plot import draw_detections
from src.utils.process_data import evaluate


def main(eval_config_path: str):
    if eval_config_path:
        with open(eval_config_path, 'r') as f:
            config = yaml.load(f, yaml.SafeLoader)
    else:
        config = dict()

    use_cpu = config.get("misc", dict()).get("use_cpu", False)
    display_detections = config.get("misc", dict()).get("display_detections", False)
    device = Device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config_model = config.get("model", dict())
    model_type = config_model.pop("type")
    weights_file_path = config_model.pop("weights_file_path", None)

    config_dataset = config.get("dataset", dict())
    dataset_type = config_dataset.pop("type")

    dataset = initialize_dataset(dataset_type, "test", config_dataset)

    config_dataloader = config.pop("dataloader")

    dataloader = DataLoader(
        dataset, shuffle=False, collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None,
        **config_dataloader
    )

    model = initialize_model(
        model_name=model_type,
        class_config=dataset.label_idx_to_id,
        config=config_model,
        device=device,
        weights_file_path=weights_file_path,
    )
    print(f"Evaluating model: {model_type} on dataset: {dataset_type}")

    outputs = evaluate(device, model, dataloader)

    output_dir = config.get("misc", dict()).get("detections_display_output_dir")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "detections"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "ground_truth"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)
    copy(eval_config_path, os.path.join(output_dir, os.path.basename(eval_config_path)))

    beta = config.get("misc", dict()).get("beta", 1.0)
    human_review_threshold = config.get("misc", dict()).get("human_review_threshold", 0.95)

    labels_given = all([datum.get("targets") is not None for datum in dataset])

    if labels_given:
        mAP, mAP_50, mAP_75 = mean_average_precision(
            outputs["detections"], [datum["targets"] for datum in dataset]
        )

        res = eval_detections(
            outputs["detections"], [datum["targets"] for datum in dataset],  beta, human_review_threshold,
        )
        metrics = res["metrics"]
        precision = metrics["precision"]
        recall = metrics["recall"]
        confidences = metrics["confidences"]
        f_scores = metrics["f_scores"]

        op_ix = np.argmax(f_scores)
        confidence_threshold = confidences[op_ix]

        metrics["optimal_confidence_threshold"] = confidence_threshold
        metrics["model_precision"] = precision[-1]
        metrics["model_recall"] = recall[-1]

        metrics["human_review_confidence"] = human_review_threshold
        metrics["mAP"] = mAP
        metrics["mAP_50"] = mAP_50
        metrics["mAP_75"] = mAP_75

        AUC = auc(recall, precision)

        performance_message = (
            f"Without a confidence threshold, "
            f"precision: {precision[-1]:2.2f}, recall: {recall[-1]:2.2f}.\n"
            f"With a confidence threshold of {confidence_threshold:2.2f}, "
            f"precision: {precision[op_ix]:2.2f}, recall: {recall[op_ix]:2.2f}.\n"
            f"This threshold retains {100 * op_ix / len(f_scores):2.2f}% of predictions\n"
            f"F-{beta:2.2f} score: {f_scores[op_ix]:2.2f}, "
            f"Precision-recall curve AUC {AUC:2.2f}.\n"
            f"mAP: {mAP:2.2f}, mAP_50: {mAP_50:2.2f}, mAP_75: {mAP_75:2.2f}"
        )
        print(performance_message)

        with open(os.path.join(output_dir, "performance.log"), "wt") as outfile:
            outfile.write(performance_message)
        with open(os.path.join(output_dir, "metrics.json"), "w") as outfile:
            json.dump(metrics, outfile, indent=4)

        precision_recall_curve = plot_precision_recall_curve(precision, recall, show=display_detections)

        if not display_detections:
            precision_recall_curve.savefig(os.path.join(output_dir, "precision_recall_curve.png"))

    for idx, (datum, prediction) in enumerate(zip(dataset, res["predictions"])):
        # display detections
        figure = draw_detections(
            image=datum["images"],
            predictions=prediction,
            show=display_detections,
            # figure_size_inches=(datum["images"].shape[1] / 500, datum["images"].shape[0] / 500)
        )
        if not display_detections:
            plt.savefig(os.path.join(output_dir, "detections", f"{idx:05d}.png"), dpi=500)
            plt.close(figure.number)

        # display ground truth
        figure = draw_detections(
            image=datum["images"],
            ground_truth=datum["targets"],
            show = display_detections,
            # figure_size_inches = (datum["images"].shape[1] / 500, datum["images"].shape[0] / 500)
        )

        if not display_detections:
            plt.savefig(os.path.join(output_dir, "ground_truth", f"{idx:05d}.png"), dpi=500)
            plt.close(figure.number)

        # write report metrics for each image
        report_metrics = image_report_metrics(prediction, datum["targets"])
        with open(os.path.join(output_dir, "reports", f"{idx:05d}.json"), "wt") as outfile:
            json.dump(report_metrics, outfile, indent=4)


if __name__ == "__main__":
    main(eval_config_path="./config/eval_detection_yolo.yaml")
