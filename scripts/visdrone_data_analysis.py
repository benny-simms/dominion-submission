import math
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from src.datasets.visdrone_dataset import VisDroneDataset

if __name__ == "__main__":
    dataset = VisDroneDataset(
        data_path="./data/VisDrone2019-DET/", split_type="test", split_filename="./config/splits/test_split.dat"
    )

    all_detections = [
        (bbox, label)
        for datum in dataset
        for bbox, label in zip(datum["targets"]["boxes"], datum["targets"]["labels"])
    ]

    # plot average object size
    det_areas = [math.sqrt((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) for bbox, _ in all_detections]

    plt.figure(figsize=(6, 4), dpi=300)
    plt.hist(det_areas, bins=100, edgecolor='darkblue')

    plt.xlabel('Pixels Squared')
    plt.ylabel('Frequency')
    plt.title('VisDrone Testing Set: Total Detection Area (Pixels Squared)')
    plt.show()

    # plot class label distribution
    class_labels = [label for _, label in all_detections]
    class_counts = Counter(class_labels)

    categories = list(class_counts.keys())
    categories = [dataset.label_idx_to_id[key] for key in categories]
    frequencies = list(class_counts.values())
    cmap = plt.get_cmap("tab10")
    colors = cmap(np.linspace(0, 1, len(categories)))

    plt.figure(figsize=(6, 4), dpi=300)
    plt.bar(categories, frequencies, color=colors, edgecolor="black")
    plt.tick_params(axis="x", labelsize=6)

    plt.ylabel("Frequency")
    plt.title("VisDrone Testing Set: Distribution of Detection Classes")
    plt.show()




