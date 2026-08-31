import math
import os
import yaml
from collections import defaultdict
from typing import Any, Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.special.cython_special import y1
from torch.utils.data import Dataset

from src.utils.data_manipulations import data_to_tensor


class VisDroneSample:
    def __init__(
        self,
        filename: str,
        image: np.ndarray,
        labels: List[List[float]],

        **kwargs
    ):
        self.filename = filename
        self.image = image
        self.labels = labels

        self.suppl_info =  dict(kwargs)

    def to_dict(self):
        return {
            "filename": self.filename,
            "image": self.image,
            "labels": self.labels,
            "suppl_info": self.suppl_info
        }

    @classmethod
    def from_dict(cls, dict_obj: Dict[str, Any]):
        return VisDroneSample(
            filename=dict_obj["filename"],
            image=dict_obj["image"],
            labels=dict_obj["labels"],
            **dict_obj.get("suppl_info", {})
        )


class VisDroneDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split_type: str,
        split_filename: Optional[str] = None,
        single_class_detection: bool = False,
        class_config_path: Optional[str] = None,
    ):
        self.data_path = data_path
        self.single_class_detection = single_class_detection

        self._samples = []

        if split_type not in ["test", "train", "val"]:
            raise ValueError(f"split_type {split_type} is not supported")
        self.split_type = split_type

        self._split_file_ids = None
        if split_filename is not None:
            with open(split_filename, "r") as f:
                self._split_file_ids = [line.strip() for line in f.readlines()]

        # label class mapping direct from VisDrone documentation
        if class_config_path is not None:
            with open(class_config_path, "r") as f:
                class_config = yaml.load(f, yaml.SafeLoader)
            self.class_mapping = class_config.get("class_map", None)
            self.label_idx_to_id = class_config.get("label_idx_to_id", dict())
        else:
            self.class_mapping = None
            self.label_idx_to_id = {
                0: 'pedestrian',
                1: 'people',
                2: 'bicycle',
                3: 'car',
                4: 'van',
                5: 'truck',
                6: 'tricycle',
                7: 'awning-tricycle',
                8: 'bus',
                9: 'motorcycle'
            }

        self._load_samples()

    def _load_samples(self):
        image_folder = os.path.join(self.data_path, "images", self.split_type)
        label_folder = os.path.join(self.data_path, "labels", self.split_type)
        filenames = os.listdir(image_folder)
        for filename in filenames:
            file_id = os.path.join(self.data_path, f"images/{self.split_type}/", filename)
            if not self._split_file_ids or (file_id in self._split_file_ids):
                image = cv2.imread(file_id)
                label_file = os.path.splitext(filename)[0] + ".txt"
                labels = []
                with open(os.path.join(label_folder, label_file), "r") as f:
                    for detection in [det.split(" ") for det in f.read().strip().splitlines()]:
                        if detection[4] != 0:
                            cls_id = int(detection[0])
                            cx, cy, w, h = map(float, detection[1:])
                            if cls_id >= 0:
                                if self.class_mapping is not None:
                                    cls_id = self.class_mapping[cls_id]
                                labels.append((cls_id, cx, cy, w, h))
                self._samples.append(
                    VisDroneSample(image=image, labels=labels, filename=filename)
                )

        print(f"Loaded: {len(self._samples)} images")

    @staticmethod
    def _pad_image(img: np.ndarray, mod: int = 32):
        """Pads H, W to multiples of a common factor. Necessary for YOLO Detection model inference"""
        H, W = img.shape[:2]

        pad_h = (math.ceil(H / mod) * mod) - H
        pad_w = math.ceil(W / mod) * mod - W

        # pad bottom and right
        return np.pad(img, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)


    def __len__(self):
        return len(self._samples)

    def plot_detections(self, sample_idx: int):
        image = self._samples[sample_idx].image
        labels = self._samples[sample_idx].labels

        H, W = image.shape[:2]

        for detection in labels:
            class_id, cx, cy, w, h = detection

            # convert to x1, y1, x2, y2 coordinates
            x1 = int((cx - w/2) * W)
            x2 = int((cx + w/2) * W)
            y1 = int((cy - h/2) * H)
            y2 = int((cy + h/2) * H)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{self.label_idx_to_id[class_id]}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        plt.figure(figsize=(10, 7))
        plt.imshow(image)
        plt.show()

    @staticmethod
    def convert_yolo_to_xyxy(labels: List[float], img_h: int, img_w: int) -> List[float]:
        """Convert to xyxy here, so we don't have to rely on image size for future calcs."""
        cx, cy, w, h = labels

        # convert to x1, y1, x2, y2 coordinates
        x1 = (cx - w/2) * img_w
        y1 = (cy - h/2) * img_h
        x2 = (cx + w/2) * img_w
        y2 = (cy + h/2) * img_h

        return [x1, y1, x2, y2]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image = self.samples[index].image

        labels = self.samples[index].labels
        if self.single_class_detection:
            class_ids = np.ones(len(labels), dtype=int)
        else:
            class_ids = [det[0] for det in labels]
            class_ids = np.array(class_ids)

        bboxes = [self.convert_yolo_to_xyxy(det[1:], *image.shape[:2])
                  for det in labels]
        bboxes = np.array(bboxes).astype(np.float32).reshape((-1, 4))

        image = self._pad_image(image, 32)

        output = {"images": image}
        output["targets"] = {"boxes": bboxes, "labels": class_ids}

        return output

    @staticmethod
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        collated_batch = defaultdict(list)
        for j in range(len(batch)):
            for key in ["images", "targets"]:
                collated_batch[key].append(data_to_tensor(batch[j][key]))

        return collated_batch

    @property
    def num_classes(self) -> int:
        return 2 if self.single_class_detection else len(self.label_idx_to_id)

    @property
    def samples(self) -> List[VisDroneSample]:
        return self._samples


if __name__ == "__main__":
    dataset = VisDroneDataset(
        data_path="./data/VisDrone2019-DET/", split_type="test",
        class_config_path="./config/visdrone_to_coco_config.yaml",
        split_filename="./config/splits/test_split.dat"
    )
    print(dataset[0])

    for idx in range(10):
        dataset.plot_detections(idx)
        input()