from typing import Any, Dict, Optional

from torch import device as Device
from torch.nn import Module
from torch.utils.data import Dataset

from src.datasets.visdrone_dataset import VisDroneDataset
from src.models.yolo_detection import YOLODetection


def initialize_dataset(dataset_name: str, split_name: str, config: Dict[str, Any]) -> Dataset:
    if split_name not in ["train, validation", "test"]:
        raise ValueError(f"split_name: {split_name} should be train, validation or test")
    if dataset_name == "vis_drone":
        dataset = VisDroneDataset(**config)
    else:
        raise NotImplementedError(f"dataset_name: {dataset_name} has not been implemented")

    return dataset

def initialize_model(
    model_name: str, class_config: Optional[Dict[int, str]], config: Dict[str, Any], device: Device, weights_file_path: Optional[str] = None
) -> Module:
    if model_name == "yolo":
        model = YOLODetection(class_config=class_config, **config)
    else:
        raise ValueError(f"model_name: {model_name} has not been implemented")

    model.to(device)

    return model
