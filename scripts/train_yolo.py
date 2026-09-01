import yaml
from ultralytics import YOLO


def main(model_file_path: str, config_file_path: str, resume_training: bool = False):
    """
    Script for training a YOLO model
    ----------
    model_file_path: path to the pretrained YOLO model
    config_file_path: path to the training configuration file
    resume_training: if true, resumes training from a checkpoint specified in the config
    """

    with open(config_file_path, "r") as f:
        config = yaml.load(f, yaml.SafeLoader)

    model = YOLO(model_file_path)
    model.train(data=config.pop("data_config_file_path"), resume=resume_training, **config["training"])

if __name__ == "__main__":
    main(model_file_path="./models/yolo26-detection/yolo26n.pt",
         config_file_path="./config/finetune_detection_yolo.yaml")
