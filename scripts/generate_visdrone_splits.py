import os

import numpy as np

def generate_random_splits(
    n_ids: int,
    input_folder: str,
    output_folder: str,
    train_split: float = 0.8,
    validation_split: float = 0.1
):
    """
    Generates random splits using the filenames from a folder as unique IDs
    Defaults to a standard 80/10/10 split
    ----------
    n_ids: size of combined train/val/test data samples
    input_folder: data root containing the image/label files
    output_folder: folder to save split file paths
    train_split: proportion of training data, must be between 0 and 1
    validation_split: proportion of validation data, must be between 0 and 1
    """
    os.makedirs(output_folder, exist_ok=True)

    assert 0 < (train_split + validation_split) < 1, "Split values must sum to 1"
    test_split = 1.0 - train_split - validation_split
    N_TRAIN = int(train_split * n_ids)
    N_VALIDATION = int(validation_split * n_ids)
    N_TEST = int(test_split * n_ids)

    train_filenames = np.array(os.listdir(os.path.join(input_folder, "train")))
    val_filenames = np.array(os.listdir(os.path.join(input_folder, "val")))
    test_filenames = np.array(os.listdir(os.path.join(input_folder, "test")))

    train_index_list = np.random.permutation(len(train_filenames))
    train_idx = train_index_list[:N_TRAIN]

    train_split_indices = [os.path.join(input_folder, "train/", filename) for filename in train_filenames[train_idx]]

    with open(os.path.join(output_folder, "train_split.txt"), "w") as file:
        file.writelines([split_idx + "\n" for split_idx in train_split_indices])

    val_index_list = np.random.permutation(len(val_filenames))
    val_idx = val_index_list[:N_VALIDATION]

    val_split_indices = [os.path.join(input_folder, "val/", filename) for filename in val_filenames[val_idx]]

    with open(os.path.join(output_folder, "val_split.txt"), "w") as file:
        file.writelines([split_idx + "\n" for split_idx in val_split_indices])

    test_index_list = np.random.permutation(len(test_filenames))
    test_idx = test_index_list[:N_TEST]

    test_split_indices = [os.path.join(input_folder, "test/", filename) for filename in test_filenames[test_idx]]

    with open(os.path.join(output_folder, "test_split.dat"), "w") as file:
        file.writelines([split_idx + "\n" for split_idx in test_split_indices])


if __name__ == "__main__":
    N_IDS = 1000
    generate_random_splits(N_IDS, "./data/VisDrone2019-DET/images/",
                           "./config/splits/")
