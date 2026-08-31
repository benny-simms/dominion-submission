from typing import Any, Dict, List, Union, Tuple

import numpy as np
import torch
from torch import Tensor
from torch import device as Device


def data_to_device(
    data: Union[Tensor, List[Tensor], Dict[str, Any], Any], device: Device
) -> Union[Tensor, List[Any], Tuple[Any], Dict[str, Any], Any]:
    """
    Put a data structure on specific device.
    Accept Tensor, a List, a Tuple, or a Dict; it works recursively for nested datatype.
    Any other datatype is return as is.

    Returns a Tensor if data was a Tensor, a List if data was a List and a Dict if data was a Dict.
    """
    if isinstance(data, Tensor):
        return data.to(device)
    elif isinstance(data, list):
        out_list = []
        for i in range(len(data)):
            out_list.append(data_to_device(data[i], device))
        return out_list
    elif isinstance(data, tuple):
        return tuple(data_to_device(datum, device) for datum in data)
    elif isinstance(data, dict):
        out_dict = dict()
        for k, v in data.items():
            out_dict[k] = data_to_device(v, device)
        return out_dict
    else:
        return data


def data_to_tensor(
    data: Union[np.ndarray, List[Any], Dict[str, Any], Any],
) -> Union[Tensor, List[Any], Tuple[Any], Dict[str, Any], Any]:
    """
    Transform a numpy data structure to a torch.Tensor.
    Accept np.ndarray, a List, a Tuple, or a Dict; it works recursively for nested datatype.
    Any other datatype is return as is.

    Returns a Tensor if data was a np.ndarray, a List if data was a List, a Tuple if data was a Tuple, and a Dict if
    data was a Dict.
    """
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, list):
        out_list = []
        for i in range(len(data)):
            out_list.append(data_to_tensor(data[i]))
        return out_list
    elif isinstance(data, tuple):
        return tuple(data_to_tensor(datum) for datum in data)
    elif isinstance(data, dict):
        out_dict = dict()
        for k, v in data.items():
            out_dict[k] = data_to_tensor(v)
        return out_dict
    else:
        return data


def data_to_numpy(
    data: Union[Tensor, List[Any], Tuple[Any], Dict[str, Any], Any],
) -> Union[np.ndarray, List[Any], Tuple[Any], Dict[str, Any], Any]:
    """
    Transform a torch.Tensor to a numpy data structure.
    Accept Tensor, a List, a Tuple, or a Dict; it works recursively for nested datatype.
    Any other datatype is return as is.

    Returns a np.ndarray if data was a Tensor, a List if data was a List, a Tuple if data was a Tuple, and a Dict if
    data was a Dict.
    """
    if isinstance(data, Tensor):
        return data.detach().numpy()
    elif isinstance(data, list):
        out_list = []
        for i in range(len(data)):
            out_list.append(data_to_numpy(data[i]))
        return out_list
    elif isinstance(data, tuple):
        return tuple(data_to_numpy(datum) for datum in data)
    elif isinstance(data, dict):
        out_dict = dict()
        for k, v in data.items():
            out_dict[k] = data_to_numpy(v)
        return out_dict
    else:
        return data
