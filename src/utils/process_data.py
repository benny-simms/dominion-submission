from typing import Any, Dict, Optional

import torch
from torch import device as Device
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader

from src.utils.data_manipulations import data_to_device, data_to_numpy


def _process_data(
    device: Device,
    model: Module,
    dataloader: DataLoader,
    optimizer: Optional[Optimizer] = None,
    compute_loss: bool = False,
    gradient_accumulation_steps: int = 1,
) -> Dict[str, Any]:
    """
    Loops through the dataloader batches, pushes the data to the model
    and optimizes (if optimizer not None) the model's weights.
    Returns model outputs and/or loss (depending on the parameters) in the same dictionary.
    """
    if optimizer is not None:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    outputs = dict()
    for batch_ix, batch in enumerate(dataloader):
        if optimizer is None and not compute_loss:
            batch.pop("targets", None)
        batch = data_to_device(batch, device)

        output = model(**batch)

        loss: Optional[Tensor] = output.pop("loss", None)
        if loss is not None:
            loss /= gradient_accumulation_steps
            total_loss += loss.item()

        if optimizer is not None:
            loss.backward()
            if (batch_ix + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        output = data_to_device(output, Device("cpu"))
        output = data_to_numpy(output)
        for k in output.keys():
            if not outputs.get(k):
                outputs[k] = []
            outputs[k].extend(output[k])

    if model.training or compute_loss:
        total_loss /= len(dataloader)
        outputs["loss"] = total_loss

    return outputs


def train(
    device: Device,
    model: Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    gradient_accumulation_steps: int = 1,
) -> Dict[str, Any]:
    """
    Loop through the data, predict an output using the model, evaluate its loss, and optimize the model's weights.
    Returns the total loss.
    """
    return _process_data(
        device, model, dataloader, optimizer, gradient_accumulation_steps=gradient_accumulation_steps
    )


def validate(device: Device, model: Module, dataloader: DataLoader) -> Dict[str, Any]:
    """
    Loop through the data, predict an output using the model, and evaluate its loss.
    Returns the predictions and the total loss.
    """
    with torch.no_grad():
        return _process_data(device, model, dataloader, compute_loss=True)


def evaluate(device: Device, model: Module, dataloader: DataLoader) -> Dict[str, Any]:
    """
    Loop through the data and predict an output using the model.
    Returns the model outputs.
    """
    with torch.no_grad():
        return _process_data(device, model, dataloader)
