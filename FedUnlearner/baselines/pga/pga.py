import torch
from typeguard import typechecked
from typing import Dict, List, Union
import time
from copy import deepcopy
import os
from tqdm import tqdm

from .pga_utils import get_model_ref, get_threshold, unlearn
from FedUnlearner.fed_learn import train_local_model
from FedUnlearner.utils import average_weights

"""
PGA
Source:
https://arxiv.org/pdf/2207.05521.pdf &
https://proceedings.mlr.press/v222/nguyen24a/nguyen24a.pdf
"""


def fed_avg(w):
    """
    Returns the average of the weights.
    """
    w_avg = deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


@typechecked
def run_pga(global_model: torch.nn.Module,
            weights_path: str,
            clientwise_dataloaders: Dict[int, torch.utils.data.DataLoader],
            num_clients: int,
            forget_client: List[int],
            optimizer_name: str,
            lr: float,
            model: str,
            dataset: str,
            num_classes: int,
            pretrained: bool,
            num_training_iterations: int,
            num_local_epochs: int,
            device: str,
            num_unlearn_rounds=1,
            num_post_training_rounds=1,) -> torch.nn.Module:

    forget_client_model_path = os.path.join(
        weights_path, f"iteration_{num_training_iterations - 1}", f"client_{forget_client[0]}.pth")
    forget_client_model = deepcopy(global_model)
    forget_client_model.load_state_dict(torch.load(forget_client_model_path))
    start_time = time.time()
    # Reference Model
    model_ref = get_model_ref(global_model=global_model,
                              forget_client_model=forget_client_model,
                              num_clients=num_clients,
                              model=model,
                              dataset=dataset,
                              num_classes=num_classes,
                              pretrained=pretrained,
                              device=device)
    # Threshold to which unlearning is to be performed
    threshold = get_threshold(model_ref=model_ref,
                              model=model,
                              dataset=dataset,
                              num_classes=num_classes,
                              pretrained=pretrained,
                              device=device)

    unlearned_global_model = unlearn(
        global_model=global_model,
        forget_client_model=forget_client_model,
        model_ref=model_ref,
        distance_threshold=2.2,
        loader=clientwise_dataloaders[forget_client[0]],
        optimizer_name=optimizer_name,
        device=device,
        threshold=threshold,
        clip_grad=5,
        epochs=num_unlearn_rounds
    )

    total_time = time.time() - start_time

    ######################## post train ############################

    finetuned_model = deepcopy(unlearned_global_model)

    for round in range(num_post_training_rounds):
        print(f"Finetuning PGA unlearned Model: {round}")

        new_local_weights = []
        chosen_clients = [i for i in range(0, num_clients)
                          if i not in forget_client]
        for client_idx in tqdm(chosen_clients):

            client_dataloader = clientwise_dataloaders[client_idx]
            client_model = deepcopy(finetuned_model)

            if client_idx == forget_client[0]:
                continue
            if optimizer_name == 'adam':
                optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)
            elif optimizer_name == 'sgd':
                optimizer = torch.optim.SGD(client_model.parameters(), lr=lr)
            else:
                raise ValueError(f"Optimizer {optimizer_name} not supported")

            loss_fn = torch.nn.CrossEntropyLoss()
            train_local_model(model=client_model, dataloader=client_dataloader,
                              loss_fn=loss_fn, optimizer=optimizer, num_epochs=num_local_epochs,
                              device=device)
            new_local_weights.append(client_model.state_dict())

        # server aggregation
        updated_global_weights = fed_avg(new_local_weights)
        finetuned_model.load_state_dict(updated_global_weights)

    return finetuned_model
