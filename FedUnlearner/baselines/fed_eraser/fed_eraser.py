from typeguard import typechecked
from typing import Dict, List, Tuple, Union
import torch
from FedUnlearner.utils import average_weights
from FedUnlearner.fed_learn import *
from tqdm import tqdm
import copy


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


def fed_eraser_one_step(
    old_client_models,
    new_client_models,
    global_model_before_forget,
    global_model_after_forget,
    device
):
    old_param_update = dict()  # oldCM - oldGM_t
    new_param_update = dict()  # newCM - newGM_t

    new_global_model_state = global_model_after_forget  # newGM_t
    return_model_state = (
        dict()
    )  # newGM_t + ||oldCM - oldGM_t||*(newCM - newGM_t)/||newCM - newGM_t||

    assert len(old_client_models) == len(new_client_models)
    for layer in global_model_before_forget.keys():
        old_param_update[layer] = 0 * global_model_before_forget[layer]
        new_param_update[layer] = 0 * global_model_before_forget[layer]
        return_model_state[layer] = 0 * global_model_before_forget[layer]

        for i in range(len(new_client_models)):
            old_param_update[layer] += old_client_models[i][layer]
            new_param_update[layer] += new_client_models[i][layer].to(device)

        old_param_update[layer] /= len(new_client_models)  # oldCM
        new_param_update[layer] /= len(new_client_models)  # newCM

        old_param_update[layer] = (
            old_param_update[layer] - global_model_before_forget[layer]
        )  # oldCM - oldGM_t
        new_param_update[layer] = (
            new_param_update[layer] - global_model_after_forget[layer]
        )  # newCM - newGM_t

        step_length = torch.norm(
            old_param_update[layer])  # ||oldCM - oldGM_t||
        step_direction = new_param_update[layer] / torch.norm(
            new_param_update[layer]
        )  # (newCM - newGM_t)/||newCM - newGM_t||

        return_model_state[layer] = (
            new_global_model_state[layer] + step_length * step_direction
        )

    return return_model_state


@typechecked
def run_fed_eraser(
        global_model: torch.nn.Module,
        weights_path: str,
        forget_clients: List[int],
        clientwise_dataloaders: Dict[int, torch.utils.data.DataLoader],
        device: str,
        optimizer_name: str,
        num_clients: int,
        num_rounds: int,
        lr: float,
        num_unlearn_rounds=1,
        local_cali_round=1,
        num_post_training_rounds=1) -> torch.nn.Module:
    old_global_models = []

    for round in range(num_rounds):
        global_weights_path = os.path.join(weights_path,
                                           f"iteration_{round}",
                                           "global_model.pth")
        global_param = torch.load(global_weights_path)
        old_global_models.append(global_param)

    # new_global_models = []
    chosen_clients = [i for i in range(0, num_clients)
                      if i not in forget_clients]
    rounds = [i
              for i in range(0, num_rounds, num_rounds // num_unlearn_rounds)]
    new_prev_global_model = copy.deepcopy(global_model)
    for i, round in enumerate(rounds):
        iteration_weights_path = os.path.join(weights_path,
                                              f"iteration_{round}")
        roundth = num_rounds + i
        print(f"Round {roundth+1}/{num_rounds + num_unlearn_rounds}")
        list_params = []

        old_client_parameters = []
        new_client_parameters = []
        # For first round of unlearning, only fedavg client upates
        if round == 0:
            for client in chosen_clients:
                client_parameters = torch.load(os.path.join(iteration_weights_path,
                                                            f"client_{client}.pth"))
                old_client_parameters.append(client_parameters)

            new_global_model = average_weights(
                iteration_weights_path, device=device)
            # new_global_models.append(new_global_model)
            continue

        old_global_model = old_global_models[round]

        new_prev_global_model.load_state_dict(new_global_model)

        # for other rounds
        for client in tqdm(chosen_clients):
            client_parameters = torch.load(os.path.join(iteration_weights_path,
                                                        f"client_{client}.pth"))
            old_client_parameters.append(client_parameters)

            if optimizer_name == 'adam':
                optimizer = torch.optim.Adam(
                    new_prev_global_model.parameters(), lr=lr)
            elif optimizer_name == 'sgd':
                optimizer = torch.optim.SGD(
                    new_prev_global_model.parameters(), lr=lr)
            else:
                raise ValueError(f"Optimizer {optimizer_name} not supported")

            loss_fn = torch.nn.CrossEntropyLoss()
            new_client_parameter = train_local_model(
                model=deepcopy(new_prev_global_model),
                dataloader=clientwise_dataloaders[client],
                loss_fn=loss_fn,
                optimizer=optimizer,
                num_epochs=local_cali_round,
                device=device)
            new_client_parameters.append({k: v.cpu() for k, v in new_client_parameter.items()})
        # Loss functions
        new_global_model = fed_eraser_one_step(
            old_client_parameters,
            new_client_parameters,
            old_global_model,
            new_prev_global_model.state_dict(),
            device=device
        )
    ### Post train ####
    unlearned_global_model = copy.deepcopy(global_model)
    global_param = copy.deepcopy(new_global_model)
    unlearned_global_model.load_state_dict(new_global_model)

    start_round = num_rounds + len(rounds)
    end_round = start_round + num_post_training_rounds
    for round in range(start_round, end_round):
        chosen_clients = [i for i in range(0, num_clients)
                          if i not in forget_clients]
        list_params = []
        for client in tqdm(chosen_clients):
            if optimizer_name == 'adam':
                optimizer = torch.optim.Adam(
                    unlearned_global_model.parameters(), lr=lr)
            elif optimizer_name == 'sgd':
                optimizer = torch.optim.SGD(
                    unlearned_global_model.parameters(), lr=lr)
            else:
                raise ValueError(f"Optimizer {optimizer_name} not supported")

            loss_fn = torch.nn.CrossEntropyLoss()
            print(f"-----------client {client} starts training----------")
            tem_param = train_local_model(
                model=deepcopy(unlearned_global_model),
                dataloader=clientwise_dataloaders[client],
                loss_fn=loss_fn,
                optimizer=optimizer,
                num_epochs=local_cali_round,
                device=device
            )
            list_params.append(tem_param)

        # server aggregation
        global_param = fed_avg(list_params)
    unlearned_global_model.load_state_dict(global_param)
    return unlearned_global_model
