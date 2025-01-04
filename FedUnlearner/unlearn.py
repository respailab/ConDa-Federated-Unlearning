import torch
import os
from typing import List
from typeguard import typechecked
from collections import OrderedDict


@typechecked
def get_client_contribution(start_model: OrderedDict, weights_path: str):
    checkpoint_list = os.listdir(weights_path)
    checkpoint_list = [checkpoint for checkpoint in checkpoint_list if checkpoint.startswith("client_") and checkpoint.endswith(".pth")]
    checkpoint_list = [int(checkpoint[7:-4]) for checkpoint in checkpoint_list]
    checkpoint_list.sort()

    client_wise_differences = dict()
    for client_id in checkpoint_list:
        client_weights = torch.load(os.path.join(weights_path, f"client_{client_id}.pth"), map_location = 'cpu')
        difference = dict()
        for param in start_model.keys():
            difference[param] = torch.abs(start_model[param] - client_weights[param])
        client_wise_differences[client_id] = difference

    return client_wise_differences

def get_group_contribution(contributions):
    """
    Get the average contribution of the group.
    """
    avg_contributions = dict()
    for key in contributions[0].keys():
        avg_contributions[key] = torch.zeros_like(contributions[0][key])
        for i in range(len(contributions)):
            avg_contributions[key] += contributions[i][key]
        avg_contributions[key] = torch.div(avg_contributions[key], len(contributions))
    return avg_contributions
    
@typechecked
def unlearn(global_model: OrderedDict, forget_clients: List[int], total_num_clients: int, 
            weights_path: str, dampening_constant: float, dampening_upper_bound: float, ratio_cutoff: float):
    training_weights_path = os.path.join(weights_path, "full_training")
    start_model = torch.load(os.path.join(training_weights_path, "initial_model.pth"), map_location = 'cpu')

    checkpoint_folders = os.listdir(training_weights_path)

    # keep only elements in checkpoints_folders which are directories
    client_folders = [folder for folder in checkpoint_folders if os.path.isdir(os.path.join(training_weights_path, folder))]
    client_folders = [folder for folder in checkpoint_folders if folder.startswith("iteration")]
    client_iterations = [int(folder.split("_")[1]) for folder in client_folders]
    client_iterations.sort()

    # get average contribution by clients
    avg_contributions = dict()
    for iteration in client_iterations:

        iteration_path = os.path.join(training_weights_path, f"iteration_{iteration}")

        # get client-wise contribution for current iteration.
        client_contributions = get_client_contribution(start_model, iteration_path)

        # Add in the contibutions for each client
        for client_id, contributions in client_contributions.items():

            # If the contributiosns doesn't exist for the client_id, make an empty dict.
            if avg_contributions.get(client_id) is None:
                avg_contributions[client_id] = dict()
            
            # Iterate through each param and add in the contributions 
            for param in start_model.keys():
                # If the param id doesn't exists, add a zero vector.
                if avg_contributions[client_id].get(param) is None:
                    avg_contributions[client_id][param] = torch.zeros_like(contributions[param])
                avg_contributions[client_id][param] += contributions[param]
    
    


    forget_client_contribution = [avg_contributions[forget_client] for forget_client in forget_clients]
    avg_forget_client_contribution = get_group_contribution(forget_client_contribution)
    total_client_contribution = [avg_contributions[client] for client in range(total_num_clients) if client != forget_clients[0]]
    avg_client_contribution = get_group_contribution(total_client_contribution)
    # unlearning, dampen the contributions/weights
    unlearned_global_weights = apply_dampening(global_model, avg_client_contribution, avg_forget_client_contribution, 
                                               dampening_constant = dampening_constant, 
                                               dampening_upper_bound = dampening_upper_bound, 
                                               ratio_cutoff = ratio_cutoff)
    return unlearned_global_weights



def apply_dampening(global_model, forget_client_contributions, retain_clients_contirbutions, dampening_constant,
                    dampening_upper_bound, ratio_cutoff):
        """
        Apply dampening to the global model based on the gradients of the local models.
        Args:
            global_model: The global model which will be dampened. 
            forget_client_contributions: The gradient contributions of the forget clients/models.
            retain_clients_contributions: The gradient contributions of the retain clients/models.
            dampening_constant: The dampening constant.
            dampening_upper_bound: The upper bound for the final dampening factor. Used to cap the increasing of 
            the parameters.
            ratio_cutoff: The cutoff/filter factor for ratios. Any parameter having the ratio greater than this value will not be updated.
              A high ratio means less contribution of the forget model, leading to less dampening. 
        Returns:
            The updated global model.
        """

        with torch.no_grad():
          for (global_name, forget_grads), (index, retain_grads) in zip(
              forget_client_contributions.items(),
              retain_clients_contirbutions.items()
          ):

              if len(forget_grads.shape) > 0:
                # Synapse Dampening with parameter dampening constant
                weight = global_model[global_name]
                # diff = torch.abs(g2_grads - g1_grads) # torch.abs(torch.abs(g2_grads) - torch.abs(g1_grads))
                retain_contribution = torch.abs(retain_grads) # epsilon
                forget_contribution = torch.abs(forget_grads)
                ratio = retain_contribution / forget_contribution
                update_locations = (ratio < ratio_cutoff)
                dampening_factor = torch.mul(ratio, dampening_constant)

                update = dampening_factor[update_locations]
                # Bound by 1 to prevent parameter values to increase.
                min_locs = (update > dampening_upper_bound)
                update[min_locs] = dampening_upper_bound
                weight[update_locations] = weight[update_locations].mul(update)
        return global_model