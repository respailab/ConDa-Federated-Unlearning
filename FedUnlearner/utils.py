import torch
import copy
import os
from typing import Dict
from typeguard import typechecked

def average_weights(weights_path, device):
    """
    Returns the average of the weights.
    """
    w_list = os.listdir(weights_path)
    w_list = [w for w in w_list if w.startswith("client_") and w.endswith(
        ".pth")]  # should have a structure client_{x}.pth
    assert len(w_list) > 0, "Weights path is empty"
    num_models = len(w_list)
    w_avg = torch.load(os.path.join(
        weights_path, w_list[0]), map_location=device)

    for w_path in w_list[1:]:
        new_w = torch.load(os.path.join(
            weights_path, w_path), map_location=device)
        for key in w_avg.keys():
            w_avg[key] += new_w[key]

    for key in w_avg.keys():
        w_avg[key] = torch.div(w_avg[key], num_models)
    return w_avg


def print_exp_details(args):
    print('\nExperimental details:')
    print(f"    Model     : {args.model}")
    print(f"    Optimizer : {args.optimizer}")
    print(f"    Learning  : {args.lr}")
    print(f"    Global Rounds   : {args.num_training_iterations}\n")

    print('    Federated parameters:')
    if args.client_data_distribution == "iid":
        print('    IID')
    else:
        print('    Non-IID')
    print(
        f"    Number of participating users  : {args.num_participating_clients}")
    print(f"    Batch size   : {args.batch_size}")
    print(f"    Local Epochs       : {args.num_local_epochs}\n")

def get_labels_from_dataset(dataset):
    """
    Get labels from a torch Dataset or Subset without iterating through the whole dataset.
    Args:
        dataset: torch.utils.data.Dataset or torch.utils.data.Subset
    Returns:
        labels: list of labels
    """
    if isinstance(dataset, torch.utils.data.Subset):
        # Access the underlying dataset and indices
        subset_indices = dataset.indices
        if hasattr(dataset.dataset, 'targets'):
            labels = [dataset.dataset.targets[i] for i in subset_indices]
            # Convert labels to integers if they are tensors
            return [label.item() if isinstance(label, torch.Tensor) else label for label in labels]
        else:
            raise AttributeError(
                "The underlying dataset does not have a 'targets' attribute.")
    elif hasattr(dataset, 'targets'):
        labels = dataset.targets
        # Convert labels to integers if they are tensors
        return [label.item() if isinstance(label, torch.Tensor) else label for label in labels]
    else:
        raise AttributeError(
            "The dataset does not have a 'targets' attribute.")


@typechecked
def print_clientwise_class_distribution(clientwise_dataset: Dict[int, torch.utils.data.Dataset],
                                        num_classes: int, num_workers: int = 0):
    def create_labels(classes):
        labels = dict()
        for i in range(classes):
            labels[i] = 0
        return labels

    for client_id, client_dataset in clientwise_dataset.items():
        labels = create_labels(num_classes)
        data_labels = get_labels_from_dataset(client_dataset)
        for label in data_labels:
            labels[label] += 1
        print(f"Data distribution for client : {client_id} :::: { labels}")
