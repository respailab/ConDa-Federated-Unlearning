import numpy as np
from copy import deepcopy
import torch
from torch.utils.data import Dataset, Subset, DataLoader, ConcatDataset
from FedUnlearner.fed_learn import test_local_model


class PoisoningDataset(Dataset):
    def __init__(self, poisoning_samples, poisoning_labels):
        self.poisoning_samples = poisoning_samples
        self.poisoning_labels = poisoning_labels

    def __len__(self):
        return len(self.poisoning_samples)

    def __getitem__(self, idx):
        return self.poisoning_samples[idx], self.poisoning_labels[idx]


def create_poisoning_dataset(clientwise_dataset, forget_clients, num_poisoning_samples, test_split=0.2):
    """
    Create a poisoning dataset.
    Args:
        clientwise_dataset (dict): Dictionary containing datasets for each client.
        forget_clients (list): List of client IDs that should be targeted for poisoning.
        test_split (float): Proportion of the dataset to be used as test set.
        num_poisoning_samples (int): Number of samples to poison.

    Returns:
        clientwise_dataset: dict, modified clientwise dataset with poisoned labels
        poisoning_context: context dictionary with details about the poisoning
    """
    poisoning_context = {}
    poisoning_context['num_samples'] = num_poisoning_samples
    poisoning_context['poisoned_clients'] = forget_clients
    poisoning_context['train_poisoned_samples'] = {}
    poisoning_context['train_poisoned_labels'] = {}
    poisoning_context['test_poisoned_samples'] = {}
    poisoning_context['test_poisoned_labels'] = {}

    for client_id, dataset in clientwise_dataset.items():

        if client_id in forget_clients:
            idx = np.arange(len(dataset))
            poisoning_train_idx = np.random.choice(
                len(dataset), num_poisoning_samples, replace=False)
            remaining_idx = np.setdiff1d(idx, poisoning_train_idx)

            poisoning_test_idx = deepcopy(poisoning_train_idx)

            # Poison train dataset
            train_labels = [dataset[i][1].item() if isinstance(
                dataset[i][1], torch.Tensor) else dataset[i][1] for i in poisoning_train_idx]

            clean_labels = deepcopy(train_labels)

            poisoning_train_idx.sort()  # shuffling the train idices with respect to the labels

            train_poisoning_samples = [dataset[i][0]
                                       for i in poisoning_train_idx]
            test_poisoning_samples = [dataset[i][0]
                                      for i in poisoning_test_idx]

            poisoning_context['train_poisoned_samples'][client_id] = train_poisoning_samples
            poisoning_context['test_poisoned_samples'][client_id] = test_poisoning_samples
            poisoning_context['train_poisoned_labels'][client_id] = train_labels
            poisoning_context['test_poisoned_labels'][client_id] = clean_labels

            poisoning_dataset = PoisoningDataset(
                train_poisoning_samples, train_labels)

            client_remaining_dataset = Subset(dataset, remaining_idx)

            clientwise_dataset[client_id] = ConcatDataset(
                [client_remaining_dataset, poisoning_dataset])

    return clientwise_dataset, poisoning_context


def evaluate_poisoning_attack(model, poisoning_context, device):
    """
    Evaluate backdoor attack.
    Args:
        model: torch.nn.Module, model
        poisoning_context: dict, poisoning context
    Returns:
        Results: dict, results []
    """
    results = {}
    overall_test_acc = 0
    for client_id in poisoning_context['poisoned_clients']:
        train_dataset = PoisoningDataset(
            poisoning_samples=poisoning_context['train_poisoned_samples'][client_id],
            poisoning_labels=poisoning_context['train_poisoned_labels'][client_id])
        test_dataset = PoisoningDataset(
            poisoning_samples=poisoning_context['test_poisoned_samples'][client_id],
            poisoning_labels=poisoning_context['test_poisoned_labels'][client_id])

        # Create dataloaders for backdoor train and test dataset
        train_dataloader = DataLoader(
            train_dataset, batch_size=64, shuffle=False)
        test_dataloader = DataLoader(
            test_dataset, batch_size=64, shuffle=False)

        # Test both train and test backdoor dataset for accuracy metric
        train_acc, _ = test_local_model(
            model=model, dataloader=train_dataloader, device=device)
        test_acc, _ = test_local_model(
            model=model, dataloader=train_dataloader, device=device)
        overall_test_acc += test_acc

        results[client_id] = {}
        results[client_id]['train'] = {
            "num_samples": len(train_dataset), "accuracy": train_acc}
        results[client_id]['test'] = {
            "num_samples": len(test_dataset), "accuracy": test_acc}

    # calculate overall accuracy.
    results['overall_poisoned_accuracy'] = overall_test_acc/len(results)

    return results
