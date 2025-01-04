import torch
from typeguard import typechecked
from typing import Tuple, Dict

from FedUnlearner.utils import average_weights
from copy import deepcopy
from tqdm import tqdm
import shutil
import os

@typechecked
def fed_train(num_training_iterations: int, test_dataloader: torch.utils.data.DataLoader, 
              clientwise_dataloaders: dict[int, torch.utils.data.DataLoader], weights_path : str,
              global_model: torch.nn.Module, num_local_epochs: int, lr: float, optimizer_name: str, device: str = 'cpu'):
    """
    """
    
    if os.path.exists(weights_path):
        shutil.rmtree(weights_path)
    
    os.makedirs(weights_path)

    # savve the initial model
    torch.save(global_model.state_dict(), os.path.join(weights_path, "initial_model.pth"))
    global_model.to(device)
    global_model.train()
    
    clients = clientwise_dataloaders.keys()

    client_contributions = {}
    for client_idx in clients:
        client_contributions[client_idx] = []

    for iteration in range(num_training_iterations):
        print(f"Global Iteration: {iteration}")
        iteration_weights_path = os.path.join(weights_path, f"iteration_{iteration}")
        os.makedirs(iteration_weights_path, exist_ok = True)
        for client_idx in clients:
            print(f"Client: {client_idx}")
            client_dataloader = clientwise_dataloaders[client_idx]
            client_model = deepcopy(global_model)
            if optimizer_name == 'adam':
                optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)  # create optimizer
            elif optimizer_name == 'sgd':
                optimizer = torch.optim.SGD(client_model.parameters(), lr=lr)
            else:
                raise ValueError(f"Optimizer {optimizer_name} not supported")
            
            loss_fn = torch.nn.CrossEntropyLoss()  # create loss function
            
            
            train_local_model(model = client_model, dataloader = client_dataloader, 
                              loss_fn = loss_fn, optimizer = optimizer, num_epochs = num_local_epochs, 
                              device = device)
            torch.save(client_model.state_dict(), os.path.join(iteration_weights_path, f"client_{client_idx}.pth"))

            # test_acc_client, test_loss_client = test_local_model(client_model, test_dataloader, loss_fn, device)
            # print(f"Test Accuracy for client {client_idx} : {test_acc_client*100}, Loss : {test_loss_client}")

        # update gloal model
        updated_global_weights = average_weights(iteration_weights_path, device = device)
        global_model.load_state_dict(updated_global_weights)
        torch.save(global_model.state_dict(), os.path.join(iteration_weights_path, "global_model.pth"))

        

        # evaluate global model
        test_acc_global, test_loss_global = test_local_model(global_model, test_dataloader, loss_fn, device)
        print(f"Test Accuracy for global model : {test_acc_global*100}, Loss : {test_loss_global}")
    torch.save(global_model.state_dict(), os.path.join(weights_path, "final_model.pth"))

    return global_model

            

@typechecked
def train_local_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn, 
                        optimizer: torch.optim.Optimizer, num_epochs: int, device: str = 'cpu'):
    model = model.to(device)
    model.train()


    for iter in range(num_epochs):
        tqdm_iterator = tqdm(dataloader, desc = f"Epoch: {iter}")
        for images, labels in tqdm_iterator:
            images, labels = images.to(device), labels.to(device)

            model.zero_grad()
            log_probs = model(images)
            loss = loss_fn(log_probs, labels)
            loss.backward()
            optimizer.step()
            tqdm_iterator.set_postfix({"loss": loss.item()})

    return model.state_dict()

@torch.no_grad()
@typechecked
def test_local_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, 
                     loss_fn = None, device: str = 'cpu'):
    model = model.to(device)
    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            log_probs = model(images)
            if loss_fn is not None:
                test_loss += loss_fn(log_probs, labels).item()
            pred = log_probs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(dataloader.dataset)
    test_acc = correct / len(dataloader.dataset)
    return test_acc, test_loss

@torch.no_grad()
@typechecked
def get_classwise_accuracy(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, 
                           num_classes: int, device: str = 'cpu') -> Dict[int, float]:
    model = model.to(device)
    model.eval()
    classwise_correct = {}
    classwise_nums = {}
    for i in range(num_classes):
        classwise_correct[i] = 0
        classwise_nums[i] = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            log_probs = model(images)
            pred = log_probs.argmax(dim=1, keepdim=True)

            for i in range(num_classes):
                mask = (labels == i)
                cls_preds = pred.view_as(labels)[mask]
                cls_labels = labels[mask]
                classwise_correct[i] += cls_preds.eq(cls_labels).sum().item()
                classwise_nums[i] += mask.sum().item()
    classwise_acc = {}
    for i in range(num_classes):
        classwise_acc[i] = classwise_correct[i] / classwise_nums[i]
    return classwise_acc


@typechecked
def get_clientwise_accuracy(model: torch.nn.Module, clientwise_dataloaders: dict[int, torch.utils.data.DataLoader],
                            device: str = 'cpu') -> dict[int, float]:
    """
    Get the clientwise accuracy.
    """
    results = {}
    for client_id, dataloader in clientwise_dataloaders.items():
        test_acc, _ = test_local_model(model = model, dataloader = dataloader, device = device)
        results[client_id] = test_acc*100
    return results


def get_performance(model, test_dataloader, clientwise_dataloader, num_classes, device):
    results = {}
    test_acc, test_loss = test_local_model(model = model, dataloader = test_dataloader, device=device)
    results['test_acc'] = test_acc

    results['classwise_acc'] = get_classwise_accuracy(model = model, dataloader = test_dataloader, 
                                                      num_classes = num_classes, device = device)
    results['clientwise_acc'] = get_clientwise_accuracy(model = model, clientwise_dataloaders = clientwise_dataloader,
                                                        device = device)

    return results
