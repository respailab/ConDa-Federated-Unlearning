from torch.utils.data import Dataset, ConcatDataset, Subset, DataLoader
from FedUnlearner.fed_learn import test_local_model
import numpy as np

class BackdoorDataset(Dataset):
    def __init__(self, backdoor_samples, backdoor_label):
        self.backdoor_samples = backdoor_samples
        self.backdoor_label = backdoor_label

    def __len__(self):
        return len(self.backdoor_samples)

    def __getitem__(self, idx):
        return self.backdoor_samples[idx], self.backdoor_label


def backdoor_transformer(image, backdoor_pixels):
    """
    Injects backdoor pixels in the image.
    Args:
        image: list, image
        backdoor_pixels: list, backdoor_pixels
    """
    top_left_corner = backdoor_pixels[0]
    bottom_right_corner = backdoor_pixels[1]
    image[top_left_corner[0]:bottom_right_corner[0], top_left_corner[1]:bottom_right_corner[1]] = 1.
    return image



def create_backdoor_dataset(clientwise_dataset, forget_clients, backdoor_pixels, backdoor_label, num_samples):
    """
    Create a backdoor dataset.
    Args:
        clientwise_dataset: dict, clientwise dataset
        forget_clients: list, forget clients
        backdoor_pixels: list, backdoor pixels
        backdoor_label: int, backdoor label
        num_samples: int, number of samples
    Returns:
        backdoor_dataset: dict, backdoor dataset
        backdoor_context: dict, backdoor context
    """

    backdoor_context = {}
    backdoor_context['num_samples'] = num_samples
    backdoor_context['backdoor_label'] = backdoor_label
    backdoor_context['backdoor_pixels'] = backdoor_pixels
    backdoor_context['backdoor_clients'] = forget_clients

    backdoor_context['train_backdoor_samples'] = {}
    backdoor_context['test_backdoor_samples'] = {}

    for client_id, dataset in clientwise_dataset.items():
        if client_id in forget_clients:
            # Select the dataset indices
            idx = np.arange(len(dataset))#list(range(len(dataset)))
            np.random.shuffle(idx)
            backdoor_train_idx = idx[:num_samples].tolist()
            backdoor_test_idx = idx[num_samples:num_samples+int(len(dataset)*0.2)].tolist()
            

            # Inject Backdoor pixels in selected train and test images    
            train_backdoor_samples = [backdoor_transformer(
                dataset[i][0], backdoor_pixels) for i in backdoor_train_idx]
            test_backdoor_samples = [backdoor_transformer(
                dataset[i][0], backdoor_pixels) for i in backdoor_test_idx]

            backdoor_context['train_backdoor_samples'][client_id] = train_backdoor_samples
            backdoor_context['test_backdoor_samples'][client_id] = test_backdoor_samples

            backdoor_dataset = BackdoorDataset(
                backdoor_samples=train_backdoor_samples, backdoor_label=backdoor_label)

            # merge the backdoor samples with the original dataset
            clientwise_dataset[client_id] = ConcatDataset(
                [dataset, backdoor_dataset])
            

    return clientwise_dataset, backdoor_context


def evaluate_backdoor_attack(model, backdoor_context, device):
    """
    Evaluate backdoor attack.
    Args:
        model: torch.nn.Module, model
        backdoor_context: dict, backdoor context
    Returns:
        Results: dict, results []
    """
    results = {}
    overall_test_acc = 0
    for client_id in backdoor_context['backdoor_clients']:
        train_dataset = BackdoorDataset(
            backdoor_samples=backdoor_context['train_backdoor_samples'][client_id], backdoor_label=backdoor_context['backdoor_label'])
        test_dataset = BackdoorDataset(
            backdoor_samples=backdoor_context['test_backdoor_samples'][client_id], backdoor_label=backdoor_context['backdoor_label'])

        # Create dataloaders for backdoor train and test dataset
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Test both train and test backdoor dataset for accuracy metric
        train_acc, _ = test_local_model(model = model, dataloader = train_dataloader, device = device)
        test_acc, _ = test_local_model(model = model, dataloader = test_dataloader, device = device)
        overall_test_acc += test_acc

        results[client_id] = {}
        results[client_id]['train'] = {
            "num_samples": len(train_dataset), "accuracy": train_acc}
        results[client_id]['test'] = {
            "num_samples": len(test_dataset), "accuracy": test_acc}

    results['overall_backdoor_accuracy'] = overall_test_acc/len(results)  # calculate overall accuracy.

    return results
