import copy
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
from xgboost import XGBClassifier

"""
Membership Inference Attack from
https://www.chenwang.net.cn/publications/FedEraser-IWQoS21.pdf
Code: https://www.dropbox.com/s/1lhx962axovbbom/FedEraser-Code.zip?dl=0&e=5&file_subpath=%2FFedEraser-Code
"""

@torch.no_grad
def train_attack_model(shadow_global_model, shadow_client_loaders, shadow_test_loader, dataset, device):
    shadow_model = shadow_global_model
    n_class_dict = dict()
    n_class_dict['mnist'] = 10
    n_class_dict['cifar10'] = 10
    n_class_dict['cifar100'] = 100

    N_class = n_class_dict[dataset]

    device = torch.device(device)
    shadow_model.to(device)

    shadow_model.eval()
    ####
    pred_4_mem_list = []
    for _, dataloader in shadow_client_loaders.items():
        for data, target in dataloader:
            data = data.to(device)
            out = shadow_model(data)
            out = softmax(out, dim=1)
            pred_4_mem_list.append(out.cpu().detach().numpy())

    pred_4_mem = np.concatenate(pred_4_mem_list, axis=0)

    ####
    pred_4_nonmem_list = []
    for data, target in shadow_test_loader:
        data = data.to(device)
        out = shadow_model(data)
        out = softmax(out, dim=1)
        pred_4_nonmem_list.append(out.cpu().detach().numpy())
    pred_4_nonmem = np.concatenate(pred_4_nonmem_list, axis=0)

    att_y = np.hstack(
        (np.ones(pred_4_mem.shape[0]), np.zeros(pred_4_nonmem.shape[0])))
    att_y = att_y.astype(np.int16)

    att_X = np.vstack((pred_4_mem, pred_4_nonmem))
    att_X.sort(axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        att_X, att_y, test_size=0.1)

    # For possible division by zero error
    scale_pos_weight = pred_4_nonmem.shape[0] / pred_4_mem.shape[0] if pred_4_mem.shape[0] > 0 else 1
    attacker = XGBClassifier(n_estimators=300,
                             n_jobs=-1,
                             max_depth=30,
                             objective='binary:logistic',
                             booster="gbtree",
                             # learning_rate=None,
                             # tree_method = 'gpu_hist',
                             scale_pos_weight=scale_pos_weight
                             )

    attacker.fit(X_train, y_train)
    return attacker


def evaluate_mia_attack(target_model: torch.nn.Module,
                        attack_model: torch.nn.Module,
                        client_loaders,
                        test_loader,
                        dataset: str,
                        forget_client_idx: int,
                        device:str):
    results = {}

    n_class_dict = dict()
    n_class_dict['mnist'] = 10
    n_class_dict['cifar10'] = 10
    n_class_dict['cifar100'] = 100

    N_class = n_class_dict[dataset]

    target_model.to(device)

    target_model.eval()

    # The predictive output of forgotten user data after passing through the target model.
    unlearn_X_list = []
    with torch.no_grad():
        for data, target in client_loaders[forget_client_idx]:
            data = data.to(device)
            out = target_model(data)
            out = softmax(out, dim=1)
            unlearn_X_list.append(out.cpu().detach().numpy())


    unlearn_X = np.concatenate(unlearn_X_list, axis=0)
    unlearn_X.sort(axis=1)
    unlearn_y = np.ones(unlearn_X.shape[0])
    unlearn_y = unlearn_y.astype(np.int16)

    N_unlearn_sample = len(unlearn_y)

    # Shuffle the test dataset
    shuffled_test_loader = DataLoader(test_loader.dataset, batch_size=test_loader.batch_size, shuffle=True)


    # Test data, predictive output obtained after passing the target model
    test_X_list = []
    total_samples_collected = 0
    with torch.no_grad():
        for data, target in shuffled_test_loader:
            data = data.to(device)
            out = target_model(data)
            out = softmax(out, dim=1)
            test_X_list.append(out.cpu().detach().numpy())
            total_samples_collected += out.shape[0]

            if total_samples_collected > N_unlearn_sample:
                break
    

    test_X = np.concatenate(test_X_list, axis=0)[:N_unlearn_sample]
    test_X.sort(axis=1)
    test_y = np.zeros(test_X.shape[0])
    test_y = test_y.astype(np.int16)

    # The data of the forgotten user passed through the output of the target model, and the data of the test set passed through the output of the target model were spliced together
    # The balanced data set that forms the 50% train 50% test.
    combined_X = np.vstack((unlearn_X, test_X))
    combined_Y = np.hstack((unlearn_y, test_y))

    pred_Y = attack_model.predict(combined_X)
    pred_proba_Y = attack_model.predict_proba(combined_X)[:, 1]

    accuracy = accuracy_score(combined_Y, pred_Y)
    precision = precision_score(combined_Y, pred_Y, pos_label=1)
    recall = recall_score(combined_Y, pred_Y, pos_label=1)
    f1 = f1_score(combined_Y, pred_Y, pos_label=1)

    results['mia_attacker_accuracy'] = accuracy
    results['mia_attacker_precision'] = precision
    results['mia_attacker_recall'] = recall
    results['mia_attacker_f1'] = f1
    results['mia_attacker_predictions'] = pred_Y.tolist()
    results['mia_attacker_probabilities'] = pred_proba_Y.tolist()

    return results
