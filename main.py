import torch
import torchvision
import argparse
import json
import time
import os
from copy import deepcopy


from FedUnlearner.utils import print_exp_details, print_clientwise_class_distribution
from FedUnlearner.data_utils import get_dataset, create_dirichlet_data_distribution, create_iid_data_distribution
from FedUnlearner.fed_learn import fed_train, get_performance
from FedUnlearner.unlearn import unlearn as unlearn_ours
from FedUnlearner.models import AllCNN, ResNet18
from FedUnlearner.attacks.backdoor import create_backdoor_dataset, evaluate_backdoor_attack
from FedUnlearner.baselines import run_pga, run_fed_eraser
from FedUnlearner.attacks.mia import train_attack_model, evaluate_mia_attack
from FedUnlearner.attacks.poisoning import create_poisoning_dataset, evaluate_poisoning_attack


# create argument parser
parser = argparse.ArgumentParser(description='FedUnlearner')

# add arguments
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--exp_path", default="./experiments/", type=str)
parser.add_argument('--model', type=str, default='allcnn', choices=["allcnn", 'resnet18'],
                    help='model name')
parser.add_argument('--pretrained', type=bool,
                    default=False, help='use pretrained model')

parser.add_argument('--dataset', type=str, default='cifar10', choices=["mnist", "cifar10"],
                    help='dataset name')
parser.add_argument('--optimizer', type=str, default='adam', choices=["sgd", "adam"],
                    help='optimizer name')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float,
                    default=0.0001, help='weight decay')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--num_local_epochs', type=int,
                    default=1, help='number of local epochs')

parser.add_argument('--num_training_iterations', type=int, default=1,
                    help='number of training iterations for global model')
parser.add_argument('--num_participating_clients', type=int, default=-1, help='number of users participating in trainig, \
                                                                                    -1 if all are required to participate')

# baslines
parser.add_argument('--baselines', type=str, nargs="*", default=[], choices=['pga', 'fed_eraser'],
                    help='baseline methods for unlearning')

# backdoor attack related arguments
parser.add_argument('--apply_backdoor', action='store_true',
                    help='apply backdoor attack')
parser.add_argument('--backdoor_position', type=str, default='corner', choices=["corner", "center"],
                    help='backdoor position')
parser.add_argument('--num_backdoor_samples_per_forget_client', type=int, default=10,
                    help='number of backdoor samples per forget client')
parser.add_argument('--backdoor_label', type=int,
                    default=0, help='backdoor label')

# membership inference attack related arguments
parser.add_argument('--apply_membership_inference', type=bool, default=False,
                    help='apply membership inference attack')
parser.add_argument('--attack_type', type=str, default='blackbox', choices=["blackbox", "whitebox"],
                    help='attack type')

# label posioning attack related arguments
parser.add_argument('--apply_label_poisoning', type=bool, default=False,
                    help='apply label poisoning attack')
parser.add_argument('--num_label_poison_samples', type=int, default=10,
                    help='number of label poisoning samples')

# provide indexes of clients which are to be forgotten, allow multiple clients to be forgotten
parser.add_argument('--forget_clients', type=int, nargs='+',
                    default=[0], help='forget clients')
parser.add_argument('--total_num_clients', type=int,
                    default=10, help='total number of clients')
parser.add_argument('--client_data_distribution', type=str, default='dirichlet',
                    choices=["dirichlet", "iid"], help='client data distribution')
parser.add_argument('--dampening_constant', type=float,
                    default=0.5, help='dampening constant')
parser.add_argument('--dampening_upper_bound', type=float,
                    default=0.5, help='dampening upper bound')
parser.add_argument('--ratio_cutoff', type=float,
                    default=0.5, help='ratio cutoff')
parser.add_argument('--device', type=str, default='cpu',
                    choices=["cpu", "cuda"], help='device name')
parser.add_argument('--seed', type=int, default=None, help='random seed')
parser.add_argument('--verbose', type=bool, default=True, help='verbose')
parser.add_argument("--num_workers", type=int, default=32,
                    help="number of workers for data loading")

if __name__ == "__main__":

    args = parser.parse_args()
    weights_path = os.path.abspath(os.path.join(args.exp_path, args.exp_name))

    print_exp_details(args)
    summary = {}
    # get the dataset
    train_dataset, test_dataset, num_classes = get_dataset(args.dataset)

    # create client groups
    client_groups = None

    if args.client_data_distribution == 'dirichlet':
        clientwise_dataset = create_dirichlet_data_distribution(train_dataset,
                                                                num_clients=args.total_num_clients, num_classes=num_classes)
    elif args.client_data_distribution == 'iid':
        clientwise_dataset = create_iid_data_distribution(train_dataset, num_clients=args.total_num_clients,
                                                          num_classes=num_classes)
    else:
        raise "Invalid client data distribution"

    # print the clientwise class distribution
    print_clientwise_class_distribution(clientwise_dataset, num_classes)

    if args.num_participating_clients > 1:
        print(
            f"Cutting of num participating client to: {args.num_participating_clients}")
        clientwise_dataset = {i: clientwise_dataset[i] for i in range(
            args.num_participating_clients)}
        print("Clientwise distribution after cutting: ")
        print_clientwise_class_distribution(clientwise_dataset, num_classes)
    # get the forget client

    if len(args.forget_clients) > 1:
        raise "Only one client forgetting supported at the moment."
    forget_client = args.forget_clients[0]

    if args.apply_backdoor:
        backdoor_dataset = None
        backdoor_pixels = None
        image_size = 224
        patch_size = 30
        if args.backdoor_position == 'corner':
            # [top left corner of patch, bottom right corner of patch]
            backdoor_pixels = [(0, 0), (patch_size, patch_size)]
        elif args.backdoor_position == 'center':
            backdoor_pixels = [(image_size//2 - patch_size//2, image_size//2 - patch_size//2),
                               (image_size//2 + patch_size//2, image_size//2 + patch_size//2)]
        else:
            raise "Invalid backdoor position"

        print(
            f"Size of client dataset before backdoor ingestion: {len(clientwise_dataset[args.forget_clients[0]])}")
        clientwise_dataset, backdoor_context = create_backdoor_dataset(clientwise_dataset=clientwise_dataset,
                                                                       forget_clients=args.forget_clients,
                                                                       backdoor_pixels=backdoor_pixels,
                                                                       backdoor_label=args.backdoor_label,
                                                                       num_samples=args.num_backdoor_samples_per_forget_client
                                                                       )

        print(
            f"Size of client dataset after backdoor ingestion: {len(clientwise_dataset[args.forget_clients[0]])}")

    if args.apply_label_poisoning:
        clientwise_dataset, poisoning_context = create_poisoning_dataset(clientwise_dataset=clientwise_dataset,
                                                                         forget_clients=args.forget_clients,
                                                                         test_split=0.2,
                                                                         num_poisoning_samples=args.num_label_poison_samples)

    # create dataloaders for the clients
    clientwise_dataloaders = {}
    for client_id, client_dataset in clientwise_dataset.items():
        print(f"Creating data loader for client: {client_id}")
        client_dataloader = torch.utils.data.DataLoader(
            client_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        clientwise_dataloaders[client_id] = client_dataloader
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers)

    # train the model
    global_model = None
    retrained_global_model = None
    if args.model == 'allcnn':
        if args.dataset == 'mnist':
            global_model = AllCNN(num_classes=num_classes, num_channels=1)
        else:
            global_model = AllCNN(num_classes=num_classes)
    elif args.model == 'resnet18':
        if args.dataset == 'mnist':
            global_model = ResNet18(num_classes=num_classes,
                                    pretrained=args.pretrained, num_channels=1)
        else:
            global_model = ResNet18(num_classes=num_classes,
                                    pretrained=args.pretrained)
    else:
        raise "Invalid model name"
    retrained_global_model = deepcopy(global_model)
    print(f"Model: {global_model}")
    train_path = os.path.abspath(os.path.join(weights_path, "full_training"))
    # train the model
    global_model = fed_train(num_training_iterations=args.num_training_iterations, test_dataloader=test_dataloader,
                             clientwise_dataloaders=clientwise_dataloaders,
                             global_model=global_model, num_local_epochs=args.num_local_epochs,
                             device=args.device, weights_path=train_path, lr=args.lr, optimizer_name=args.optimizer)

    perf = get_performance(model=global_model, test_dataloader=test_dataloader, num_classes=num_classes,
                           clientwise_dataloader=clientwise_dataloaders, device=args.device)
    summary['performance'] = {}
    summary['performance']['after_training'] = perf
    if args.verbose:
        print(f"Performance after training : {perf}")
    # -------------------------------------------------------
    # train mia attack model
    if args.apply_membership_inference:
        shadow_model = deepcopy(global_model)
        # attack_model = XGBClassifier()
        attack_model = train_attack_model(shadow_global_model=shadow_model,
                                          shadow_client_loaders=clientwise_dataloaders,
                                          shadow_test_loader=test_dataloader,
                                          dataset=args.dataset,
                                          device=args.device)
    # ---------------------------------------------------------
    # evaluate attack accuracy
    if args.apply_backdoor:

        # ------------------------------------------
        # TO-DO: Implement data poisoning and eval from https://arxiv.org/abs/2402.14015 (give credit in code!)

        ''' 
        -- Create data loaders with poison/clean data --
        To be implemented in here (till ca. line 44): https://github.com/drimpossible/corrective-unlearning-bench/blob/main/src/main.py
        FYI kep corrupt size at 3 for the pixel attack patch, manip_set_size=opt.forget_set_size determines how many poisoned samples
        From line 80 onwards keep opt.deletion_size == opt.forget_set_size to have all poison samples known. Unlearning unkown samples is
        a different hard problem beyond the scope of this FL-UL paper
        Helper functions in here: https://github.com/drimpossible/corrective-unlearning-bench/blob/main/src/datasets.py

        -- Eval --
        For evaluation, report the accuracies on the poisoned data with the clean labels 
        (i.e., what they should be, not what the manipulated poisoned sample says it is) and
        the accuracy on the remaining clean data. See figures 2 & 3
        '''
        # ------------------------------------------

        backdoor_results = evaluate_backdoor_attack(model=global_model, backdoor_context=backdoor_context,
                                                    device=args.device)
        summary['backdoor_results'] = {}
        summary['backdoor_results']['global_model_after_training'] = backdoor_results

        backdoor_client = deepcopy(global_model)
        backdoor_client_path = os.path.abspath(os.path.join(
            train_path, f"iteration_{args.num_training_iterations - 1}", f"client_{args.forget_clients[0]}.pth"))
        backdoor_client.load_state_dict(torch.load(backdoor_client_path))

        backdoor_results_client = evaluate_backdoor_attack(model=backdoor_client, backdoor_context=backdoor_context,
                                                           device=args.device)
        summary['backdoor_results']['backdoor_client_after_training'] = backdoor_results_client

        if args.verbose:
            print(
                f"Backdoor results after training : {summary['backdoor_results']}")

    # evaluate poisoning accuracy
    if args.apply_label_poisoning:
        poisoning_results = evaluate_poisoning_attack(model=global_model,
                                                      poisoning_context=poisoning_context,
                                                      device=args.device)
        summary['poisoning_results'] = {}
        summary['poisoning_results']['global_model_after_training'] = poisoning_results

        poisoning_client = deepcopy(global_model)
        poisoning_client_path = os.path.abspath(os.path.join(
            train_path, f"iteration_{args.num_training_iterations - 1}", f"client_{args.forget_clients[0]}.pth"))
        poisoning_client.load_state_dict(torch.load(poisoning_client_path))

        poisoning_results_client = evaluate_poisoning_attack(model=poisoning_client,
                                                             poisoning_context=poisoning_context,
                                                             device=args.device)
        summary['poisoning_results']['poisoning_client_after_training'] = poisoning_results_client

        if args.verbose:
            print(
                f"Poisoning results after training : {summary['poisoning_results']}")

    retrain_path = os.path.join(weights_path, "retraining")
    # train the model on retain data
    retain_clientwise_dataloaders = {key: value for key, value in clientwise_dataloaders.items()
                                     if key not in args.forget_clients}
    print(f"Retain Client wise Loaders: {retain_clientwise_dataloaders}")

    retrained_global_model = fed_train(num_training_iterations=args.num_training_iterations, test_dataloader=test_dataloader,
                                       clientwise_dataloaders=retain_clientwise_dataloaders,
                                       global_model=retrained_global_model, num_local_epochs=args.num_local_epochs,
                                       device=args.device, weights_path=retrain_path, lr=args.lr, optimizer_name=args.optimizer)

    perf = get_performance(model=retrained_global_model, test_dataloader=test_dataloader,
                           clientwise_dataloader=clientwise_dataloaders,
                           num_classes=num_classes, device=args.device)
    summary['performance']['after_retraining'] = perf
    if args.verbose:
        print(f"Performance after retraining : {perf}")
    # evaluate attack accuracy on retrained model
    if args.apply_backdoor:
        retrained_backdoor_results = evaluate_backdoor_attack(model=retrained_global_model,
                                                              backdoor_context=backdoor_context, device=args.device)
        summary['backdoor_results']['after_retraining'] = retrained_backdoor_results
        if args.verbose:
            print(
                f"Backdoor results after retraining : {retrained_backdoor_results}")

    if args.apply_label_poisoning:
        retrained_poisoning_results = evaluate_poisoning_attack(model=retrained_global_model,
                                                                poisoning_context=poisoning_context,
                                                                device=args.device)
        summary['poisoning_results']['after_retraining'] = retrained_poisoning_results

        if args.verbose:
            print(
                f"Poisoning results after retraining : {retrained_poisoning_results}")

    # Run Baseline methods and check the performance on them
    baselines_methods = args.baselines
    for baseline in baselines_methods:
        if baseline == 'pga':
            global_model_pga = deepcopy(global_model)
            unlearned_pga_model = run_pga(global_model=global_model_pga,
                                          weights_path=train_path,
                                          clientwise_dataloaders=clientwise_dataloaders,
                                          forget_client=args.forget_clients,
                                          model=args.model,
                                          dataset=args.dataset,
                                          num_clients=args.total_num_clients,
                                          num_classes=num_classes,
                                          pretrained=args.pretrained,
                                          num_training_iterations=args.num_training_iterations,
                                          device=args.device,
                                          lr=args.lr,
                                          optimizer_name=args.optimizer,
                                          num_local_epochs=args.num_local_epochs,
                                          num_unlearn_rounds=1,
                                          num_post_training_rounds=1)
            perf = get_performance(model=unlearned_pga_model, test_dataloader=test_dataloader,
                                   clientwise_dataloader=clientwise_dataloaders, num_classes=num_classes,
                                   device=args.device)
            summary['performance']['after_pga'] = perf
            if args.verbose:
                print(f"Performance after pga : {perf}")
            # check backdoor on PGA model
            if args.apply_backdoor:
                forget_backdoor_pga = evaluate_backdoor_attack(model=unlearned_pga_model, backdoor_context=backdoor_context,
                                                               device=args.device)
                summary['backdoor_results']['after_pga'] = forget_backdoor_pga

                if args.verbose:
                    print(
                        f"Backdoor results after pga : {forget_backdoor_pga}")
            if args.apply_label_poisoning:
                forget_poisoning_pga = evaluate_poisoning_attack(model=unlearned_pga_model,
                                                                 poisoning_context=poisoning_context,
                                                                 device=args.device)
                summary['poisoning_results']['after_pga'] = forget_poisoning_pga

                if args.verbose:
                    print(
                        f"Poisoning results after pga : {forget_poisoning_pga}")
        elif baseline == 'fed_eraser':
            global_model_federaser = deepcopy(global_model)
            unlearned_federaser_model = run_fed_eraser(global_model=global_model_federaser,
                                                       weights_path=train_path,
                                                       clientwise_dataloaders=clientwise_dataloaders,
                                                       forget_clients=args.forget_clients,
                                                       num_clients=args.total_num_clients,
                                                       num_rounds=args.num_training_iterations,
                                                       device=args.device,
                                                       lr=args.lr,
                                                       optimizer_name=args.optimizer,
                                                       local_cali_round=1,
                                                       num_unlearn_rounds=1,
                                                       num_post_training_rounds=1)
            perf = get_performance(model=unlearned_federaser_model, test_dataloader=test_dataloader,
                                   clientwise_dataloader=clientwise_dataloaders, num_classes=num_classes,
                                   device=args.device)
            summary['performance']['after_federaser'] = perf
            if args.verbose:
                print(f"Performance after federaser : {perf}")
            # check backdoor on Federaser model
            if args.apply_backdoor:
                forget_backdoor_federaser = evaluate_backdoor_attack(model=unlearned_federaser_model, backdoor_context=backdoor_context,
                                                                     device=args.device)
                summary['backdoor_results']['after_federaser'] = forget_backdoor_federaser

                if args.verbose:
                    print(
                        f"Backdoor results after federaser : {forget_backdoor_federaser}")
            if args.apply_label_poisoning:
                forget_poisoning_federaser = evaluate_poisoning_attack(model=unlearned_federaser_model,
                                                                       poisoning_context=poisoning_context,
                                                                       device=args.device)
                summary['poisoning_results']['after_federaser'] = forget_poisoning_federaser

                if args.verbose:
                    print(
                        f"Poisoning results after federaser : {forget_poisoning_federaser}")

    unlearned_global_weights = unlearn_ours(global_model=global_model.cpu().state_dict(), forget_clients=args.forget_clients,
                                            total_num_clients=args.total_num_clients, weights_path=weights_path,
                                            dampening_constant=args.dampening_constant, dampening_upper_bound=args.dampening_upper_bound,
                                            ratio_cutoff=args.ratio_cutoff)
    unlearned_global_model = deepcopy(global_model)
    unlearned_global_model.load_state_dict(unlearned_global_weights)
    # check classwise accuracy
    perf = get_performance(model=unlearned_global_model, test_dataloader=test_dataloader,
                           clientwise_dataloader=clientwise_dataloaders, num_classes=num_classes,
                           device=args.device)
    summary['performance']['after_unlearning'] = perf
    if args.verbose:
        print(f"Performance after unlearning : {perf}")
    # check backdoor on unlearned model
    if args.apply_backdoor:
        forget_backdoor_attacks = evaluate_backdoor_attack(model=unlearned_global_model, backdoor_context=backdoor_context,
                                                           device=args.device)
        summary['backdoor_results']['after_unlearning'] = forget_backdoor_attacks

        if args.verbose:
            print(
                f"Backdoor results after unlearning : {forget_backdoor_attacks}")

    # check poisoning on unlearned model
    if args.apply_label_poisoning:
        forget_poisoning_attacks = evaluate_poisoning_attack(model=unlearned_global_model,
                                                             poisoning_context=poisoning_context,
                                                             device=args.device)
        summary['poisoning_results']['after_unlearning'] = forget_poisoning_attacks

        if args.verbose:
            print(
                f"Poisoning results after unlearning : {forget_poisoning_attacks}")
    # check mia precision and recall on all model
    summary['mia_attack'] = {}
    if args.apply_membership_inference:
        unlearning_mia_result = evaluate_mia_attack(target_model=deepcopy(unlearned_global_model),
                                                    attack_model=attack_model,
                                                    client_loaders=clientwise_dataloaders,
                                                    test_loader=test_dataloader,
                                                    dataset=args.dataset,
                                                    forget_client_idx=args.forget_clients[0],
                                                    device=args.device)
        summary['mia_attack']['after_unlearning'] = unlearning_mia_result
        if args.verbose:
            print(
                f"MIA results after unlearning : {unlearning_mia_result}")

        retrained_mia_result = evaluate_mia_attack(target_model=deepcopy(retrained_global_model),
                                                   attack_model=attack_model,
                                                   client_loaders=clientwise_dataloaders,
                                                   test_loader=test_dataloader,
                                                   dataset=args.dataset,
                                                   forget_client_idx=args.forget_clients[0],
                                                   device=args.device)
        summary['mia_attack']['after_retraining'] = retrained_mia_result
        if args.verbose:
            print(
                f"MIA results after retraining : {retrained_mia_result}")

        for baseline in baselines_methods:
            if baseline == 'pga':

                pga_mia_result = evaluate_mia_attack(target_model=deepcopy(unlearned_pga_model),
                                                     attack_model=attack_model,
                                                     client_loaders=clientwise_dataloaders,
                                                     test_loader=test_dataloader,
                                                     dataset=args.dataset,
                                                     forget_client_idx=args.forget_clients[0],
                                                     device=args.device)
                summary['mia_attack']['after_pga'] = pga_mia_result
                if args.verbose:
                    print(
                        f"MIA results after pga : {pga_mia_result}")
            elif baseline == 'fed_eraser':
                federaser_mia_result = evaluate_mia_attack(target_model=deepcopy(unlearned_federaser_model),
                                                           attack_model=attack_model,
                                                           client_loaders=clientwise_dataloaders,
                                                           test_loader=test_dataloader,
                                                           dataset=args.dataset,
                                                           forget_client_idx=args.forget_clients[0],
                                                           device=args.device)
                summary['mia_attack']['after_federaser'] = federaser_mia_result
                if args.verbose:
                    print(
                        f"MIA results after federaser : {federaser_mia_result}")

    # Add configurations to the summary
    summary['config'] = vars(args)

    # Create a timestamp for the summary file name
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Dump the summary into a file with the summary-timestamp name
    with open(os.path.join(weights_path, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
