
# ConDa: Fast Federated Unlearning with Contribution Dampening
[![Website](https://img.shields.io/badge/Project-Website-blue.svg)](https://respailab.github.io/ConDa-Federated-Unlearning/) 
[![Arvix](https://img.shields.io/badge/Arvix-2410.04144-darkred.svg)](https://arxiv.org/abs/2410.04144)




## Authors

[Vikram S. Chundawat](https://github.com/vikram2000b), [Pushkar Niroula](https://www.linkedin.com/in/pushkar-niroula/), [Prasanna Dhungana](https://www.linkedin.com/in/prasanna-dhungana/), [Stefan Schoepf](https://github.com/if-loops), [Murari Mandal](https://murarimandal.github.io/), [Alexandra Brintrup](blank)


## Overview
![Teaser](https://github.com/respailab/ConDa-Federated-Unlearning/blob/main/Assets/Teaser.png)

Federated Learning (FL) enables collaborative model training across decentralized data sources or clients. While adding new participants to a shared model is straightforward, removing a participant and their related information from the shared model remains challenging. To address this, we introduce Contribution Dampening (ConDa), a framework that performs efficient unlearning by tracking the parameters affected by each client and applying synaptic dampening to those influenced by the forgetting client. Our technique does not require clients' data or any retraining and imposes no computational overhead on either the client or server side.







## Installation

To set up the ConDa framework, follow these steps:

Clone the Repository:

```bash
git clone https://github.com/respailab/ConDa-Federated-Unlearning.git
cd ConDa-Federated-Unlearning
```
Set Up the Environment:

We recommend using a virtual environment to manage dependencies. You can set this up using venv:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```
Install Dependencies:

```bash
pip install -r requirements.txt
```
Ensure you have the necessary libraries installed as specified in the requirements.txt file.
    
## Usage
```bash
python main.py --exp_name <experiment_name> [options]
```
Arguments:

`--exp_name`: Specify the name of your experiment.
Optional Arguments:

`--exp_path`: Set the path to save experiment outputs. Default is ./experiments/.

`--model`: Choose the model architecture. Options are allcnn (default) or resnet18.

`--pretrained`: Use a pretrained model. Default is False.

`--dataset`: Select the dataset. Options are cifar10 (default) or mnist.

`--optimizer`: Choose the optimizer. Options are adam (default) or sgd.

`--lr`: Set the learning rate. Default is 0.001.

`--momentum`: Specify the momentum (applicable if using SGD). Default is 0.9.

`--weight_decay`: Set the weight decay (L2 regularization). Default is 0.0001.

`--batch_size`: Define the batch size for training. Default is 128.

`--num_local_epochs`: Number of local epochs per client. Default is 1.

`--num_training_iterations`: Total number of global training iterations. Default is 1.

`--num_participating_clients`: Number of clients participating in training. Use -1 to include all clients. Default is -1.

`--baselines`: Specify baseline methods for unlearning. Options are pga or fed_eraser. Default is an empty list.

`--apply_backdoor`: Apply a backdoor attack. Use this flag to enable.

`--backdoor_position`: Position of the backdoor trigger. Options are corner (default) or center.

`--num_backdoor_samples_per_forget_client`: Number of backdoor samples per client to forget. Default is 10.

`--backdoor_label`: Label assigned to backdoor samples. Default is 0.

`--apply_membership_inference`: Apply a membership inference attack. Default is False.

`--attack_type`: Type of attack for membership inference. Options are blackbox (default) or whitebox.

`--forget_clients`: Indexes of clients to forget. Default is [0].

`--total_num_clients`: Total number of clients in the federation. Default is 10.

`--client_data_distribution`: Distribution of client data. Options are dirichlet (default) or iid.

`--dampening_constant`: Dampening constant for unlearning. Default is 0.5.

`--dampening_upper_bound`: Upper bound for dampening. Default is 0.5.

`--ratio_cutoff`: Ratio cutoff value. Default is 0.5.

`--device`: Device to run the training on. Options are cpu (default) or cuda.

`--seed`: Random seed for reproducibility. Default is None.

`--verbose`: Enable verbose output. Default is True.

`--num_workers`: Number of workers for data loading. Default is 32.

### Example:

```bash
python main.py --exp_name experiment_name --dataset mnist --model allcnn --baselines pga fed_eraser 
--apply_backdoor --num_backdoor_samples_per_forget_client 500 --apply_membership_inference True 
--device cuda:0 --total_num_clients 10 --num_training_iterations 100 --dampening_constant 1.0 
--dampening_upper_bound 1.0 --pretrained True --ratio_cutoff 0.5 --seed 42
```

Note: For a comprehensive list of arguments and their descriptions, you can run:

```bash
python main.py --help
```
This command will display all available options with their default values and descriptions, providing a quick reference for configuring your experiments.

Ensure that you adjust the parameters to suit your specific requirements and experiment objectives.
## Acknowledgements

This research is supported by the Science and Engineering Research Board (SERB), India under Grant SRG/2023/001686.

## Citation
If you find this useful for your research, please cite the following:
```bibtex
@misc{chundawat2024condafastfederatedunlearning,
      title={ConDa: Fast Federated Unlearning with Contribution Dampening}, 
      author={Vikram S Chundawat and Pushkar Niroula and Prasanna Dhungana and Stefan Schoepf and Murari Mandal and Alexandra Brintrup},
      year={2024},
      eprint={2410.04144},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.04144}, 
}
```


