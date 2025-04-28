## Download WikiArt Data


## How to Run Our (baseline) Art Classifier



## Approach 1: How to Run Our Adversarial Approach

1) We started with the CIFAR-10 starter code in NeurIPS Machine Unlearning competition in Kaggle: https://www.kaggle.com/competitions/neurips-2023-machine-unlearning
This served as a starting point for experimenting with our adversarial unlearning, and inspired the initial code for training the model conducting Membership Inference Attacks (MIAs). The starter kit from the competition
with our unlearning process can be found in unlearning_CIFAR10.ipynb. Sections written by us are clearly marked as many of the evaluation and data loading sections were pre-supplied. Run all cells to replicate results.

2) The adversarial approach was then applied to the WikiArt dataset. The code for loading the art data, pre-training, and unlearning can be found in Unlearning.ipynb. After downloading WikiArt Data into the same root folder, run all cells to replicate results.

## Approach 2: How to Run Our Loss Function Approach

