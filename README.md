## Download WikiArt Data

Download data from wikiart into same root folder as code. Download from this GitHub repository using the first https URL (https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset).
Make sure the folder structure looks like the following
- root folder
  - data
    - wikiart
      - Abstract_Expressionism
      - Action_painting
      - Analytical_Cubism
      - ... etc
    - genre_class.txt
    - genre_train.csv
    - genre_val.csv
  - Unlearning.ipynb
  - art-classifier-weights.pt
  - art_loader.py
  - art_resnet.py
  - artist_split.py
  - loss_unlearn.py
  - loss_unlearn_plot.ipynb
  - main.py
  - trainer.py
  - unlearning_CIFAR10.ipynb

## How to Run Our (baseline) Art Classifier

1) Run code for model training or inference by typing python main.py in command line. If running training, comment out inference code. If running inference, comment out training code.

## Approach 1: How to Run Our Adversarial Approach

1) We started with the CIFAR-10 starter code in NeurIPS Machine Unlearning competition in Kaggle: https://www.kaggle.com/competitions/neurips-2023-machine-unlearning
This served as a starting point for experimenting with our adversarial unlearning, and inspired the initial code for training the model conducting Membership Inference Attacks (MIAs). The starter kit from the competition
with our unlearning process can be found in unlearning_CIFAR10.ipynb. Sections written by us are clearly marked as many of the evaluation and data loading sections were pre-supplied. Run all cells to replicate results.

2) The adversarial approach was then applied to the WikiArt dataset. The code for loading the art data, pre-training, and unlearning can be found in Unlearning.ipynb. After downloading WikiArt Data into the same root folder, run all cells to replicate results.

## Approach 2: How to Run Our Loss Function Approach
