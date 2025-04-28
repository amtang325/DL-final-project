import os
import sys

import torch

from trainer import (
    Trainer, 
    get_optimizer,
    train_transforms,
    test_transforms
)
from art_resnet import ArtResnet

size = (224, 224)
optimizer_config = {"optimizer_type": "adam", "lr": 1e-3, "weight_decay": 1e-5}
checkpoint = torch.load('art-classifier-weights.pt')
my_resnet = ArtResnet()
my_resnet.load_state_dict(checkpoint['model_state_dict'])
optimizer = get_optimizer(my_resnet, optimizer_config)


trainer = Trainer(
    data_dir="./data/",
    model=my_resnet,
    optimizer=optimizer,
    train_data_transforms=train_transforms(size),
    val_data_transforms=test_transforms(size),
    batch_size=32
)

# Training
trainer.training_loop(num_epochs=1)
torch.save({'model_state_dict': my_resnet.state_dict(), 'optimizer_state_dict':optimizer.state_dict()}, './art-classifier-weights.pt')

# Inference
# trainer.inference_loop()