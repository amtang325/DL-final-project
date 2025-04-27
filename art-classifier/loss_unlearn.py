import os
import sys

import torch
from torch.utils.data import SubsetRandomSampler, DataLoader

from typing import Tuple, Dict, Any
from tqdm import tqdm

from trainer import (
    Trainer, 
    get_optimizer,
    train_transforms,
    test_transforms,
    compute_accuracy,
    compute_loss
)
from art_resnet import ArtResnet
from art_loader import ArtLoader
from art_resnet import ArtResnet
from torch.optim import Optimizer
from torch.utils.data import DataLoader

def unlearn_loss(data_dir, model, optimizer, batch_size, num_epochs):
    train_data_transforms = train_transforms(size=(224, 224))
    val_data_transforms = test_transforms(size=(224, 224))
    
    train_dataset = ArtLoader(data_dir, split="train", transform=train_data_transforms, train_file="genre_train_with_dali.csv")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = ArtLoader(data_dir, split="test", transform=val_data_transforms, test_file="genre_val_with_dali.csv")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    for i in range(num_epochs):
        train_loss, train_acc = train_unlearn_epoch(model, train_loader, optimizer)
        val_loss, val_acc = test(model, val_loader)
        print("Train Loss (Unlearn):", train_loss)
        print("Train Accuracy (Unlearn): ", train_acc)
        print("Validation Loss (Unlearn): ", val_loss)
        print("Validation Accuracy (Unlearn): ", val_acc)    

def learn_loss(data_dir, model, optimizer, batch_size, num_epochs):
    train_data_transforms = train_transforms(size=(224, 224))
    val_data_transforms = test_transforms(size=(224, 224))
    
    train_dataset = ArtLoader(data_dir, split="train", transform=train_data_transforms, train_file="genre_train_with_dali.csv")

    num_samples = 9600
    sampler = SubsetRandomSampler(indices=list(range(len(train_dataset)))[:num_samples])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    val_dataset = ArtLoader(data_dir, split="test", transform=val_data_transforms, test_file="genre_val_with_dali.csv")

    num_samples = 4800
    sampler = SubsetRandomSampler(indices=list(range(len(val_dataset)))[:num_samples])
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=sampler)
    
    for i in range(num_epochs):
        train_loss, train_acc = train_learn_epoch(model, train_loader, optimizer)
        val_loss, val_acc = test(model, val_loader)
        print("Train Loss (Learn):", train_loss)
        print("Train Accuracy (Learn): ", train_acc)
        print("Validation Loss (Learn): ", val_loss)
        print("Validation Accuracy (Learn): ", val_acc) 

def train_unlearn_epoch(model, train_loader, optimizer) -> Tuple[float, float]:
    model.train()
    train_loss_total = 0
    train_accuracy_total = 0
    total = 0
    for (x, y) in tqdm(train_loader):
        n = x.shape[0]
        logits = model(x)
        batch_acc = compute_accuracy(logits, y)
        train_accuracy_total += (batch_acc * n)

        batch_loss = compute_loss(model, logits, y, is_normalize=True)
        neg_loss = -batch_loss
        train_loss_total += (batch_loss * n)

        total += n

        optimizer.zero_grad()
        neg_loss.backward()
        optimizer.step()

    return train_loss_total / total, train_accuracy_total / total

def train_learn_epoch(model, train_loader, optimizer) -> Tuple[float, float]:
    model.train()
    train_loss_total = 0
    train_accuracy_total = 0
    total = 0
    for (x, y) in tqdm(train_loader):
        n = x.shape[0]
        logits = model(x)
        batch_acc = compute_accuracy(logits, y)
        train_accuracy_total += (batch_acc * n)

        batch_loss = compute_loss(model, logits, y, is_normalize=True)
        train_loss_total += (batch_loss * n)

        total += n

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    return train_loss_total / total, train_accuracy_total / total

def inference_loop_unlearn(data_dir, model, batch_size) -> None:
        val_data_transforms = test_transforms(size=(224, 224))

        val_dataset = ArtLoader(data_dir, split="test", transform=val_data_transforms, test_file="genre_val_with_dali.csv")
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        test_loss, test_acc = test(model, val_loader)
        print("Test Loss (Unlearn):", test_loss)
        print("Test Accuracy (Unlearn):", test_acc)

def inference_loop_learn(data_dir, model, batch_size) -> None:
        val_data_transforms = test_transforms(size=(224, 224))
        
        val_dataset = ArtLoader(data_dir, split="test", transform=val_data_transforms, test_file="genre_val_without_dali.csv")

        num_samples = 4800
        sampler = SubsetRandomSampler(indices=list(range(len(val_dataset)))[:num_samples])

        val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=sampler)

        test_loss, test_acc = test(model, val_loader)
        print("Test Loss (Learn):", test_loss)
        print("Test Accuracy (Learn):", test_acc)

def test(model, val_loader) -> Tuple[float, float]:
    model.eval()
    test_accuracy_total = 0
    test_loss_total = 0
    total = 0
    for (x, y) in tqdm(val_loader):

        n = x.shape[0]
        logits = model(x)

        batch_acc = compute_accuracy(logits, y)
        test_accuracy_total += (batch_acc * n)

        batch_loss = compute_loss(model, logits, y, is_normalize=True)
        test_loss_total += (batch_loss * n)
        total += n

    return test_loss_total / total, test_accuracy_total / total

def main():
    checkpoint = torch.load('unlearn_art_loss_function_0.pt')
    my_resnet = ArtResnet()
    my_resnet.load_state_dict(checkpoint['model_state_dict'])
    optimizer_config = {"optimizer_type": "adam", "lr": 1e-3, "weight_decay": 1e-5}

    #Stage 0
    print("Stage 0")
    inference_loop_unlearn(data_dir="./data/", model=my_resnet, batch_size=32)
    inference_loop_learn(data_dir="./data/", model=my_resnet, batch_size=32)

    #Stage 1
    print("Stage 1")
    optimizer = get_optimizer(my_resnet, optimizer_config)
    unlearn_loss(model=my_resnet, optimizer=optimizer, data_dir="./data/", batch_size=32, num_epochs=5)
    torch.save({'model_state_dict': my_resnet.state_dict(), 'optimizer_state_dict':optimizer.state_dict()}, './unlearn_art_loss_function_1.pt')
    inference_loop_learn(data_dir="./data/", model=my_resnet, batch_size=32)

    #Stage 2
    print("Stage 2")
    checkpoint = torch.load('unlearn_art_loss_function_1.pt')
    my_resnet = ArtResnet()
    my_resnet.load_state_dict(checkpoint['model_state_dict'])
    optimizer = get_optimizer(my_resnet, optimizer_config)
    learn_loss(model=my_resnet, optimizer=optimizer, data_dir="./data/", batch_size=32, num_epochs=1)
    torch.save({'model_state_dict': my_resnet.state_dict(), 'optimizer_state_dict':optimizer.state_dict()}, './unlearn_art_loss_function_2.pt')
    inference_loop_unlearn(data_dir="./data/", model=my_resnet, batch_size=32)
    
    #Stage 3
    print("Stage 3")
    checkpoint = torch.load('unlearn_art_loss_function_2.pt')
    my_resnet = ArtResnet()
    my_resnet.load_state_dict(checkpoint['model_state_dict'])
    optimizer = get_optimizer(my_resnet, optimizer_config)
    unlearn_loss(model=my_resnet, optimizer=optimizer, data_dir="./data/", batch_size=32, num_epochs=5)
    torch.save({'model_state_dict': my_resnet.state_dict(), 'optimizer_state_dict':optimizer.state_dict()}, './unlearn_art_loss_function_3.pt')
    inference_loop_unlearn(data_dir="./data/", model=my_resnet, batch_size=32)
    
    #Stage 4
    print("Stage 4")
    checkpoint = torch.load('unlearn_art_loss_function_3.pt')
    my_resnet = ArtResnet()
    my_resnet.load_state_dict(checkpoint['model_state_dict'])
    optimizer = get_optimizer(my_resnet, optimizer_config)
    learn_loss(model=my_resnet, optimizer=optimizer, data_dir="./data/", batch_size=32, num_epochs=1)
    torch.save({'model_state_dict': my_resnet.state_dict(), 'optimizer_state_dict':optimizer.state_dict()}, './unlearn_art_loss_function_4.pt')
    inference_loop_learn(data_dir="./data/", model=my_resnet, batch_size=32)

if __name__ == "__main__":
    main()