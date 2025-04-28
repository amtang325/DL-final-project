from typing import Tuple, Dict, Any
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
from art_loader import ArtLoader
from art_resnet import ArtResnet
from torch.optim import Optimizer
from torch.utils.data import DataLoader

def get_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr = config.get("lr", 1e-3), weight_decay = config.get("weight_decay", 1e-5))

def train_transforms(size: Tuple[int, int]) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.Resize(size),
        transforms.ToTensor()
    ])

def test_transforms(size: Tuple[int, int]) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    logit_classes = torch.argmax(logits, dim = 1)
    return torch.sum(torch.where(logit_classes == labels, 1, 0)) / len(labels)


def compute_loss(model: ArtResnet, model_output: torch.Tensor, target_labels: torch.Tensor, is_normalize: bool = True) -> torch.Tensor:
    loss = model.loss_criterion(model_output, target_labels)
    if is_normalize and model.loss_criterion.reduction != 'mean':
        loss = loss / len(target_labels)
    return loss

class Trainer:
    def __init__(
        self,
        data_dir: str,
        model: ArtResnet,
        optimizer: Optimizer,
        train_data_transforms: transforms.Compose,
        val_data_transforms: transforms.Compose,
        batch_size: int = 100,
    ) -> None:
        self.model = model

        self.train_dataset = ArtLoader(
            data_dir, split="train", transform=train_data_transforms
        )
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )

        self.val_dataset = ArtLoader(
            data_dir, split="test", transform=val_data_transforms
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=True
        )
        self.optimizer = optimizer

    def training_loop(self, num_epochs: int) -> None:
        for i in range(num_epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.test()
            print("Train Loss:", train_loss)
            print("Train Accuracy: ", train_acc)
            print("Validation Loss: ", val_loss)
            print("Validation Accuracy: ", val_acc)
    
    def inference_loop(self) -> None:
        test_loss, test_acc = self.test()
        print("Test Loss:", test_loss)
        print("Test Accuracy:", test_acc)

    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        train_loss_total = 0
        train_accuracy_total = 0
        total = 0
        for (x, y) in tqdm(self.train_loader):
            n = x.shape[0]
            logits = self.model(x)
            batch_acc = compute_accuracy(logits, y)
            train_accuracy_total += (batch_acc * n)

            batch_loss = compute_loss(self.model, logits, y, is_normalize=True)
            train_loss_total += (batch_loss * n)

            total += n

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

        return train_loss_total / total, train_accuracy_total / total

    def test(self) -> Tuple[float, float]:
        self.model.eval()
        test_accuracy_total = 0
        test_loss_total = 0
        total = 0
        for (x, y) in tqdm(self.val_loader):

            n = x.shape[0]
            logits = self.model(x)

            batch_acc = compute_accuracy(logits, y)
            test_accuracy_total += (batch_acc * n)

            batch_loss = compute_loss(self.model, logits, y, is_normalize=True)
            test_loss_total += (batch_loss * n)
            total += n

        return test_loss_total / total, test_accuracy_total / total
