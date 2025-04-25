import torch
import torch.nn as nn
from torchvision.models import resnet50


class ArtResnet(nn.Module):
    def __init__(self):
        super().__init__()
        model = resnet50(pretrained = True)
        layers = list(model.children())
        self.conv_layers = nn.Sequential(
            layers[0],
            layers[1],
            layers[2],
            layers[3],
            layers[4],
            layers[5],
            layers[6],
        )

        for p in self.conv_layers.parameters():
            p.requires_grad = False

        self.fc_layers = nn.Sequential(
            layers[7],
            layers[8],
            nn.Flatten(),
            nn.Linear(in_features = 2048, out_features = 10, bias = True)
        )
        
        self.loss_criterion = nn.CrossEntropyLoss(reduction = 'mean')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        model_output = self.conv_layers(x)
        model_output = self.fc_layers(model_output)
        return model_output
