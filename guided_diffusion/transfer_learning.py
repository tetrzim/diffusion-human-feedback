from torchvision import models
import torch
import torch.nn as nn
from . import dist_util


def create_ResNet18_model(
    pretrained_weight = True,
    transfer_learning = True,
):
    """
    If pretrained_weight is set to True, load the pretrained weights.
    If transfer_learning is set to True, the gradient options for all layers except the last fully connected layer are frozen.
    """
    if pretrained_weight:
        model = models.resnet18(weights="DEFAULT")
    else:
        model = models.resnet18()

    if transfer_learning:
        for param in model.parameters():
            param.requires_grad = False
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs,1)
    
    return model


def create_pretrained_model(
    model_name="resnet18",
    output_dim=1,
    model_name_list=['resnet18', 'resnet34', 'alexnet', 'densenet', 'mobilenet'],
    ckpt=["logs/LSUN_reward/transfer_learning/100,150_samples_pos_weight_0.1_resnet18_2/model002000.pt", "logs/LSUN_reward/transfer_learning/100,150_samples_pos_weight_0.1_resnet34/model009999.pt", "logs/LSUN_reward/transfer_learning/100,150_samples_pos_weight_0.1_alexnet/model006000.pt", "logs/LSUN_reward/transfer_learning/100,150_samples_pos_weight_0.1_densenet/model000500.pt", "logs/LSUN_reward/transfer_learning_2/100,150_samples_pos_weight_0.1_mobilenet/model001000.pt"]
):

    if model_name == "resnet18":
        model = models.resnet18(weights="DEFAULT")
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, output_dim)
    elif model_name == "resnet34":
        model = models.resnet34(weights="DEFAULT")
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, output_dim)
    elif model_name == "alexnet":
        model = models.alexnet(weights="DEFAULT")
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, output_dim)
    elif model_name == "densenet":
        model = models.densenet121(weights="DEFAULT")
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, output_dim)
    elif model_name == "efficientnet":
        model = models.efficientnet_b3(weights="DEFAULT")
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, output_dim)
    elif model_name == "mobilenet":
        model = models.mobilenet_v3_large(weights="DEFAULT")
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, output_dim)
    elif model_name == "googlenet":
        model = models.googlenet(weights="DEFAULT")
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, output_dim)
    elif model_name == "vgg11":
        model = models.vgg11(weights="DEFAULT")
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, output_dim)
    elif model_name == "ensemble2":
        model_list = []
        for i, name in enumerate(model_name_list):
            model = create_pretrained_model(name)
            model.load_state_dict(
                dist_util.load_state_dict(ckpt[i], map_location="cpu")
            )
            model.to(dist_util.dev())
            model.eval()
            model_list.append(model)
        return ensemble_2(model_list)
    elif model_name == "ensemble4":
        model_list = []
        for i, name in enumerate(model_name_list):
            model = create_pretrained_model(name)
            model.load_state_dict(
                dist_util.load_state_dict(ckpt[i], map_location="cpu")
            )
            model.to(dist_util.dev())
            model.eval()
            model_list.append(model)
        return ensemble_4(model_list)
    else:
        raise ValueError("invalid model name")

    return model


class ensemble_2(nn.Module):

    def __init__(self,ft_list):
        super().__init__()
        self.ft_list = ft_list

    def forward(self,x):
        out_list = [ft(x) for ft in self.ft_list]
        return sum(out_list) / len(out_list)
        

class ensemble_4(nn.Module):

    def __init__(self, ft_list):
        super().__init__()
        self.ft_list = ft_list

    def forward(self, x):
        out_list = [ft(x) for ft in self.ft_list]
        out_tensor = torch.concat(out_list, 1)
        out = torch.max(out_tensor, -1).values.reshape(-1, 1)
        return out


def sample_sort(model, samples):
    logits = torch.flatten(model(samples))
    index = torch.sort(logits)[1]

    samples = samples[index]

    return samples
