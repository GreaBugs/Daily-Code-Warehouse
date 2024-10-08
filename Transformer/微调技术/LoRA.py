import torch
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential


# LoRA Module for Conv2D
class LoRAConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, rank=4, stride=1, padding=0):
        super(LoRAConv2d, self).__init__()
        # Original convolutional layer (frozen)
        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # LoRA low-rank matrices (A and B)
        self.lora_A = nn.Conv2d(in_channels, rank, kernel_size=1, stride=1)
        self.lora_B = nn.Conv2d(rank, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        output = self.conv(x) + self.lora_B(self.lora_A(x))  # Adding low-rank approximation
        return output


# LoRA Module for Linear layers
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super(LoRALinear, self).__init__()
        # Original linear layer (frozen)
        self.linear = Linear(in_features, out_features)
        # LoRA low-rank matrices (A and B)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

    def forward(self, x):
        return self.linear(x) + self.lora_B(self.lora_A(x))


# Modified ShenYue network with LoRA
class ShenYue(nn.Module):
    def __init__(self):
        super(ShenYue, self).__init__()
        self.model1 = Sequential(
            LoRAConv2d(3, 32, 5, padding=2, stride=1),
            MaxPool2d(2),
            LoRAConv2d(32, 32, 5, padding=2, stride=1),
            MaxPool2d(2),
            LoRAConv2d(32, 64, 5, padding=2, stride=1),
            MaxPool2d(2),
            Flatten(),
            LoRALinear(1024, 64),
            LoRALinear(64, 10)
        )

    def forward(self, input):
        output = self.model1(input)
        return output


# Function to freeze original layers but allow LoRA to be trainable
def freeze_original_weights(model):
    for name, param in model.named_parameters():
        if 'lora_A' not in name and 'lora_B' not in name:
            param.requires_grad = False  # Freeze non-LoRA parameters


# Load pretrained weights and apply LoRA
def load_pretrained_and_apply_lora(model, pretrained_weights_path):
    # Load the pretrained model weights
    pretrained_dict = torch.load(pretrained_weights_path)
    model.load_state_dict(pretrained_dict, strict=False)  # Load weights (not strict due to LoRA layers)

    # Freeze the original weights except LoRA parameters
    freeze_original_weights(model)


# Example usage
shen_yue_model = ShenYue()

# Path to the pre-trained model weights
pretrained_weights_path = "path_to_pretrained_model_weights.pth"

# Load pretrained weights and apply LoRA adaptation
load_pretrained_and_apply_lora(shen_yue_model, pretrained_weights_path)

# Now, only the LoRA parameters are trainable, rest of the model is frozen
