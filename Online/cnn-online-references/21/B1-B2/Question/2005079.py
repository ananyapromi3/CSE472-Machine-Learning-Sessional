import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


from torch.optim.optimizer import Optimizer


class TensorAdaptiveSGD(Optimizer):
    def __init__(self, params, lr, weight_decay=0.0, adapt_const=0.1):

        defaults = dict(lr=lr, weight_decay=weight_decay,
                        adapt_const=adapt_const)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            adapt_const = group['adapt_const']

            for param in group['params']:
                if param.grad is None:
                    continue

                # TODO 1: Compute mean absolute gradient as a tensor
                mean_abs_g = param.grad.abs().mean()

                # TODO 2: Compute adaptive learning rate as a tensor
                adaptive_lr = lr / (1 + adapt_const * mean_abs_g)

                # TODO 3: Update the parameter using weighted SGD
                param.add_(-adaptive_lr * (param.grad + weight_decay * param))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# Load dataset
dataset = datasets.ImageFolder(
    root="images/",
    transform=transform
)

# DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=30,
    shuffle=True,
    num_workers=0
)

images, labels = next(iter(dataloader))

print("Batch image tensor shape:", images.shape)
print("Batch labels tensor shape:", labels.shape)

# Number of classes
num_classes = len(dataset.classes)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels):
        super(InceptionBlock, self).__init__()

        # TODO: define 1×1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # TODO: define 1×1 conv
        # TODO: define 3×3 conv
        self.branch2_1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.branch2_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            self.branch2_1,
            self.branch2_2
        )

        # TODO: define 1×1 conv
        # TODO: define 5×5 conv
        self.branch3_1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.branch3_2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            self.branch3_1,
            self.branch3_2
        )

        # TODO: define maxpool
        # TODO: define 1×1 conv
        self.branch4_1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
        self.branch4_2 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.branch4 = nn.Sequential(
            self.branch4_1,
            self.branch4_2
        )

    def forward(self, x):
        # TODO: branch 1 forward
        x1 = self.branch1(x)

        # TODO: branch 2 forward
        x2 = self.branch2(x)

        # TODO: branch 3 forward
        x3 = self.branch3(x)

        # TODO: branch 4 forward
        x4 = self.branch4(x)

        # TODO: Final output from inception block ,
        # You can use torch.cat to concatenate along channel dimension
        # Syntax: torch.cat([tensor1, tensor2, ...], dim=channel_dimension)
        x = torch.cat([x1, x2, x3, x4], dim=1)

        return x


class MiniInceptionNet(nn.Module):
    def __init__(self):
        super(MiniInceptionNet, self).__init__()

        # TODO: Initial stem convolutional layer
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=8, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # TODO: First Inception Block
        self.inc_block = InceptionBlock(64)

        # TODO: Downsampling convolutional layer
        self.downsample = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # TODO: Global average pooling and final FC layer
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 1)
        )

    def forward(self, x):

        # TODO : Stem convolutional layer
        x = self.init_conv(x)

        # TODO : First Inception Block
        x = self.inc_block(x)

        # TODO: Downsampling convolutional layer
        x = self.downsample(x)

        # TODO: Global average pooling and final FC layer
        x = self.gap(x)

        return x


device = torch.device("cpu")

model = MiniInceptionNet().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = TensorAdaptiveSGD(
    model.parameters(), lr=0.1, weight_decay=0.01, adapt_const=0.5)

num_epochs = 10

print(
    f"\nTraining Custom ResNet with {sum(p.numel() for p in model.parameters())} parameters")
print(f"Device: {device}\n")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.float().to(device)

        # Forward pass
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)

        predicted = (torch.sigmoid(outputs) > 0.5).long()
        total += labels.size(0)
        correct += (predicted == labels.long()).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")
