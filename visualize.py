import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from dataset import OmamaDataset
from torch.utils.data import DataLoader, random_split

# 1. Setup device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 2. Reload the exact same Validation dataset split we used in training
data_dir = "/Users/ab/Downloads/256x256_studylevel"
dataset = OmamaDataset(data_dir)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
# We use a fixed seed so we get the exact same validation images
_, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# 3. Rebuild the empty model and load your saved weights
model = models.resnet18()
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)
model.load_state_dict(torch.load("mammogram_model.pth"))
model = model.to(device)
model.eval()

# 4. Grab one batch of images
images, labels = next(iter(val_loader))
images = images.unsqueeze(1).to(device)
labels = labels.unsqueeze(1).to(device, dtype=torch.float32)

# 5. Ask the AI for predictions
with torch.no_grad():
    outputs = model(images)
    # Convert raw output to a probability between 0 and 1 using Sigmoid
    probabilities = torch.sigmoid(outputs)
    predicted = (probabilities > 0.5).float()

# 6. Plot the images and the predictions
fig, axes = plt.subplots(1, 4, figsize=(15, 5))
for i in range(4):
    img = images[i].cpu().squeeze()
    true_label = int(labels[i].cpu().item())
    pred_prob = probabilities[i].cpu().item()
    pred_label = int(predicted[i].cpu().item())
    
    axes[i].imshow(img, cmap='gray')
    
    # Color the title green if correct, red if wrong
    color = 'green' if true_label == pred_label else 'red'
    axes[i].set_title(f"Truth: {true_label}\nPred: {pred_label} ({pred_prob*100:.1f}%)", color=color)
    axes[i].axis('off')

plt.tight_layout()
plt.show()