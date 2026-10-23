import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
from dataset import OmamaDataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

data_dir = "/Users/ab/Downloads/256x256_studylevel"
dataset = OmamaDataset(data_dir, crop_size=128)
print(f"Found {len(dataset)} valid studies (4 views each).")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True) # Lowered batch size to 16 for memory
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# CHANGE HERE: Accept 4 channels for the 4 views!
model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 3

for epoch in range(EPOCHS):
    print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
    model.train()
    running_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        # images are already (Batch, 4, 128, 128) so we don't need unsqueeze!
        images = images.to(device)
        labels = labels.unsqueeze(1).to(device, dtype=torch.float32)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if batch_idx % 50 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

print("\nEvaluating on validation set...")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.unsqueeze(1).to(device, dtype=torch.float32)
        outputs = model(images)
        predicted = (outputs > 0.0).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\nFinal Validation Accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), "mammogram_multiview_model.pth")
print("Model saved as 'mammogram_multiview_model.pth'!")
