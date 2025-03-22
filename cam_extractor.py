import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
dfrom torch.utils.data import Dataset, DataLoader
from PIL import Image
import os


# Dataset class to load raster images
def convert_text_to_all_forms(input_txt_path, output_dir):
    with open(input_txt_path, 'r', encoding='utf-8') as file:
        text = file.read()

    os.makedirs(output_dir, exist_ok=True)

    raster_img = text_to_raster(text)
    raster_img.save(os.path.join(output_dir, 'raster.png'))

class RasterTextDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.file_list[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Simple CNN model for detecting patterns
class PatternDetector(nn.Module):
    def __init__(self):
        super(PatternDetector, self).__init__()
        self.features = models.resnet18(pretrained=False)
        self.features.fc = nn.Linear(self.features.fc.in_features, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.features(x)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Training loop
def train_model(data_dir, epochs=10, batch_size=8, learning_rate=1e-4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = RasterTextDataset(root_dir=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PatternDetector()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(epochs):
        for images in dataloader:
            labels = torch.ones(images.size(0), 1)  # placeholder
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), os.path.join(data_dir, 'pattern_detector.pth'))

# Example training call:
# train_model('output_visualisations')
