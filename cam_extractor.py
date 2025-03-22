import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchcam.methods import GradCAMpp
import numpy as np

MAX_RASTER_WIDTH = 2000
MAX_RASTER_HEIGHT = 10000

def text_to_raster(input_file, output_file, char_pixel_size=1, max_width=MAX_RASTER_WIDTH):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    lines = [line.rstrip('\n') for line in lines][:MAX_RASTER_HEIGHT]
    max_len = min(max(len(line) for line in lines), max_width)

    height = len(lines) * char_pixel_size
    width = max_len * char_pixel_size
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    for y, line in enumerate(lines):
        for x, ch in enumerate(line[:max_len]):
            if ch != ' ':
                draw.rectangle(
                    [x * char_pixel_size, y * char_pixel_size,
                     (x + 1) * char_pixel_size, (y + 1) * char_pixel_size],
                    fill=(0, 0, 0))
    img.save(output_file)

class PatternDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = models.resnet18(weights=None)
        self.features.fc = nn.Linear(self.features.fc.in_features, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x.requires_grad_()  # ensure gradients flow
        x = self.features(x)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def train_dummy_model(output_path):
    model = PatternDetector()
    dummy_input = torch.rand(8, 3, 224, 224, requires_grad=True)
    dummy_labels = torch.ones(8, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    for _ in range(5):
        outputs = model(dummy_input)
        loss = criterion(outputs, dummy_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), output_path)

def analyze_text_raster(raster_path, model_path):
    model = PatternDetector()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img = Image.open(raster_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    img_tensor.requires_grad_()  # critical for hook registration

    cam_extractor = GradCAMpp(model, target_layer='features.layer4')
    output = model(img_tensor)
    cam_map = cam_extractor(class_idx=0, scores=output)[0].cpu().numpy()
    print(f'Pattern score: {output.item()}')

    heatmap_rgb = (plt.cm.jet(cam_map / (cam_map.max() + 1e-8))[:, :, :3] * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap_rgb).resize(img.size)
    blended = Image.blend(img.convert('RGBA'), heatmap_img.convert('RGBA'), alpha=0.5)
    blended.save('gradcam_overlay.png')

if __name__ == '__main__':
    if not os.path.exists('pattern_detector.pth'):
        print('Model not found. Training dummy model...')
        train_dummy_model('pattern_detector.pth')
    text_to_raster('finnegans_wake.txt', 'raster.png')
    analyze_text_raster('raster.png', 'pattern_detector.pth')
    print('Done: raster.png & gradcam_overlay.png ready.')
