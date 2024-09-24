import os
import json
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import segmentation_models_pytorch as smp
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

# 1. COCO Segmentation Dataset 로더 정의
class COCOSegmentationDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]

        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)

        self.annotations = self.coco_data['annotations']
        self.categories = {category['id']: category['name'] for category in self.coco_data['categories']}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image_id = self.get_image_id(img_name)
        annotations = [ann for ann in self.annotations if ann['image_id'] == image_id]
        mask = self.create_ocean_non_ocean_mask(annotations, image.size)

        if self.transform:
            image = self.transform(image)
            mask = torch.tensor(mask, dtype=torch.long)

        return image, mask

    def get_image_id(self, img_name):
        for image_info in self.coco_data['images']:
            if image_info['file_name'] == img_name:
                return image_info['id']
        raise ValueError(f"Image ID for {img_name} not found.")

    def create_ocean_non_ocean_mask(self, annotations, image_size):
        mask = np.zeros(image_size[::-1], dtype=np.uint8)
        for ann in annotations:
            category_name = self.categories[ann['category_id']]
            if category_name == 'ocean':
                segmentation = ann['segmentation']
                for polygon in segmentation:
                    poly_points = np.array(polygon).reshape((-1, 2))
                    cv2.fillPoly(mask, [np.int32(poly_points)], 1)
            else:
                segmentation = ann['segmentation']
                for polygon in segmentation:
                    poly_points = np.array(polygon).reshape((-1, 2))
                    cv2.fillPoly(mask, [np.int32(poly_points)], 0)
        return mask

# 2. U-Net 모델 설정
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=2,
)

# 3. 데이터 로더 설정
train_dataset = COCOSegmentationDataset(
    image_dir="/home/work/yolotest/data/train",
    annotation_file="/home/work/yolotest/data/train/_annotations.coco.json",
    transform=transforms.Compose([transforms.ToTensor()])
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

valid_dataset = COCOSegmentationDataset(
    image_dir="/home/work/yolotest/data/valid",
    annotation_file="/home/work/yolotest/data/valid/_annotations.coco.json",
    transform=transforms.Compose([transforms.ToTensor()])
)

valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

# 4. 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Learning Rate Scheduler 설정
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# 6. 모델 학습 루프
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(10):  # 에포크 수 증가
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(train_loader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        masks = masks.squeeze(1).long()
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/20], Loss: {running_loss / len(train_loader)}")

    # 검증 루프에서 성능 측정
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for images, masks in valid_loader:
            images = images.to(device)
            masks = masks.to(device)
            masks = masks.squeeze(1).long()
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

    val_loss /= len(valid_loader)
    print(f"Validation Loss: {val_loss}")
    
    # 스케줄러로 학습률 조정
    scheduler.step(val_loss)

# 7. 모델 저장
torch.save(model.state_dict(), "/home/work/yolotest/unet_ocean_segmentation.pth")
