import os
import pandas as pd
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from image_processor import ImageProcessor  # 사용자 제공 블러 클래스

# 데이터셋 정의
class FocusBlurDataset(Dataset):
    def __init__(self, csv_file, transform=None, num_samples=None):
        # CSV 파일 로드 및 샘플 수 제한
        data = pd.read_csv(csv_file, header=None)
        if num_samples and num_samples < len(data):
            data = data.sample(n=num_samples, random_state=42).reset_index(drop=True)

        self.rgb_paths = data[0].tolist()
        self.depth_paths = data[1].tolist()
        self.transform = transform
        self.processor = ImageProcessor()  # 블러 처리 클래스 초기화

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        # 이미지 및 깊이맵 경로
        img_path = self.rgb_paths[idx]
        depth_path = self.depth_paths[idx]

        # 이미지 로드
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)  # (H, W, 3)
        depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)  # (H, W)

        # 블러 이미지 생성 및 채널 확장
        blurred_image = self.processor.process_image(image)
        if len(blurred_image.shape) == 2:  # 단일 채널인 경우
            blurred_image = np.expand_dims(blurred_image, axis=-1)  # (H, W, 1)
        blurred_image = np.repeat(blurred_image, 3, axis=-1)  # 3채널로 확장 (H, W, 3)

        # 원본 이미지와 블러 이미지 결합
        combined_image = np.concatenate((image, blurred_image), axis=-1)  # (H, W, 6)

        # 데이터 확인
        # print(f"Original image shape: {image.shape}")  # 예: (480, 640, 3)
        # print(f"Blurred image shape: {blurred_image.shape}")  # 예: (480, 640, 3)
        # print(f"Combined image shape before transform: {combined_image.shape}")  # 예: (480, 640, 6)

        # PyTorch 텐서 변환
        if self.transform:
            # combined_image는 (H, W, C) -> (C, H, W)로 변환 후 transform 적용
            combined_image = combined_image.transpose(2, 0, 1)  # (C, H, W)
            combined_image = torch.tensor(combined_image).float() / 255.0  # 정규화

            # depth_map은 (H, W) -> (1, H, W)로 변환 후 transform 적용
            depth_map = np.expand_dims(depth_map, axis=0)  # (H, W) -> (1, H, W)
            depth_map = torch.tensor(depth_map).float()

        # 최종 형태 확인
        # print(f"Combined image shape after transform: {combined_image.shape}")  # 예: (6, H, W)
        # print(f"Depth map shape after transform: {depth_map.shape}")  # 예: (1, H, W)

        return combined_image, depth_map



# Attention Module 정의
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size // 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)
        
class AttentionUNet(nn.Module):
    def __init__(self):
        super(AttentionUNet, self).__init__()
        self.enc1 = self._conv_block(6, 64)  # 6채널 입력 (수정된 입력 채널 수)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)

        self.middle = self._conv_block(512, 1024)

        self.dec4 = self._conv_block(1024 + 512, 512)
        self.dec3 = self._conv_block(512 + 256, 256)
        self.dec2 = self._conv_block(256 + 128, 128)
        self.dec1 = self._conv_block(128 + 64, 64)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

        self.channel_attention = ChannelAttention(1, reduction=1)  # 수정된 입력 채널 및 reduction
        self.spatial_attention = SpatialAttention()

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # print(f"Input Shape: {x.shape}")

        enc1 = self.enc1(x)
        # print(f"After enc1: {enc1.shape}")

        enc2 = self.enc2(nn.MaxPool2d(2)(enc1))
        # print(f"After enc2: {enc2.shape}")

        enc3 = self.enc3(nn.MaxPool2d(2)(enc2))
        # print(f"After enc3: {enc3.shape}")

        enc4 = self.enc4(nn.MaxPool2d(2)(enc3))
        # print(f"After enc4: {enc4.shape}")

        middle = self.middle(nn.MaxPool2d(2)(enc4))
        # print(f"After middle: {middle.shape}")

        dec4 = self.dec4(torch.cat((middle, self._resize(enc4, middle)), dim=1))
        # print(f"After dec4: {dec4.shape}")

        dec3 = self.dec3(torch.cat((dec4, self._resize(enc3, dec4)), dim=1))
        # print(f"After dec3: {dec3.shape}")

        dec2 = self.dec2(torch.cat((dec3, self._resize(enc2, dec3)), dim=1))
        # print(f"After dec2: {dec2.shape}")

        dec1 = self.dec1(torch.cat((dec2, self._resize(enc1, dec2)), dim=1))
        # print(f"After dec1: {dec1.shape}")

        output = self.final_conv(dec1)
        # print(f"After final_conv: {output.shape}")

        output = self.channel_attention(output)
        # print(f"After channel_attention: {output.shape}")

        output = self.spatial_attention(output)
        # print(f"After spatial_attention: {output.shape}")

        # Upsample the output to match the target size
        # output = torch.nn.functional.interpolate(output, size=(512, 512), mode="bilinear", align_corners=False)
        # print(f"After upsampling: {output.shape}")
        output = torch.nn.functional.interpolate(output, size=(480, 640), mode="bilinear", align_corners=False)
        # print(f"After upsampling: {output.shape}")
        return output


    def _resize(self, enc_out, target):
        return torch.nn.functional.interpolate(enc_out, size=target.shape[2:], mode="bilinear", align_corners=False)

# 학습 중 Validation 시각화 및 저장
import matplotlib.pyplot as plt
import os
def visualize_and_save(epoch, inputs, targets, predictions, save_dir="V2"):
    os.makedirs(save_dir, exist_ok=True)
    batch_size = inputs.shape[0]

    fig, axes = plt.subplots(4, 3, figsize=(15, 20))

    for i in range(4):  # 최대 4개 샘플만 시각화
        input_vis = inputs[i].permute(1, 2, 0).cpu().numpy()
        if input_vis.shape[2] > 3:
            input_vis = input_vis[:, :, :3]  # Take first 3 channels
        elif input_vis.shape[2] == 1:
            input_vis = np.repeat(input_vis, 3, axis=2)  # Expand single channel to 3

        axes[i, 0].imshow(input_vis.astype(np.uint8))
        axes[i, 0].set_title("Input")
        axes[i, 1].imshow(targets[i].squeeze().cpu().numpy(), cmap="gray")
        axes[i, 1].set_title("Target")
        axes[i, 2].imshow(predictions[i].squeeze().detach().cpu().numpy(), cmap="gray")
        axes[i, 2].set_title("Prediction")

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch}_validation_samples.png"))
    plt.close()

# 손실 함수 정의
class DepthLoss(nn.Module):
    def __init__(self, grid_size=(4, 4)):
        super(DepthLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.grid_size = grid_size

    def forward(self, pred, target):
        # 깊이 손실 계산
        l_depth = self.l1_loss(pred, target)

        # 전역 경사 손실 계산
        grad_x_pred, grad_y_pred = self._compute_gradients(pred)
        grad_x_target, grad_y_target = self._compute_gradients(target)
        l_grad_global = self.l1_loss(grad_x_pred, grad_x_target) + self.l1_loss(grad_y_pred, grad_y_target)

        # 격자 기반 경사 손실 계산
        l_grad_grid = self._compute_grid_gradient_loss(pred, target)

        # SSIM 손실 계산
        l_ssim = 1 - self._ssim(pred, target)

        # 최종 손실
        return l_depth + l_grad_global + 0.3*l_grad_grid + l_ssim

    def _compute_gradients(self, x):
        grad_x = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
        grad_y = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        return grad_x, grad_y

    def _compute_grid_gradient_loss(self, pred, target):
        b, c, h, w = pred.shape
        grid_h, grid_w = h // self.grid_size[0], w // self.grid_size[1]

        total_loss = 0.0
        for i in range(0, h, grid_h):
            for j in range(0, w, grid_w):
                # 그리드 슬라이싱
                pred_grid = pred[:, :, i:i+grid_h, j:j+grid_w]
                target_grid = target[:, :, i:i+grid_h, j:j+grid_w]

                # 그리드 경사 계산
                grad_x_pred, grad_y_pred = self._compute_gradients(pred_grid)
                grad_x_target, grad_y_target = self._compute_gradients(target_grid)

                # 그리드 내 L1 손실 합산
                grid_loss = self.l1_loss(grad_x_pred, grad_x_target) + self.l1_loss(grad_y_pred, grad_y_target)
                total_loss += grid_loss

        # 배치 평균으로 정규화
        return total_loss / b

    def _ssim(self, x, y):
        mu_x = torch.mean(x, dim=(2, 3), keepdim=True)
        mu_y = torch.mean(y, dim=(2, 3), keepdim=True)
        sigma_x = torch.var(x, dim=(2, 3), keepdim=True)
        sigma_y = torch.var(y, dim=(2, 3), keepdim=True)
        sigma_xy = ((x - mu_x) * (y - mu_y)).mean(dim=(2, 3), keepdim=True)
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
        return torch.clamp((1 - ssim) / 2, 0, 1).mean()

import os

def train_model(csv_file, num_epochs=10, batch_size=30, lr=0.009, num_samples=2000):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512))
    ])
    dataset = FocusBlurDataset(csv_file, transform=transform, num_samples=num_samples)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionUNet().to(device)
    criterion = DepthLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Validation Loss를 추적하기 위한 변수
    best_val_loss = float('inf')
    save_dir = "saved_modelsV2"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        for images, depths in tqdm(train_loader, desc="Training"):
            images, depths = images.to(device), depths.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, depths)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        print(f"Train Loss: {train_loss:.4f}")

        # Validation 단계
        model.eval()
        val_loss = 0.0
        for images, depths in tqdm(val_loader, desc="Validation"):
            images, depths = images.to(device), depths.to(device)
            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, depths)
                val_loss += loss.item() * images.size(0)
            visualize_and_save(epoch + 1, images.cpu(), depths.cpu(), outputs.cpu())

        val_loss /= len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}")

        # 최저 Validation Loss 갱신 시 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(save_dir, f"best_model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved at {save_path}")

if __name__ == '__main__':
    csv_file = 'data/nyu2_train.csv'
    train_model(csv_file, num_epochs=50, num_samples=20000)

