import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from matplotlib import pyplot as plt
from image_processor import ImageProcessor  # 사용자 제공 블러 클래스

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
        self.enc1 = self._conv_block(6, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)

        self.middle = self._conv_block(512, 1024)

        self.dec4 = self._conv_block(1024 + 512, 512)
        self.dec3 = self._conv_block(512 + 256, 256)
        self.dec2 = self._conv_block(256 + 128, 128)
        self.dec1 = self._conv_block(128 + 64, 64)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

        self.channel_attention = ChannelAttention(1, reduction=1)
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
        enc1 = self.enc1(x)
        enc2 = self.enc2(nn.MaxPool2d(2)(enc1))
        enc3 = self.enc3(nn.MaxPool2d(2)(enc2))
        enc4 = self.enc4(nn.MaxPool2d(2)(enc3))

        middle = self.middle(nn.MaxPool2d(2)(enc4))

        dec4 = self.dec4(torch.cat((middle, self._resize(enc4, middle)), dim=1))
        dec3 = self.dec3(torch.cat((dec4, self._resize(enc3, dec4)), dim=1))
        dec2 = self.dec2(torch.cat((dec3, self._resize(enc2, dec3)), dim=1))
        dec1 = self.dec1(torch.cat((dec2, self._resize(enc1, dec2)), dim=1))

        output = self.final_conv(dec1)
        output = self.channel_attention(output)
        output = self.spatial_attention(output)
        output = torch.nn.functional.interpolate(output, size=(480, 640), mode="bilinear", align_corners=False)
        return output

    def _resize(self, enc_out, target):
        return torch.nn.functional.interpolate(enc_out, size=target.shape[2:], mode="bilinear", align_corners=False)


class ModelHandler:
    def __init__(self, model_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AttentionUNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.processor = ImageProcessor()

    def predict(self, image_path, output_path=None):
        # 이미지 로드 및 전처리
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # (H, W, 3)

        # 블러 처리된 이미지 생성
        blurred_image = self.processor.process_image(image)
        if len(blurred_image.shape) == 2:
            blurred_image = np.expand_dims(blurred_image, axis=-1)
        blurred_image = np.repeat(blurred_image, 3, axis=-1)

        # 원본 이미지와 블러 이미지를 결합
        combined_image = np.concatenate((image, blurred_image), axis=-1)
        combined_image = combined_image.transpose(2, 0, 1)  # (C, H, W)
        combined_image = torch.tensor(combined_image).float().unsqueeze(0) / 255.0  # (1, C, H, W)

        # 모델 예측
        with torch.no_grad():
            prediction = self.model(combined_image.to(self.device))

        # 예측 결과를 NumPy 배열로 변환
        prediction = prediction.squeeze().cpu().numpy()

        # 예측 결과를 이미지로 저장 (선택적으로)
        if output_path:
            # 예측 결과를 0~255로 정규화 후 저장
            prediction_normalized = (prediction - prediction.min()) / (prediction.max() - prediction.min())  # 0~1 정규화
            prediction_image = (prediction_normalized * 255).astype(np.uint8)  # 0~255 변환
            cv2.imwrite(output_path, prediction_image)  # 이미지 저장

        return prediction
