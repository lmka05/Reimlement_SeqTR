# ==============================================================================
# backbone.py — Visual Encoder (ResNet-50)
# ==============================================================================
# Trích xuất đặc trưng hình ảnh (visual features) từ ảnh đầu vào.
#
# Repo gốc dùng DarkNet-53, nhưng cần tải weights riêng + code parse phức tạp.
# Chúng ta dùng ResNet-50 từ torchvision — pretrained sẵn, không cần setup gì.
#
# ResNet-50 gồm nhiều "layer", mỗi layer giảm resolution và tăng channels:
#   Input:  [B, 3, 640, 640]
#   layer0: [B, 64, 160, 160]   (conv1 + bn + relu + maxpool)
#   layer1: [B, 256, 160, 160]  (3 bottleneck blocks)
#   layer2: [B, 512, 80, 80]   (4 bottleneck blocks)
#   layer3: [B, 1024, 40, 40]  (6 bottleneck blocks)
#   layer4: [B, 2048, 20, 20]  (3 bottleneck blocks)
#
# Chúng ta lấy output từ 3 tầng: layer2, layer3, layer4
# để feature map cuối = 20x20 (400 tokens), khớp với DarkNet gốc.
# (Trước đó dùng layer1/2/3 → 40x40 = 1600 tokens → OOM!)
# ==============================================================================

import torch
import torch.nn as nn
import torchvision.models as models


class VisualEncoder(nn.Module):
    """
    Visual Encoder dùng ResNet-50 pretrained.

    Trích xuất 3 feature maps ở các scale khác nhau:
        - C3: [B, 512, 80, 80]   — chi tiết cao
        - C4: [B, 1024, 40, 40]  — trung bình
        - C5: [B, 2048, 20, 20]  — ngữ cảnh rộng, semantic phong phú

    Feature map cuối 20x20 = 400 tokens cho Transformer (tiết kiệm memory).
    Fusion module sẽ kết hợp cả 3 scales lại.
    """

    def __init__(self, freeze_layers=True):
        """
        Args:
            freeze_layers (bool): Có đóng băng (freeze) các layer đầu không.
                True = không cập nhật weights của layer0 + layer1 khi train.
                Điều này giúp:
                1. Giảm memory GPU (ít gradient hơn)
                2. Giữ features low-level (cạnh, góc, texture) đã học tốt từ ImageNet
        """
        super().__init__()

        # Tải ResNet-50 pretrained trên ImageNet
        # weights='IMAGENET1K_V1' = dùng bộ weights đã train trên ImageNet
        resnet = models.resnet50(weights='IMAGENET1K_V1')

        # Tách ResNet thành các phần riêng biệt
        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )  # [B, 3, 640, 640] → [B, 64, 160, 160]

        self.layer1 = resnet.layer1  # → [B, 256, 160, 160]
        self.layer2 = resnet.layer2  # → [B, 512, 80, 80]   ← output C3
        self.layer3 = resnet.layer3  # → [B, 1024, 40, 40]  ← output C4
        self.layer4 = resnet.layer4  # → [B, 2048, 20, 20]  ← output C5

        # Đóng băng layer0 + layer1 (features low-level đã tốt từ ImageNet)
        if freeze_layers:
            for layer in [self.layer0, self.layer1]:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, img):
        """
        Forward pass: ảnh → 3 feature maps.

        Args:
            img (Tensor): [B, 3, 640, 640] — batch ảnh đã normalize

        Returns:
            features (list[Tensor]): 3 feature maps
                - [B, 512, 80, 80]   (C3)
                - [B, 1024, 40, 40]  (C4)
                - [B, 2048, 20, 20]  (C5 — 400 tokens cho Transformer)
        """
        x = self.layer0(img)   # → [B, 64, 160, 160]
        x = self.layer1(x)     # → [B, 256, 160, 160]
        c3 = self.layer2(x)    # → [B, 512, 80, 80]
        c4 = self.layer3(c3)   # → [B, 1024, 40, 40]
        c5 = self.layer4(c4)   # → [B, 2048, 20, 20]

        return [c3, c4, c5]


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == "__main__":
    print("=== Test VisualEncoder ===")

    model = VisualEncoder(freeze_layers=True)

    # Đếm tham số
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}")
    print(f"Trainable:    {trainable:,}")
    print(f"Frozen:       {total - trainable:,}")

    # Test forward
    dummy_img = torch.randn(2, 3, 640, 640)  # Batch of 2 images
    features = model(dummy_img)

    for i, feat in enumerate(features):
        print(f"Feature C{i+3}: {feat.shape}")
    # Expected:
    # Feature C3: torch.Size([2, 512, 80, 80])
    # Feature C4: torch.Size([2, 1024, 40, 40])
    # Feature C5: torch.Size([2, 2048, 20, 20])

    print("✅ VisualEncoder test passed!")
