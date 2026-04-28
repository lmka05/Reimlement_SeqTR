import torch
import torch.nn as nn
import torchvision.models as models

class VisualEncoder(nn.Module):
    """"
    Visual Encoder dùng ResNet-50 pretrained
    """

    def __init__(self, freeze_layers = True):
        super().__init__()

        # Tải ResNet-50 pretrained
        resnet = models.resnet50(weights = 'IMAGENET1K_V1')
        
        # Trong torchvision tách layer0 thành 4 module riêng biêt nên ta sẽ gộp nó thành 1 layer0
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        
        )

        # Các Residual blocks
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        # Không dùng layer4 (2048 channels, 20x20) — resolution quá thấp
        # và cũng không dùng avgpool + fc (phần classification gốc)

        # Đóng băng layer0 và 1
        if freeze_layers:
            for layer in [self.layer0, self.layer1]:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self,img):
        """
        Forward pass: ảnh → 3 feature maps.

        Args:
            img (Tensor): [B, 3, 640, 640] — batch ảnh đã normalize

        Returns:
            features (list[Tensor]): 3 feature maps
                - [B, 256, 160, 160]  (C2 — high resolution)
                - [B, 512, 80, 80]   (C3 — medium resolution)
                - [B, 1024, 40, 40]  (C4 — low resolution, rich semantics)
        """
        x = self.layer0(img)   # [B, 3, 640, 640] → [B, 64, 160, 160]
        c2 = self.layer1(x)    # [B, 64, 160, 160] → [B, 256, 160, 160]
        c3 = self.layer2(c2)   # [B, 256, 160, 160] → [B, 512, 80, 80]
        c4 = self.layer3(c3)   # [B, 512, 80, 80] → [B, 1024, 40, 40]

        return [c2, c3, c4]
    
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
        print(f"Feature C{i+2}: {feat.shape}")
    # Expected:
    # Feature C2: torch.Size([2, 256, 160, 160])
    # Feature C3: torch.Size([2, 512, 80, 80])
    # Feature C4: torch.Size([2, 1024, 40, 40])

    print("✅ VisualEncoder test passed!")