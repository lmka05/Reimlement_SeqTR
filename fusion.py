# ==============================================================================
# fusion.py — Multi-modal Fusion
# ==============================================================================
# Quy trình 2 bước:
#
# BƯỚC 1: Bottom-up FPN (Feature Pyramid Network)
#   Merge 3 feature maps (C2, C3, C4) thành 1 feature map duy nhất.
#   Hướng bottom-up: C2 (nhỏ nhất, chi tiết nhất) → C3 → C4 (lớn nhất)
#
#   C2 [256, 160, 160] ──downsample──→ [256, 80, 80]
#                                          ↓ concat
#   C3 [512, 80, 80]  ─────────────→ [768, 80, 80] ──downsample──→ [768, 40, 40]
#                                                                       ↓ concat
#   C4 [1024, 40, 40] ───────────────────────────────────────→ [1792, 40, 40]
#                                                                       ↓ project
#                                                                   [1024, 40, 40]
#
# BƯỚC 2: Element-wise Fusion
#   visual_feature * language_feature (nhân element-wise qua tanh)
#   → feature map chứa thông tin cả ảnh + câu mô tả
#
#   x_vis: [B, 1024, 40, 40]  (visual)
#   y:     [B, 1, 1024]       (language) → broadcast → [B, 1024, 40, 40]
#   output = tanh(x_vis) * tanh(y)    → [B, 1024, 40, 40]
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFusion(nn.Module):
    """
    Multi-modal Fusion: kết hợp visual + language features.

    Input:
        vis_feats: list [C2, C3, C4] từ backbone
        lang_feat: [B, 1, 1024] từ language encoder

    Output:
        x_fused: [B, 1024, 40, 40] — feature map đã kết hợp cả ảnh + câu
    """

    def __init__(self, vis_channels=[256, 512, 1024]):
        """
        Args:
            vis_channels: Số channels của mỗi feature map từ backbone.
                ResNet-50: [256, 512, 1024] (layer1, layer2, layer3)
        """
        super().__init__()

        c2_ch, c3_ch, c4_ch = vis_channels  # 256, 512, 1024

        # --- Bước 1a: Downsample C2 → cùng resolution với C3 ---
        # C2: [256, 160, 160] → [256, 80, 80]
        # Conv 3x3 stride 2 = giảm resolution 2 lần + học features mới
        self.down_c2 = nn.Sequential(
            nn.Conv2d(in_channels = c2_ch, out_channels = c2_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2_ch),
            nn.ReLU(inplace=True),
        )

        # --- Bước 1b: Merge C2_down + C3 → downsample → cùng resolution với C4 ---
        # Concat: [256, 80, 80] + [512, 80, 80] = [768, 80, 80]
        # Downsample: [768, 80, 80] → [768, 40, 40]
        mid_ch = c2_ch + c3_ch  # 256 + 512 = 768
        self.down_mid = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )

        # --- Bước 1c: Merge mid_down + C4 → project về 1024 channels ---
        # Concat: [768, 40, 40] + [1024, 40, 40] = [1792, 40, 40]
        # Project: [1792, 40, 40] → [1024, 40, 40]
        merged_ch = mid_ch + c4_ch  # 768 + 1024 = 1792
        self.project = nn.Sequential(
            # Conv 3x3: trộn thông tin spatial từ cả 3 scales
            nn.Conv2d(merged_ch, c4_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c4_ch),
            nn.ReLU(inplace=True),
            # Conv 1x1: nén channels xuống 1024
            nn.Conv2d(c4_ch, c4_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(c4_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, vis_feats, lang_feat):
        """
        Args:
            vis_feats (list[Tensor]): 3 feature maps từ backbone
                - C2: [B, 256, 160, 160]
                - C3: [B, 512, 80, 80]
                - C4: [B, 1024, 40, 40]

            lang_feat (Tensor): [B, 1, 1024] — vector ngôn ngữ

        Returns:
            x_fused (Tensor): [B, 1024, 40, 40] — visual-language fused features
        """
        c2, c3, c4 = vis_feats

        # ---- BƯỚC 1: Bottom-up FPN — merge 3 scales thành 1 ----

        # Downsample C2: [B, 256, 160, 160] → [B, 256, 80, 80]
        c2_down = self.down_c2(c2)

        # Concat C2_down + C3 theo chiều channels (dim=1)
        # [B, 256, 80, 80] cat [B, 512, 80, 80] → [B, 768, 80, 80]
        mid = torch.cat([c2_down, c3], dim=1)

        # Downsample mid: [B, 768, 80, 80] → [B, 768, 40, 40]
        mid_down = self.down_mid(mid)

        # Concat mid_down + C4 → project
        # [B, 768, 40, 40] cat [B, 1024, 40, 40] → [B, 1792, 40, 40]
        merged = torch.cat([mid_down, c4], dim=1)

        # Project: [B, 1792, 40, 40] → [B, 1024, 40, 40]
        x_vis = self.project(merged)

        # ---- BƯỚC 2: Element-wise Fusion (visual × language) ----

        # Reshape language vector để broadcast được với visual feature map
        # lang_feat: [B, 1, 1024] → [B, 1024] → [B, 1024, 1, 1]
        # Khi nhân với x_vis [B, 1024, 40, 40], PyTorch tự broadcast
        # [B, 1024, 1, 1] * [B, 1024, 40, 40] → [B, 1024, 40, 40]
        y = lang_feat.squeeze(1).unsqueeze(-1).unsqueeze(-1)  # [B, 1024, 1, 1]

        # Element-wise multiply qua tanh
        # tanh đưa giá trị về [-1, 1], giúp ổn định phép nhân
        # và tạo cơ chế "gating": ngôn ngữ quyết định vùng nào của ảnh được "bật/tắt"
        #
        # Ví dụ: với câu "the red car", language vector sẽ có giá trị cao ở
        # các chiều liên quan đến "red" và "car" → khi nhân với visual features,
        # các vùng trong ảnh có xe đỏ sẽ được khuếch đại, vùng khác bị giảm.
        x_fused = torch.tanh(x_vis) * torch.tanh(y)  # [B, 1024, 40, 40]

        return x_fused


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == "__main__":
    print("=== Test SimpleFusion ===")

    model = SimpleFusion(vis_channels=[256, 512, 1024])

    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total:,}")

    # Giả lập input
    B = 2
    c2 = torch.randn(B, 256, 160, 160)   # Feature scale 1
    c3 = torch.randn(B, 512, 80, 80)     # Feature scale 2
    c4 = torch.randn(B, 1024, 40, 40)    # Feature scale 3
    lang = torch.randn(B, 1, 1024)        # Language vector

    x_fused = model([c2, c3, c4], lang)
    print(f"Input:  C2={c2.shape}, C3={c3.shape}, C4={c4.shape}, lang={lang.shape}")
    print(f"Output: {x_fused.shape}")  # Expected: [2, 1024, 40, 40]
    print(f"Range:  [{x_fused.min():.3f}, {x_fused.max():.3f}]")

    # Output phải nằm trong [-1, 1] vì tanh * tanh
    assert x_fused.max() <= 1.0 and x_fused.min() >= -1.0, "Range ngoài [-1,1]!"
    print("✅ SimpleFusion test passed!")
