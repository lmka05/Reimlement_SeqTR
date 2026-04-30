import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFusion(nn.Module):
    """
    Multi-modal Fusion: kết hợp visual + language features.

    Input:
        vis_feats: list [C3, C4, C5] từ backbone
        lang_feat: [B, 1, 1024] từ language encoder

    Output:
        x_fused: [B, 1024, 20, 20] — feature map đã kết hợp cả ảnh + câu
    """

    def __init__(self, vis_channels=[512, 1024, 2048]):
        """
        Args:
            vis_channels: Số channels của mỗi feature map từ backbone.
                ResNet-50: [512, 1024, 2048] (layer2, layer3, layer4)
        """
        super().__init__()

        c3_ch, c4_ch, c5_ch = vis_channels  # 512, 1024, 2048

        self.down_c3 = nn.Sequential(
            nn.Conv2d(in_channels = c3_ch, out_channels = c3_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3_ch),
            nn.ReLU(inplace=True),
        )


        mid_ch = c3_ch + c4_ch  # 512 + 1024 = 1536
        self.down_mid = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )


        merged_ch = mid_ch + c5_ch  # 1536 + 2048 = 3584
        out_ch = 1024
        self.project = nn.Sequential(
            nn.Conv2d(merged_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, vis_feats, lang_feat):
        """
        Args:
            vis_feats (list[Tensor]): 3 feature maps từ backbone
                - C3: [B, 512, 80, 80]
                - C4: [B, 1024, 40, 40]
                - C5: [B, 2048, 20, 20]

            lang_feat (Tensor): [B, 1, 1024] — vector ngôn ngữ

        Returns:
            x_fused (Tensor): [B, 1024, 20, 20] — visual-language fused features
        """
        c3, c4, c5 = vis_feats


        # Downsample C3: [B, 512, 80, 80] → [B, 512, 40, 40]
        c3_down = self.down_c3(c3)

        # Concat C3_down + C4
        # [B, 512, 40, 40] cat [B, 1024, 40, 40] → [B, 1536, 40, 40]
        mid = torch.cat([c3_down, c4], dim=1)

        # Downsample mid: [B, 1536, 40, 40] → [B, 1536, 20, 20]
        mid_down = self.down_mid(mid)

        # Concat mid_down + C5 → project
        # [B, 1536, 20, 20] cat [B, 2048, 20, 20] → [B, 3584, 20, 20]
        merged = torch.cat([mid_down, c5], dim=1)

        # Project: [B, 3584, 20, 20] → [B, 1024, 20, 20]
        x_vis = self.project(merged)

        # Element-wise Fusion (visual × language) 

        y = lang_feat.squeeze(1).unsqueeze(-1).unsqueeze(-1)  # [B, 1024, 1, 1]

        # tanh gating: ngôn ngữ quyết định vùng nào của ảnh được "bật/tắt"
        x_fused = torch.tanh(x_vis) * torch.tanh(y)  # [B, 1024, 20, 20]

        return x_fused


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == "__main__":
    print("=== Test SimpleFusion ===")

    model = SimpleFusion(vis_channels=[512, 1024, 2048])

    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total:,}")

    B = 2
    c3 = torch.randn(B, 512, 80, 80)
    c4 = torch.randn(B, 1024, 40, 40)
    c5 = torch.randn(B, 2048, 20, 20)
    lang = torch.randn(B, 1, 1024)

    x_fused = model([c3, c4, c5], lang)
    print(f"Input:  C3={c3.shape}, C4={c4.shape}, C5={c5.shape}, lang={lang.shape}")
    print(f"Output: {x_fused.shape}")  # Expected: [2, 1024, 20, 20]
    print(f"Range:  [{x_fused.min():.3f}, {x_fused.max():.3f}]")

    assert x_fused.max() <= 1.0 and x_fused.min() >= -1.0, "Range ngoài [-1,1]!"
    print("✅ SimpleFusion test passed!")
