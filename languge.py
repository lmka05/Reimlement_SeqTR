# language.py — Language Encoder (BiGRU + GloVe)
# Pipeline:
#   Câu text → tokenize → indices → GloVe embedding → BiGRU → max-pool → y
#
# Ví dụ:
#   "the man in red"
#   → tokenize: [34, 12, 5, 89, 0, 0, ..., 0]  (15 số, pad bằng 0)
#   → GloVe:    [B, 15, 300]                     (mỗi từ → vector 300d)
#   → BiGRU:    [B, 15, 1024]                    (hidden=512, bidirectional → 512*2=1024)
#   → max-pool: [B, 1, 1024]                     (lấy max qua chiều thời gian)
#
# Output y có shape [B, 1, 1024] — cùng chiều với backbone output (1024 channels)
# để 2 bên có thể nhân element-wise trong Fusion module.
# ==============================================================================

import torch
import torch.nn as nn


class LanguageEncoder(nn.Module):
    """
    Language Encoder: GloVe Embedding + Bidirectional GRU + Max Pooling.

    Input:  ref_inds [B, max_token] — indices của các từ trong câu
    Output: y [B, 1, 1024] — vector đại diện cho cả câu
    """

    def __init__(self, glove_vectors, hidden_size=512):
        """
        Args:
            glove_vectors (Tensor): Ma trận GloVe [vocab_size, 300].
                Mỗi dòng là vector embedding 300d của 1 từ.
                Được tạo bởi hàm build_glove_matrix() trong dataset.py.

            hidden_size (int): Kích thước hidden state của GRU.
                Vì dùng bidirectional, output = 2 * hidden_size = 1024.
                Giá trị 512 được chọn để 2*512 = 1024 = backbone_out_channels.
        """
        super().__init__()

        self.hidden_size = hidden_size
        vocab_size, embed_dim = glove_vectors.shape  # [~5000, 300]

        # --- Embedding Layer ---
        # Chuyển index (số nguyên) → vector GloVe 300d.
        # nn.Embedding là bảng tra (lookup table): input index → output vector.
        #
        # from_pretrained: Khởi tạo bảng bằng ma trận GloVe đã tải.
        # freeze=True: KHÔNG cập nhật weights embedding khi train.
        #   Lý do: GloVe đã được train trên corpus khổng lồ (Wikipedia + Gigaword),
        #   embedding đã rất tốt rồi, fine-tune thêm dễ gây overfitting
        #   (vì vocabulary nhỏ ~5000 từ).
        #
        # padding_idx=0: Token PAD (index 0) luôn có embedding = vector 0.
        #   Điều này đảm bảo padding tokens không ảnh hưởng đến kết quả.
        self.embedding = nn.Embedding.from_pretrained(
            glove_vectors,     # [vocab_size, 300]
            freeze=True,       # Không train embedding
            padding_idx=0      # PAD token index
        )

        # --- Bidirectional GRU ---
        # Input:  [B, seq_len, 300]     (embedding dim)
        # Output: [B, seq_len, 1024]    (hidden_size * 2 vì bidirectional)
        self.gru = nn.GRU(
            input_size=embed_dim,        # 300 (GloVe dimension)
            hidden_size=hidden_size,     # 512
            num_layers=1,                # 1 tầng GRU (đơn giản, đủ dùng)
            bidirectional=True,          # Chạy cả 2 chiều
            batch_first=True,            # Input/output dạng [B, seq_len, ...]
            bias=True,                   # Thêm bias term
            dropout=0.0,                 # Không dropout (chỉ 1 layer)
        )

    def forward(self, ref_inds):
        """
        Forward pass: indices → language feature vector.

        Args:
            ref_inds (Tensor): [B, max_token], dtype=torch.long
                Indices của các từ. PAD positions có giá trị = 0.
                Ví dụ: [34, 12, 5, 89, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Returns:
            y (Tensor): [B, 1, 1024] — vector đại diện ngôn ngữ cho cả câu
        """
        # Tạo mask — đánh dấu vị trí nào là từ thật (True), vị trí nào là PAD (False)
        # ref_inds = [34, 12, 5, 89, 0, 0, ...] → mask = [True, True, True, True, False, False, ...]
        mask = (ref_inds != 0)  # [B, max_token]

        # chuyển index → vector 300d
        emb = self.embedding(ref_inds)  # [B, max_token, 300]

        # Đưa qua BiGRU 
        # output: [B, max_token, 1024] (1024 = 512 forward + 512 backward)
        output, _ = self.gru(emb)  # _ = hidden state cuối (không dùng)

        # Bước 4: Max-pooling qua chiều thời gian (chỉ trên các từ thật, bỏ PAD)
        #
        # Tại sao max-pool mà không dùng hidden state cuối?
        #   Hidden state cuối chỉ "nhìn" tốt phần cuối câu.
        #   Max-pool lấy giá trị lớn nhất ở MỖI chiều qua TẤT CẢ các từ,
        #   nên bắt được từ quan trọng nhất bất kể vị trí nào trong câu.
        #
        # Trick: Gán giá trị -inf cho PAD positions trước khi max,
        #        để PAD tokens không bao giờ "thắng" trong max.
        #
        # mask shape: [B, max_token] → unsqueeze → [B, max_token, 1]
        # Broadcast sang [B, max_token, 1024] khi dùng masked_fill_
        output = output.masked_fill(~mask.unsqueeze(-1), float('-inf'))

        # Max qua dim=1 (chiều thời gian) → [B, 1024]
        # keepdim=True → giữ dimension → [B, 1, 1024]
        y = output.max(dim=1, keepdim=True).values  # [B, 1, 1024]

        return y


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == "__main__":
    print("=== Test LanguageEncoder ===")

    # Tạo GloVe giả (100 từ, 300 chiều)
    fake_glove = torch.randn(100, 300)
    fake_glove[0] = 0  # PAD token = zero vector

    model = LanguageEncoder(fake_glove, hidden_size=512)

    # Đếm tham số
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}")
    print(f"Trainable:    {trainable:,} (GloVe is frozen)")

    # Test forward
    # Giả lập batch 2 câu, mỗi câu 15 từ (có padding)
    ref_inds = torch.tensor([
        [5, 10, 20, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4 từ thật + 11 PAD
        [8, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],    # 2 từ thật + 13 PAD
    ])

    y = model(ref_inds)
    print(f"Input shape:  {ref_inds.shape}")  # [2, 15]
    print(f"Output shape: {y.shape}")          # [2, 1, 1024]
    print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")

    # Kiểm tra output không chứa -inf (PAD đã bị loại đúng cách)
    assert not torch.isinf(y).any(), "Output chứa inf — lỗi masking!"
    print("✅ LanguageEncoder test passed!")
