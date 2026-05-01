# Hướng Dẫn Tự Code Lại `models/transformer.py`

Mục tiêu của file này là giúp bạn tự code lại `transformer.py` theo từng lớp, hiểu rõ module nào cần tạo trước, đầu vào là gì, đầu ra là gì, và module đó phục vụ phần nào trong SeqTR.

Trong project này, `transformer.py` không phải Transformer dịch máy thông thường. Nó là phần `SeqHead` nhận feature ảnh đã fusion với text, rồi sinh bounding box như một chuỗi token:

```text
fused feature [B, 1024, 40, 40]
    -> Transformer encoder/decoder
    -> token [x1, y1, x2, y2]
    -> bbox [B, 4]
```

## 0. Bức Tranh Tổng Thể

Trước khi code, hãy nhớ flow từ `models/model.py`:

```text
img + ref_inds
    -> VisualEncoder
    -> LanguageEncoder
    -> SimpleFusion
    -> x_fused [B, 1024, H, W]
    -> SeqHead trong transformer.py
```

`transformer.py` chỉ phụ trách đoạn cuối:

```text
x_fused + img_metas + gt_bbox(optional)
    -> nếu train: loss
    -> nếu test: pred_bbox [B, 4]
```

Các thành phần cần code theo thứ tự:

1. `SinePositionalEncoding2D`
2. `quantize_bbox`
3. `dequantize_bbox`
4. `SeqHead.__init__`
5. `SeqHead._reset_parameters`
6. `SeqHead._generate_causal_mask`
7. `SeqHead._encode`
8. `SeqHead.forward_train`
9. `SeqHead.forward_test`
10. block test dưới `if __name__ == "__main__"`

## 1. Import Cần Có

Tạo file với các import tối thiểu:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
```

Vì sao cần:

- `math`: dùng `math.pi` cho positional encoding.
- `torch`: tạo tensor, mask, token.
- `torch.nn`: tạo module, embedding, Transformer, loss.
- `torch.nn.functional`: resize mask bằng `F.interpolate`.

## 2. Module `SinePositionalEncoding2D`

### Vai trò

Transformer không tự biết pixel nào ở vị trí nào. Sau khi feature map bị flatten từ `[B, C, H, W]` thành `[B, H*W, C]`, vị trí không gian bị mất nếu ta không cộng positional encoding.

Module này tạo encoding sin/cos 2D cho ảnh.

### Input

```python
mask: Tensor bool [B, H, W]
```

Ý nghĩa:

- `False`: vùng ảnh thật.
- `True`: vùng padding.

### Output

```python
pos: Tensor float [B, d_model, H, W]
```

### Cần code gì trong `__init__`

Lưu các biến:

```python
self.num_feature = num_feature
self.temperature = temperature
self.normalize = normalize
self.scale = 2 * math.pi
```

Trong project này:

```python
num_feature = d_model // 2
```

Vì positional encoding sẽ encode cả trục `y` và trục `x`, mỗi trục dùng `d_model / 2`.

### Cần code gì trong `forward`

Các bước:

1. Đảo mask:

```python
not_mask = ~mask
```

2. Tạo tọa độ tích lũy:

```python
y_embed = not_mask.cumsum(1, dtype=torch.float32)
x_embed = not_mask.cumsum(2, dtype=torch.float32)
```

3. Normalize tọa độ về `[0, 2pi]`.

4. Tạo vector tần số `dim_t`.

5. Tính sin/cos cho `x` và `y`.

6. Ghép lại thành `[B, d_model, H, W]`.

### Tự kiểm tra

Với:

```python
mask = torch.zeros(2, 40, 40, dtype=torch.bool)
pos_enc = SinePositionalEncoding2D(128)
pos = pos_enc(mask)
```

Kỳ vọng:

```text
pos.shape == [2, 256, 40, 40]
```

## 3. Hàm `quantize_bbox`

### Vai trò

SeqTR biến bài toán dự đoán bbox thành bài toán sinh token.

Thay vì dự đoán trực tiếp:

```text
[x1, y1, x2, y2] = [160.0, 80.0, 480.0, 400.0]
```

nó đổi sang token:

```text
[250, 125, 750, 625]
```

nếu ảnh pad là `640x640` và `num_bin = 1000`.

### Input

```python
bbox: Tensor [B, 4]
img_meta: list[dict]
num_bin: int
```

`bbox` có dạng:

```text
[x1, y1, x2, y2]
```

`img_meta[i]['pad_shape']` có dạng:

```text
(H, W, 3)
```

### Output

```python
tokens: LongTensor [B, 4]
```

Giá trị token nằm trong:

```text
0 -> num_bin - 1
```

### Công thức

Tạo scale:

```python
scale = [W, H, W, H]
```

Rồi:

```python
tokens = (bbox / scale * num_bin).long()
tokens = tokens.clamp(0, num_bin - 1)
```

### Tự kiểm tra

```python
bbox = torch.tensor([[160., 80., 480., 400.]])
img_meta = [{'pad_shape': (640, 640, 3)}]
tokens = quantize_bbox(bbox, img_meta, 1000)
```

Kỳ vọng gần đúng:

```text
[[250, 125, 750, 625]]
```

## 4. Hàm `dequantize_bbox`

### Vai trò

Hàm ngược lại với `quantize_bbox`. Khi inference, decoder sinh token, ta cần đổi token về tọa độ bbox thật.

### Input

```python
tokens: LongTensor [B, 4]
img_meta: list[dict]
num_bin: int
```

### Output

```python
bbox: FloatTensor [B, 4]
```

### Công thức

```python
bbox = tokens.float() / num_bin * scale
```

với:

```python
scale = [W, H, W, H]
```

### Tự kiểm tra

```python
tokens = torch.tensor([[250, 125, 750, 625]])
img_meta = [{'pad_shape': (640, 640, 3)}]
bbox = dequantize_bbox(tokens, img_meta, 1000)
```

Kỳ vọng:

```text
[[160., 80., 480., 400.]]
```

## 5. Class `SeqHead`

Đây là module chính của file.

### Vai trò

Nhận feature đã fusion:

```python
x_fused: Tensor [B, 1024, H, W]
```

Rồi:

- Khi train: dùng `gt_bbox` để tính loss.
- Khi inference: tự sinh bbox token từng bước.

### Input từ `models/model.py`

Khi train:

```python
self.head.forward_train(x_fused, gt_bbox, img_metas)
```

Khi test:

```python
self.head.forward_test(x_fused, img_metas)
```

## 6. `SeqHead.__init__`

### Các tham số chính

```python
in_ch = 1024
d_model = 256
nhead = 8
dim_feedforward = 1024
enc_layers = 6
dec_layers = 3
num_bin = 1000
label_smoothing = 0.1
```

### Biến trạng thái cần lưu

```python
self.d_model = d_model
self.num_bin = num_bin
self.vocab_size = num_bin + 1
self.seq_len = 4
```

Giải thích:

- `num_bin = 1000`: token `0..999` là tọa độ.
- `vocab_size = 1001`: thêm token `1000` làm `END`.
- `seq_len = 4`: bbox có 4 số `[x1, y1, x2, y2]`.

### 6.1. Input Projection

Input từ fusion là:

```text
[B, 1024, H, W]
```

Transformer dùng `d_model = 256`, nên cần conv 1x1:

```python
self.input_proj = nn.Sequential(
    nn.Conv2d(in_ch, d_model, kernel_size=1, bias=True),
    nn.GroupNorm(32, d_model),
)
```

Output:

```text
[B, 256, H, W]
```

### 6.2. Positional Encoding

2D positional encoding cho visual feature:

```python
self.pos_enc_2d = SinePositionalEncoding2D(
    num_feature=d_model // 2,
    normalize=True
)
```

1D positional encoding cho chuỗi token:

```python
self.pos_enc_1d = nn.Embedding(
    num_embeddings=self.seq_len + 1,
    embedding_dim=d_model,
)
```

Vì training decoder input có 5 vị trí:

```text
[START, x1, y1, x2, y2]
```

### 6.3. Token Embedding

```python
self.token_embedding = nn.Embedding(
    num_embeddings=self.vocab_size,
    embedding_dim=d_model,
)
```

Dùng để biến token tọa độ thành vector trước khi đưa vào decoder.

### 6.4. Transformer Encoder

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=d_model,
    nhead=nhead,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    activation='relu',
    batch_first=True,
)
self.encoder = nn.TransformerEncoder(
    encoder_layer,
    num_layers=enc_layers
)
```

Input encoder:

```text
[B, H*W, 256]
```

Output encoder:

```text
memory [B, H*W, 256]
```

### 6.5. Transformer Decoder

```python
decoder_layer = nn.TransformerDecoderLayer(
    d_model=d_model,
    nhead=nhead,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    activation='relu',
    batch_first=True,
)
self.decoder = nn.TransformerDecoder(
    decoder_layer,
    num_layers=dec_layers
)
```

Decoder nhận:

```text
tgt:    token sequence [B, L, 256]
memory: visual memory  [B, H*W, 256]
```

### 6.6. Predictor

```python
self.predictor = nn.Sequential(
    nn.Linear(d_model, d_model),
    nn.ReLU(inplace=True),
    nn.Linear(d_model, d_model),
    nn.ReLU(inplace=True),
    nn.Linear(d_model, self.vocab_size),
)
```

Output:

```text
logits [B, L, 1001]
```

### 6.7. Loss

```python
self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
```

Cross entropy dùng vì đây là bài toán classification token.

## 7. `_reset_parameters`

### Vai trò

Khởi tạo weight bằng Xavier Uniform.

### Logic

```python
for p in self.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
```

Bạn gọi hàm này cuối `__init__`.

## 8. `_generate_causal_mask`

### Vai trò

Đảm bảo decoder sinh token tuần tự. Token hiện tại không được nhìn token tương lai.

### Input

```python
seq_len: int
device: torch.device
```

### Output

```python
mask: Tensor [seq_len, seq_len]
```

Ví dụ `seq_len = 5`:

```text
0     -inf  -inf  -inf  -inf
0      0    -inf  -inf  -inf
0      0     0    -inf  -inf
0      0     0     0    -inf
0      0     0     0     0
```

### Code lõi

```python
mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
mask = mask.masked_fill(mask == 1, float('-inf'))
```

## 9. `_encode`

### Vai trò

Biến feature map ảnh thành `memory` cho decoder attend vào.

### Input

```python
x_fused: Tensor [B, 1024, H, W]
img_metas: list[dict]
```

### Output

```python
memory: Tensor [B, H*W, d_model]
x_mask: Tensor bool [B, H*W]
x_pos: Tensor [B, H*W, d_model]
```

### Các bước

1. Project channel:

```python
x = self.input_proj(x_fused)
```

Shape:

```text
[B, 1024, H, W] -> [B, 256, H, W]
```

2. Tạo mask ảnh thật/padding.

Ban đầu tạo mask size ảnh pad:

```python
x_mask = x_fused.new_ones((B, input_h, input_w))
```

Rồi với mỗi ảnh:

```python
x_mask[i, :img_h, :img_w] = 0
```

3. Resize mask về size feature map:

```python
x_mask = F.interpolate(x_mask.unsqueeze(1), size=(H, W)).squeeze(1)
x_mask = x_mask.bool()
```

4. Tạo positional encoding:

```python
x_pos = self.pos_enc_2d(x_mask)
```

5. Flatten:

```python
x = x.flatten(2).transpose(1, 2)
x_pos = x_pos.flatten(2).transpose(1, 2)
x_mask = x_mask.flatten(1)
```

6. Cộng positional encoding:

```python
x_with_pos = x + x_pos
```

7. Chạy Transformer encoder:

```python
memory = self.encoder(
    x_with_pos,
    src_key_padding_mask=x_mask
)
```

### Tự kiểm tra

Với:

```python
B = 2
x_fused = torch.randn(B, 1024, 40, 40)
img_metas = [
    {'pad_shape': (640, 640, 3), 'img_shape': (480, 640, 3)},
    {'pad_shape': (640, 640, 3), 'img_shape': (640, 480, 3)},
]
```

Kỳ vọng:

```text
memory.shape == [2, 1600, 256]
x_mask.shape == [2, 1600]
x_pos.shape == [2, 1600, 256]
```

## 10. `forward_train`

### Vai trò

Training dùng teacher forcing:

```text
decoder input: [START, gt_x1, gt_y1, gt_x2, gt_y2]
target:        [gt_x1, gt_y1, gt_x2, gt_y2, END]
```

Model học predict token tiếp theo tại mỗi vị trí.

### Input

```python
x_fused: Tensor [B, 1024, H, W]
gt_bbox: Tensor [B, 4]
img_metas: list[dict]
```

### Output

```python
loss: scalar Tensor
```

### Các bước

1. Encode visual:

```python
memory, x_mask, x_pos = self._encode(x_fused, img_metas)
```

2. Quantize bbox:

```python
gt_tokens = quantize_bbox(gt_bbox, img_metas, self.num_bin)
```

Shape:

```text
[B, 4]
```

3. Tạo target có END:

```python
end_token = torch.full((B, 1), self.num_bin, dtype=torch.long, device=device)
targets = torch.cat([gt_tokens, end_token], dim=1)
```

Shape:

```text
[B, 5]
```

4. Tạo decoder input:

```python
start_embed = torch.zeros(B, 1, self.d_model, device=device)
gt_embeds = self.token_embedding(gt_tokens)
seq_input = torch.cat([start_embed, gt_embeds], dim=1)
```

Shape:

```text
[B, 5, 256]
```

5. Cộng 1D positional encoding:

```python
seq_pos = self.pos_enc_1d(torch.arange(self.seq_len + 1, device=device))
seq_pos = seq_pos.unsqueeze(0).expand(B, -1, -1)
seq_input = seq_input + seq_pos
```

6. Tạo causal mask:

```python
causal_mask = self._generate_causal_mask(self.seq_len + 1, device)
```

7. Decode:

```python
decoder_out = self.decoder(
    seq_input,
    memory,
    tgt_mask=causal_mask,
    memory_key_padding_mask=x_mask,
)
```

8. Predict logits:

```python
logits = self.predictor(decoder_out)
```

Shape:

```text
[B, 5, 1001]
```

9. Tính loss:

```python
loss = self.loss_fn(
    logits.reshape(-1, self.vocab_size),
    targets.reshape(-1)
)
```

## 11. `forward_test`

### Vai trò

Inference không có `gt_bbox`, nên model phải tự sinh từng token.

### Input

```python
x_fused: Tensor [B, 1024, H, W]
img_metas: list[dict]
```

### Output

```python
pred_bbox: Tensor [B, 4]
```

### Flow sinh token

```text
step 0: input [START]              -> predict x1
step 1: input [START, x1]          -> predict y1
step 2: input [START, x1, y1]      -> predict x2
step 3: input [START, x1, y1, x2]  -> predict y2
```

### Các bước

1. Encode visual:

```python
memory, x_mask, x_pos = self._encode(x_fused, img_metas)
```

2. Khởi tạo sequence:

```python
start_embed = torch.zeros(B, 1, self.d_model, device=device)
seq_input = start_embed
output_tokens = []
```

3. Lặp 4 bước:

```python
for step in range(self.seq_len):
    cur_len = seq_input.shape[1]
    seq_pos = self.pos_enc_1d(torch.arange(cur_len, device=device))
    seq_pos = seq_pos.unsqueeze(0).expand(B, -1, -1)
    seq_with_pos = seq_input + seq_pos

    causal_mask = self._generate_causal_mask(cur_len, device)

    decoder_out = self.decoder(
        seq_with_pos,
        memory,
        tgt_mask=causal_mask,
        memory_key_padding_mask=x_mask,
    )

    logits = self.predictor(decoder_out[:, -1, :])
    next_token = logits.argmax(dim=-1)
    output_tokens.append(next_token)

    next_embed = self.token_embedding(next_token).unsqueeze(1)
    seq_input = torch.cat([seq_input, next_embed], dim=1)
```

4. Stack token và dequantize:

```python
pred_tokens = torch.stack(output_tokens, dim=1)
pred_bbox = dequantize_bbox(pred_tokens, img_metas, self.num_bin)
```

## 12. Test Tối Thiểu Cho File

Sau khi code xong, thêm block:

```python
if __name__ == "__main__":
    head = SeqHead()

    B = 2
    x_fused = torch.randn(B, 1024, 40, 40)
    gt_bbox = torch.tensor([
        [100.0, 50.0, 400.0, 300.0],
        [200.0, 100.0, 500.0, 400.0],
    ])
    img_metas = [
        {'pad_shape': (640, 640, 3), 'img_shape': (480, 640, 3)},
        {'pad_shape': (640, 640, 3), 'img_shape': (640, 480, 3)},
    ]

    loss = head.forward_train(x_fused, gt_bbox, img_metas)
    print(loss.item())

    pred_bbox = head.forward_test(x_fused, img_metas)
    print(pred_bbox.shape)
```

Kỳ vọng:

```text
loss là một scalar
pred_bbox.shape == torch.Size([2, 4])
```

## 13. Những Lỗi Dễ Gặp

### Sai shape khi đưa vào Transformer

`nn.TransformerEncoderLayer(batch_first=True)` cần input:

```text
[B, seq_len, d_model]
```

Vì vậy feature map phải flatten như sau:

```python
x = x.flatten(2).transpose(1, 2)
```

Không phải:

```python
x = x.flatten(2)
```

### Quên causal mask

Nếu không có causal mask, decoder có thể nhìn token tương lai trong training. Như vậy bài toán trở nên quá dễ và không giống inference.

### Nhầm `num_bin` và `vocab_size`

```python
num_bin = 1000
vocab_size = 1001
END token = 1000
tọa độ hợp lệ = 0..999
```

### Nhầm thứ tự scale bbox

Bbox là:

```text
[x1, y1, x2, y2]
```

Scale phải là:

```text
[W, H, W, H]
```

Không phải:

```text
[H, W, H, W]
```

### Quên `memory_key_padding_mask`

Nếu không truyền `x_mask`, Transformer có thể attend vào vùng padding đen của ảnh.

## 14. Lộ Trình Học Theo Buổi

### Buổi 1: Quantization

Chỉ code:

- `quantize_bbox`
- `dequantize_bbox`

Mục tiêu: hiểu vì sao bbox biến thành token.

### Buổi 2: Positional Encoding

Code:

- `SinePositionalEncoding2D`
- test shape `[B, 256, H, W]`

Mục tiêu: hiểu vì sao Transformer cần vị trí.

### Buổi 3: Encoder

Code:

- `SeqHead.__init__` phần projection, pos encoding, encoder
- `_encode`

Mục tiêu: biến image feature thành memory.

### Buổi 4: Training Decoder

Code:

- token embedding
- decoder
- predictor
- `forward_train`

Mục tiêu: hiểu teacher forcing.

### Buổi 5: Inference Decoder

Code:

- `forward_test`

Mục tiêu: hiểu auto-regressive generation.

## 15. Checklist Hoàn Thành

Bạn có thể coi `transformer.py` đã ổn khi:

- `quantize_bbox` và `dequantize_bbox` gần như ngược nhau.
- `SinePositionalEncoding2D` trả về `[B, d_model, H, W]`.
- `_encode` trả về `memory [B, H*W, d_model]`.
- `forward_train` trả về loss scalar.
- `forward_test` trả về bbox `[B, 4]`.
- `models/model.py` import được `SeqHead` không lỗi.
- `python models/transformer.py` chạy được test tối thiểu.

