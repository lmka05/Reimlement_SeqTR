# ==============================================================================
# config.py — Cấu hình toàn bộ hyperparameters cho SeqTR Detection
# ==============================================================================
# File này tập trung TẤT CẢ các tham số cấu hình vào 1 chỗ duy nhất.
# Khi muốn thay đổi bất kỳ tham số nào (batch size, learning rate, đường dẫn...),
# chỉ cần sửa ở đây, không cần tìm trong các file khác.
# ==============================================================================


class Config:
    """
    Lớp chứa toàn bộ cấu hình cho project.
    Dùng class thay vì dict để có thể truy cập bằng dấu chấm: Config.batch_size
    """

    # ==========================================================================
    # 1. ĐƯỜNG DẪN DỮ LIỆU
    # ==========================================================================
    # Khi chạy LOCAL (máy cá nhân) — sửa lại cho đúng thư mục của bạn:
    img_dir = "/kaggle/input/datasets/jeffaudi/coco-2014-dataset-for-yolov3/coco2014/images/train2014"
    ann_file = "/kaggle/input/datasets/minhkhoai/seqtr-annotations-weights/annotations/refcoco-unc/instances.json"

    # Khi chạy trên KAGGLE — uncomment 2 dòng dưới, comment 2 dòng trên:
    # img_dir = "/kaggle/input/coco-train2014/train2014"
    # ann_file = "/kaggle/input/refcoco-annotations/refcoco-unc/instances.json"

    # ==========================================================================
    # 2. TIỀN XỬ LÝ ẢNH
    # ==========================================================================

    # Kích thước ảnh sau khi resize + pad.
    # SeqTR gốc dùng 640x640. Ảnh được resize giữ tỉ lệ rồi pad về kích thước này.
    # Ví dụ: ảnh 800x600 → resize thành 640x480 → pad thêm 160px chiều cao → 640x640
    img_size = 640

    # ==========================================================================
    # 3. XỬ LÝ NGÔN NGỮ (TEXT)
    # ==========================================================================

    # Số từ tối đa trong 1 câu referring expression.
    # Câu nào dài hơn sẽ bị cắt, câu nào ngắn hơn sẽ được pad thêm số 0.
    # Ví dụ: "the man in red" = 4 từ → [34, 12, 5, 89, 0, 0, 0, ..., 0] (15 phần tử)
    # Giá trị 15 là đủ cho hầu hết các câu trong RefCOCO (trung bình ~3-4 từ).
    max_token = 15

    # Chiều (dimension) của GloVe word embedding.
    # GloVe chuyển mỗi từ thành 1 vector 300 chiều.
    # Ví dụ: "dog" → [0.23, -0.11, 0.87, ..., 0.45] (300 số thực)
    glove_dim = 300

    # ==========================================================================
    # 4. KIẾN TRÚC MODEL
    # ==========================================================================

    # --- Visual Encoder (Backbone) ---
    # Số channels đầu ra của backbone (ResNet-50 layer3 = 1024 channels).
    # Đây cũng là input channels cho fusion module.
    backbone_out_channels = 1024

    # --- Language Encoder ---
    # Hidden size của GRU. Vì dùng bidirectional, output sẽ là 2 * gru_hidden = 1024.
    # Phải đảm bảo 2 * gru_hidden == backbone_out_channels để fusion hoạt động.
    gru_hidden = 512

    # --- Transformer ---
    # d_model: Chiều của vector trong Transformer.
    # Tất cả input/output của encoder & decoder đều có chiều này.
    # Giá trị 256 là cân bằng giữa hiệu suất và tốc độ.
    d_model = 256

    # Số head trong Multi-Head Attention. d_model phải chia hết cho nhead.
    # 256 / 8 = 32 dimensions per head.
    nhead = 8

    # Chiều của Feed-Forward Network bên trong mỗi Transformer layer.
    # Thường là 4 * d_model = 1024.
    dim_feedforward = 1024

    # Số layers của Transformer Encoder.
    # Encoder xử lý visual features (ảnh đã qua fusion).
    # 6 layers giúp model học được biểu diễn visual phong phú.
    enc_layers = 6

    # Số layers của Transformer Decoder.
    # Decoder sinh ra chuỗi tọa độ [x1, y1, x2, y2] theo kiểu auto-regressive.
    # 3 layers là đủ vì output sequence ngắn (chỉ 4 tokens).
    dec_layers = 3

    # Dropout rate trong Transformer. Giúp chống overfitting.
    dropout = 0.1

    # --- Quantization ---
    # Số bins để quantize tọa độ liên tục thành số nguyên rời rạc.
    # Tọa độ float [0, W] hoặc [0, H] → integer [0, 999].
    # Ví dụ: x=320.0 trong ảnh 640px → 320/640 * 1000 = 500
    # num_bin càng lớn → tọa độ càng chính xác, nhưng vocab càng lớn.
    num_bin = 1000

    # Kích thước vocabulary = num_bin + 1 (thêm 1 token END).
    # Tokens: 0-999 = giá trị tọa độ, 1000 = END token.
    vocab_size = 1001

    # ==========================================================================
    # 5. HUẤN LUYỆN (TRAINING)
    # ==========================================================================

    # Batch size: Số ảnh xử lý cùng lúc trong 1 step.
    # - V100 32GB (paper gốc): 128
    # - Kaggle T4 16GB: 8
    # - Laptop 4-8GB: 4
    batch_size = 8

    # Learning rate (tốc độ học).
    # Paper gốc dùng lr=0.0005 với batch_size=128.
    # Theo Linear Scaling Rule: lr_new = lr_base * (batch_new / batch_base)
    # → 0.0005 * (8 / 128) = 0.00003125
    lr = 0.00003125

    # Số epoch huấn luyện.
    # 1 epoch = duyệt qua toàn bộ training set 1 lần.
    # Paper gốc dùng 60 epochs. Để test nhanh, có thể giảm xuống 5-10.
    epochs = 60

    # Warmup: Trong N epoch đầu, learning rate tăng dần từ 0 lên lr.
    # Giúp model ổn định ở giai đoạn đầu training (tránh gradient quá lớn).
    warmup_epochs = 5

    # Epoch mà learning rate sẽ giảm (nhân với decay_ratio).
    # Ví dụ: epoch 50 → lr giảm 10 lần (0.000125 → 0.0000125).
    decay_epoch = 50
    decay_ratio = 0.1

    # Gradient clipping: Giới hạn norm của gradient để tránh exploding gradients.
    # Nếu gradient norm > 0.15, nó sẽ bị scale xuống còn 0.15.
    grad_clip = 0.15

    # --- Exponential Moving Average (EMA) ---
    # EMA duy trì 1 bản sao "trung bình" của model weights.
    # Bản sao này thường cho kết quả tốt hơn model gốc khi test.
    # ema_decay = 0.999 nghĩa là: shadow = 0.999 * shadow + 0.001 * current_weights
    ema = True
    ema_decay = 0.999

    # --- Label Smoothing ---
    # Thay vì hard target [0, 0, 1, 0, ...], dùng soft target [0.0001, 0.0001, 0.9, 0.0001, ...]
    # Giúp model tránh over-confident và generalize tốt hơn.
    label_smoothing = 0.1

    # ==========================================================================
    # 6. LOGGING & CHECKPOINT
    # ==========================================================================

    # In log mỗi N batches.
    log_interval = 20

    # Random seed: Đảm bảo kết quả reproducible (chạy lại ra cùng kết quả).
    seed = 6666

    # Số workers cho DataLoader. Kaggle nên dùng 2, local có thể dùng 4.
    num_workers = 2

    # Thư mục lưu checkpoint & log.
    work_dir = "./work_dir"
