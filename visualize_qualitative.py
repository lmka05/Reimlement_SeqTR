# ==============================================================================
# visualize_qualitative.py — Vẽ Qualitative Results
# ==============================================================================
# Script này tạo hình minh họa cho phần Qualitative Results 
#
# Chức năng chính:
#   1. Load model SeqTR từ checkpoint (best.pth)
#   2. Chọn ngẫu nhiên N ảnh từ 1 split (testA / testB / val)
#   3. Với mỗi ảnh:
#       - Chạy inference để lấy predicted bounding box
#       - Vẽ GT bbox (ĐỎ) và Pred bbox (XANH DƯƠNG) lên cùng 1 ảnh
#       - Hiển thị câu referring expression làm tiêu đề
#       - Hiển thị giá trị IoU
#   4. Ghép tất cả thành 1 figure grid và lưu file PNG
#
# Cách chạy:
#   cd seqtr_reimpl
#   python visualize_qualitative.py --checkpoint work_dir/best.pth
#
# Tùy chọn đầy đủ:
#   python visualize_qualitative.py \
#       --checkpoint work_dir/best.pth \
#       --split testA \
#       --num-samples 4 \
#       --seed 42 \
#       --output qualitative_results.png
#
# Style (giống paper TransVG):
#   🔵 Xanh dương (Blue)  = Predicted bounding box
#   🔴 Đỏ (Red)           = Ground truth bounding box
# ==============================================================================

import os
import sys
import argparse
import random
import numpy as np
from PIL import Image

import torch
import matplotlib
# Dùng backend 'Agg' để không cần hiển thị GUI (chạy trên server/Kaggle được)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import các module từ project SeqTR Reimplement
# config.py nằm ở root → import trực tiếp
from config import Config

# models/ package → chứa SeqTRDet (model tổng hợp)
from models import SeqTRDet

# datasets/ package → chứa RefCOCODataset và các hàm xử lý ảnh
from datasets import RefCOCODataset
from datasets.dataset import resize_image_keep_ratio

# utils/ package → chứa build_vocab, build_glove_matrix
from utils import build_vocab, build_glove_matrix

# evaluate.py nằm ở root → import trực tiếp
from evaluate import compute_iou_batch


# ==============================================================================
# PHẦN 1: HÀM VẼ BOUNDING BOX
# ==============================================================================

def draw_bbox_on_axes(ax, bbox, color, linewidth=3, label=None):
    """
    Vẽ 1 bounding box lên matplotlib Axes.

    Sử dụng matplotlib.patches.Rectangle để vẽ hình chữ nhật.
    Rectangle nhận góc trên-trái (x1, y1) và kích thước (width, height).

    Args:
        ax (matplotlib.axes.Axes): Axes để vẽ lên (1 ô trong grid figure)
        bbox (list | Tensor): [x1, y1, x2, y2] — tọa độ bounding box
        color (str): Màu viền. Ví dụ: 'red', 'blue', '#FF0000'
        linewidth (int): Độ dày đường viền (pixel). Mặc định = 3
        label (str | None): Nhãn hiển thị trong legend. Ví dụ: 'GT', 'Pred'

    Ví dụ:
        bbox = [100, 50, 300, 200]
        → Rectangle tại (100, 50), kích thước (200, 150)
        → Vẽ hình chữ nhật từ góc trên-trái (100,50) đến góc dưới-phải (300,200)
    """
    # Chuyển bbox về list float (phòng trường hợp nhận Tensor)
    x1, y1, x2, y2 = [float(v) for v in bbox]

    # Tính chiều rộng và chiều cao
    width = x2 - x1
    height = y2 - y1

    # Tạo hình chữ nhật:
    #   - (x1, y1): góc trên-trái
    #   - width, height: kích thước
    #   - linewidth: độ dày viền
    #   - edgecolor: màu viền
    #   - facecolor: 'none' = không tô bên trong (chỉ vẽ viền)
    rect = patches.Rectangle(
        (x1, y1), width, height,
        linewidth=linewidth,
        edgecolor=color,
        facecolor='none',
        label=label,
    )

    # Thêm hình chữ nhật vào axes
    ax.add_patch(rect)


# ==============================================================================
# PHẦN 2: LOAD MODEL TỪ CHECKPOINT
# ==============================================================================

def load_model(checkpoint_path, config, device):
    """
    Load model SeqTR từ checkpoint file.

    Quy trình:
        1. Build vocabulary từ file annotations
        2. Load GloVe embeddings (hoặc dùng random nếu không có gensim)
        3. Khởi tạo model SeqTRDet
        4. Load weights từ checkpoint (ưu tiên EMA weights)
        5. Chuyển model sang eval mode

    Args:
        checkpoint_path (str): Đường dẫn tới file .pth (ví dụ: 'work_dir/best.pth')
        config: Config object chứa hyperparameters
        device (torch.device): 'cuda' hoặc 'cpu'

    Returns:
        model (SeqTRDet): Model đã load weights, ở eval mode
        token2idx (dict): Vocabulary — ánh xạ từ → index
        idx2token (dict): Vocabulary ngược — ánh xạ index → từ
    """
    # ----- Bước 1: Build vocabulary -----
    # Đọc tất cả các câu trong annotations để xây dựng bảng từ vựng
    print("📖 Building vocabulary...")
    token2idx, idx2token = build_vocab(config.ann_file)
    print(f"   Vocab size: {len(token2idx)}")

    # ----- Bước 2: Load GloVe embeddings -----
    # GloVe (Global Vectors) chuyển mỗi từ thành vector 300 chiều
    # Nếu không có thư viện gensim → dùng random embeddings (ảnh hưởng nhỏ khi test)
    try:
        import gensim.downloader as api
        print("📦 Loading GloVe embeddings...")
        glove_model = api.load("glove-wiki-gigaword-300")
        glove_matrix = build_glove_matrix(token2idx, glove_model, config.glove_dim)
    except ImportError:
        print("⚠️  gensim chưa cài → dùng random embeddings (không ảnh hưởng test)")
        glove_matrix = torch.randn(len(token2idx), config.glove_dim) * 0.01
        glove_matrix[0] = 0  # PAD token = zero vector

    # ----- Bước 3: Khởi tạo model -----
    print("🏗️  Building model...")
    model = SeqTRDet(config, glove_matrix).to(device)

    # ----- Bước 4: Load checkpoint -----
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Ưu tiên dùng EMA (Exponential Moving Average) weights nếu có.
    # EMA weights thường cho kết quả tốt hơn weights gốc vì nó là trung bình
    # trượt của nhiều epoch → ổn định hơn.
    if 'ema_shadow' in ckpt:
        print("   ✅ Using EMA weights (better performance)")
        model.load_state_dict(ckpt['ema_shadow'], strict=True)
    else:
        print("   ✅ Using standard model weights")
        model.load_state_dict(ckpt['model_state_dict'], strict=True)

    epoch = ckpt.get('epoch', '?')
    print(f"   Loaded from epoch {epoch}")

    # ----- Bước 5: Eval mode -----
    # model.eval() tắt dropout và batch normalization training behavior
    model.eval()

    return model, token2idx, idx2token


# ==============================================================================
# PHẦN 3: INFERENCE 1 SAMPLE
# ==============================================================================

def inference_single(model, dataset, index, device):
    """
    Chạy inference trên 1 sample cụ thể từ dataset.

    Quy trình:
        1. Lấy sample từ dataset (ảnh tensor, câu tokenized, GT bbox, metadata)
        2. Đưa vào model (forward pass với gt_bbox=None → inference mode)
        3. Model trả về predicted bbox
        4. Tính IoU giữa prediction và ground truth

    Args:
        model (SeqTRDet): Model đã load, eval mode
        dataset (RefCOCODataset): Dataset chứa samples
        index (int): Index của sample trong dataset
        device (torch.device): 'cuda' hoặc 'cpu'

    Returns:
        result (dict): Dictionary chứa tất cả thông tin cần thiết để vẽ:
            - 'image_id': ID ảnh COCO
            - 'expression': Câu referring expression
            - 'gt_bbox': [4] GT bbox (x1, y1, x2, y2) — hệ tọa độ ảnh 640×640
            - 'pred_bbox': [4] Predicted bbox — hệ tọa độ ảnh 640×640
            - 'iou': Giá trị IoU (float)
            - 'img_shape': (H, W) kích thước ảnh sau resize (trước pad)
            - 'scale': Tỉ lệ scale từ ảnh gốc → ảnh resize
    """
    # Lấy sample từ dataset
    # Dataset.__getitem__ trả về:
    #   img:      [3, 640, 640]  — ảnh đã resize + pad + normalize
    #   ref_inds: [max_token]    — câu đã tokenize thành indices
    #   gt_bbox:  [4]            — GT bbox [x1, y1, x2, y2] trên ảnh resize
    #   img_meta: dict           — metadata (image_id, expression, shapes, scale)
    img, ref_inds, gt_bbox, img_meta = dataset[index]

    # Thêm batch dimension: [3, 640, 640] → [1, 3, 640, 640]
    # Model yêu cầu input dạng batch, kể cả khi chỉ có 1 ảnh
    img_batch = img.unsqueeze(0).to(device)
    ref_batch = ref_inds.unsqueeze(0).to(device)

    # Forward pass — inference mode (gt_bbox=None)
    # Model sẽ:
    #   1. Trích xuất visual features (ResNet backbone)
    #   2. Mã hóa câu (BiGRU)
    #   3. Fusion visual + language
    #   4. Transformer decoder sinh 4 tokens auto-regressive
    #   5. Dequantize tokens → tọa độ float [x1, y1, x2, y2]
    #
    # [CẬP NHẬT] Model mới nhận img_shapes (Tensor [B, 4]) thay vì img_metas (list[dict])
    # img_shapes mỗi dòng = [pad_h, pad_w, img_h, img_w]
    # Lý do: để tương thích DataParallel (tensor chia được, dict thì không)
    img_shapes = torch.tensor([[
        img_meta['pad_shape'][0],   # pad_h (640)
        img_meta['pad_shape'][1],   # pad_w (640)
        img_meta['img_shape'][0],   # img_h (sau resize, trước pad)
        img_meta['img_shape'][1],   # img_w (sau resize, trước pad)
    ]], dtype=torch.float32).to(device)  # [1, 4]

    with torch.no_grad():
        pred_bbox = model(img_batch, ref_batch, img_shapes, gt_bbox=None)
    # pred_bbox: [1, 4] → squeeze → [4]
    pred_bbox = pred_bbox.squeeze(0).cpu()

    # Tính IoU (Intersection over Union) giữa prediction và ground truth
    # IoU ∈ [0, 1]: 1 = trùng khớp hoàn toàn, 0 = không giao nhau
    iou = compute_iou_batch(
        pred_bbox.unsqueeze(0),  # [1, 4]
        gt_bbox.unsqueeze(0)     # [1, 4]
    ).item()  # Scalar float

    # Trả về dict chứa tất cả thông tin
    result = {
        'image_id': img_meta['image_id'],
        'expression': img_meta['expression'],
        'gt_bbox': gt_bbox.numpy(),           # [4] numpy
        'pred_bbox': pred_bbox.numpy(),       # [4] numpy
        'iou': iou,
        'img_shape': img_meta['img_shape'],   # (H_resized, W_resized, 3)
        'scale': img_meta['scale_factor'][0], # scalar float
    }

    return result


# ==============================================================================
# PHẦN 4: LOAD ẢNH GỐC VÀ RESIZE (KHÔNG PAD)
# ==============================================================================

def load_display_image(image_id, img_dir, img_size):
    """
    Load ảnh gốc từ disk và resize (KHÔNG pad) — dùng để hiển thị.

    Tại sao không dùng ảnh từ dataset?
        Dataset trả về ảnh đã pad thành 640×640 (có viền đen).
        Ảnh pad không đẹp cho báo cáo → ta load lại ảnh gốc và chỉ resize.

    Tại sao cần resize?
        Vì GT bbox và Pred bbox đều ở hệ tọa độ ảnh resize (đã scale).
        Nếu dùng ảnh gốc (chưa resize), bbox sẽ lệch vị trí.
        → Cần resize ảnh về cùng hệ tọa độ với bbox.

    Args:
        image_id (int): ID ảnh COCO (ví dụ: 72 → COCO_train2014_000000000072.jpg)
        img_dir (str): Thư mục chứa ảnh COCO
        img_size (int): Kích thước max để resize (640)

    Returns:
        img_resized (np.ndarray): Ảnh đã resize, shape [H, W, 3], dtype=uint8
    """
    # Tạo đường dẫn file ảnh
    # Format: COCO_train2014_000000000072.jpg (12 chữ số, padding bằng 0)
    img_path = os.path.join(img_dir, "COCO_train2014_%012d.jpg" % image_id)

    # Đọc ảnh bằng PIL → chuyển sang numpy
    # PIL tự động đọc ở dạng RGB, shape [H, W, 3]
    pil_img = Image.open(img_path).convert('RGB')
    img_np = np.array(pil_img)

    # Resize giữ tỉ lệ: cạnh dài nhất = img_size (640)
    # CHÚ Ý: Không pad — ảnh sẽ có kích thước nhỏ hơn 640 ở 1 chiều
    # Ví dụ: ảnh 800×600 → resize → 640×480 (không thêm viền đen)
    img_resized, _ = resize_image_keep_ratio(img_np, img_size)

    return img_resized


# ==============================================================================
# PHẦN 5: TẠO FIGURE GRID (HÀM CHÍNH VẼ KẾT QUẢ)
# ==============================================================================

def create_qualitative_figure(results, img_dir, img_size, output_path,
                               ncols=4, figscale=5):
    """
    Tạo figure grid với nhiều ảnh, mỗi ảnh có GT bbox (đỏ) và Pred bbox (xanh).

    Layout ví dụ (4 ảnh, ncols=4 → 1 hàng × 4 cột):
    ┌────────────┬────────────┬────────────┬────────────┐
    │  (a) ...   │  (b) ...   │  (c) ...   │  (d) ...   │
    │  🔵 Pred   │  🔵 Pred   │  🔵 Pred   │  🔵 Pred   │
    │  🔴 GT     │  🔴 GT     │  🔴 GT     │  🔴 GT     │
    └────────────┴────────────┴────────────┴────────────┘

    Style theo paper TransVG:
        - 🔵 Xanh dương (Blue)  = Predicted bounding box
        - 🔴 Đỏ (Red)           = Ground truth bounding box
        - Tiêu đề = câu referring expression + IoU

    Args:
        results (list[dict]): List kết quả từ inference_single()
        img_dir (str): Thư mục ảnh COCO
        img_size (int): Kích thước resize (640)
        output_path (str): Đường dẫn file output (ví dụ: 'qualitative_results.png')
        ncols (int): Số cột trong grid (mặc định: 4)
        figscale (float): Tỉ lệ kích thước mỗi ô (inch). Mặc định: 5

    Returns:
        None (lưu file PNG)
    """
    n = len(results)

    # Tính số hàng cần thiết
    # Ví dụ: 4 ảnh, ncols=4 → 1 hàng; 8 ảnh, ncols=4 → 2 hàng
    nrows = (n + ncols - 1) // ncols  # Chia lên (ceiling division)

    # Tạo figure và grid các axes
    # figsize = (width, height) tính bằng inch
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * figscale, nrows * figscale),
        squeeze=False,  # Luôn trả về 2D array axes[row][col], kể cả 1 hàng
    )

    # Vẽ từng ảnh lên grid
    for i, result in enumerate(results):
        row = i // ncols  # Hàng thứ mấy
        col = i % ncols   # Cột thứ mấy
        ax = axes[row][col]

        # ----- Load ảnh gốc đã resize (không pad) -----
        img_display = load_display_image(result['image_id'], img_dir, img_size)

        # ----- Hiển thị ảnh -----
        ax.imshow(img_display)

        # ----- Vẽ Ground Truth bbox (ĐỎ) -----
        # Clip bbox để không vượt quá viền ảnh (phần pad bị cắt)
        gt_bbox = clip_bbox(result['gt_bbox'], img_display.shape)
        draw_bbox_on_axes(
            ax, gt_bbox,
            color='red',        # Đỏ = Ground Truth
            linewidth=3,
            label='Ground Truth',
        )

        # ----- Vẽ Predicted bbox (XANH DƯƠNG) -----
        pred_bbox = clip_bbox(result['pred_bbox'], img_display.shape)
        draw_bbox_on_axes(
            ax, pred_bbox,
            color='blue',       # Xanh = Prediction
            linewidth=3,
            label='Prediction',
        )

        # ----- Tiêu đề: expression + IoU -----
        # Cắt câu expression nếu quá dài (hơn 40 ký tự → thêm "...")
        expr = result['expression']
        if len(expr) > 50:
            expr = expr[:47] + "..."

        # Thêm ký tự (a), (b), (c), (d) giống paper
        label_char = chr(ord('a') + i)  # 0→'a', 1→'b', 2→'c', 3→'d'

        # Format tiêu đề: "(a) the man in red | IoU: 0.85"
        title = f"({label_char}) {expr}"
        iou_text = f"IoU: {result['iou']:.2f}"

        # Đặt tiêu đề phía dưới ảnh
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8, wrap=True)

        # Hiển thị IoU ở góc trên-bên-trái ảnh
        # Dùng text box với nền bán trong suốt để dễ đọc
        ax.text(
            8, 8, iou_text,
            fontsize=11,
            fontweight='bold',
            color='white',
            verticalalignment='top',
            bbox=dict(
                boxstyle='round,pad=0.3',  # Hình chữ nhật bo góc
                facecolor='black',          # Nền đen
                alpha=0.7,                  # Độ trong suốt 70%
            ),
        )

        # Tắt trục tọa độ (không hiển thị số tick x, y)
        ax.axis('off')

    # Ẩn các ô thừa (nếu số ảnh < nrows × ncols)
    for i in range(n, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axes[row][col].axis('off')

    # ----- Legend chung cho cả figure -----
    # Tạo 2 patch giả để hiển thị trong legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='none', edgecolor='blue', linewidth=2, label='Prediction'),
        Patch(facecolor='none', edgecolor='red', linewidth=2, label='Ground Truth'),
    ]
    fig.legend(
        handles=legend_elements,
        loc='lower center',           # Vị trí: giữa-dưới figure
        ncol=2,                        # 2 legend items nằm ngang
        fontsize=13,
        frameon=True,                  # Có viền
        edgecolor='gray',
        fancybox=True,
        shadow=True,
        bbox_to_anchor=(0.5, -0.02),   # Dịch xuống dưới 1 chút
    )

    # ----- Điều chỉnh layout và lưu -----
    plt.tight_layout()

    # Tạo thư mục output nếu chưa có
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Lưu file PNG
    # dpi=150: độ phân giải đủ cho báo cáo
    # bbox_inches='tight': cắt viền trắng thừa
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)

    print(f"\n💾 Đã lưu figure: {output_path}")


# ==============================================================================
# PHẦN 6: HÀM TIỆN ÍCH — CLIP BBOX
# ==============================================================================

def clip_bbox(bbox, img_shape):
    """
    Clip (giới hạn) tọa độ bbox để không vượt quá biên ảnh.

    Tại sao cần clip?
        GT bbox và Pred bbox ở hệ tọa độ ảnh 640×640 (có pad).
        Nhưng ảnh hiển thị chỉ có kích thước resize (ví dụ 640×480).
        Nếu bbox có y2=500 mà ảnh chỉ cao 480 → cần clip y2 về 480.

    Args:
        bbox (np.ndarray): [x1, y1, x2, y2]
        img_shape (tuple): (H, W, 3) hoặc (H, W) — kích thước ảnh hiển thị

    Returns:
        clipped_bbox (np.ndarray): [x1, y1, x2, y2] đã clip
    """
    h, w = img_shape[:2]
    clipped = bbox.copy()

    # Clip x1, x2 vào khoảng [0, W-1]
    clipped[0] = np.clip(clipped[0], 0, w - 1)
    clipped[2] = np.clip(clipped[2], 0, w - 1)

    # Clip y1, y2 vào khoảng [0, H-1]
    clipped[1] = np.clip(clipped[1], 0, h - 1)
    clipped[3] = np.clip(clipped[3], 0, h - 1)

    return clipped


# ==============================================================================
# PHẦN 7: LƯU TỪNG ẢNH RIÊNG LẺ
# ==============================================================================

def save_individual_images(results, img_dir, img_size, output_dir):
    """
    Lưu từng ảnh kết quả riêng lẻ (ngoài figure grid).

    Mỗi ảnh được lưu thành 1 file PNG riêng, tiện cho việc
    chèn từng ảnh vào slide hoặc báo cáo LaTeX.

    Args:
        results (list[dict]): List kết quả inference
        img_dir (str): Thư mục ảnh COCO
        img_size (int): Kích thước resize
        output_dir (str): Thư mục lưu ảnh output

    Output files:
        output_dir/qualitative_0.png
        output_dir/qualitative_1.png
        ...
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, result in enumerate(results):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        # Load ảnh
        img_display = load_display_image(result['image_id'], img_dir, img_size)
        ax.imshow(img_display)

        # Vẽ GT bbox (đỏ)
        gt_bbox = clip_bbox(result['gt_bbox'], img_display.shape)
        draw_bbox_on_axes(ax, gt_bbox, color='red', linewidth=3)

        # Vẽ Pred bbox (xanh)
        pred_bbox = clip_bbox(result['pred_bbox'], img_display.shape)
        draw_bbox_on_axes(ax, pred_bbox, color='blue', linewidth=3)

        # Tiêu đề
        ax.set_title(
            f"{result['expression']}\nIoU: {result['iou']:.2f}",
            fontsize=11, wrap=True,
        )
        ax.axis('off')

        # Lưu file
        out_path = os.path.join(output_dir, f"qualitative_{i}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"💾 Đã lưu {len(results)} ảnh riêng lẻ vào: {output_dir}/")


# ==============================================================================
# PHẦN 8: HÀM MAIN — ĐIỂM BẮT ĐẦU CHƯƠNG TRÌNH
# ==============================================================================

def main():
    """
    Hàm chính — điều phối toàn bộ quy trình visualization.

    Quy trình:
        1. Parse arguments từ command line
        2. Load model từ checkpoint
        3. Tạo dataset cho split cần visualize
        4. Chọn ngẫu nhiên N samples
        5. Chạy inference trên từng sample
        6. Tạo figure grid và lưu PNG
        7. (Tùy chọn) Lưu từng ảnh riêng lẻ
    """

    # ===== PARSE ARGUMENTS =====
    parser = argparse.ArgumentParser(
        description='SeqTR Detection — Qualitative Results Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python visualize_qualitative.py --checkpoint work_dir/best.pth
  python visualize_qualitative.py --checkpoint best.pth --split testA --num-samples 4
  python visualize_qualitative.py --checkpoint best.pth --indices 0 10 42 99
        """,
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Đường dẫn tới checkpoint (.pth). Ví dụ: work_dir/best.pth',
    )
    parser.add_argument(
        '--split', type=str, default='testA',
        choices=['val', 'testA', 'testB'],
        help='Split để lấy ảnh visualize (mặc định: testA)',
    )
    parser.add_argument(
        '--num-samples', type=int, default=4,
        help='Số ảnh cần visualize (mặc định: 4)',
    )
    parser.add_argument(
        '--indices', nargs='+', type=int, default=None,
        help='Chỉ định index cụ thể thay vì random. Ví dụ: --indices 0 10 42 99',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed cho reproducibility (mặc định: 42)',
    )
    parser.add_argument(
        '--output', type=str, default='qualitative_results.png',
        help='Đường dẫn file output (mặc định: qualitative_results.png)',
    )
    parser.add_argument(
        '--save-individual', action='store_true',
        help='Lưu thêm từng ảnh riêng lẻ (ngoài figure grid)',
    )
    parser.add_argument(
        '--ncols', type=int, default=4,
        help='Số cột trong figure grid (mặc định: 4)',
    )

    args = parser.parse_args()

    # ===== SETUP =====
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config

    print("=" * 60)
    print("🎨 SeqTR — Qualitative Results Visualization")
    print("=" * 60)
    print(f"   Device:     {device}")
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   Split:      {args.split}")
    print(f"   Num samples:{args.num_samples}")
    print(f"   Seed:       {args.seed}")
    print(f"   Output:     {args.output}")
    print("=" * 60)

    # ===== BƯỚC 1: LOAD MODEL =====
    print("\n🔧 BƯỚC 1: Load model")
    model, token2idx, idx2token = load_model(args.checkpoint, config, device)

    # ===== BƯỚC 2: TẠO DATASET =====
    print(f"\n📁 BƯỚC 2: Tạo dataset [{args.split}]")
    dataset = RefCOCODataset(
        config.ann_file, config.img_dir, args.split,
        token2idx, config.max_token, config.img_size,
    )
    print(f"   Dataset size: {len(dataset)} samples")

    # ===== BƯỚC 3: CHỌN SAMPLES =====
    print(f"\n🎲 BƯỚC 3: Chọn samples")

    if args.indices is not None:
        # Chế độ chỉ định index cụ thể
        selected_indices = args.indices
        print(f"   Chế độ: chỉ định index → {selected_indices}")
    else:
        # Chế độ random
        random.seed(args.seed)
        selected_indices = random.sample(
            range(len(dataset)),
            min(args.num_samples, len(dataset)),
        )
        print(f"   Chế độ: random (seed={args.seed}) → {selected_indices}")

    # ===== BƯỚC 4: INFERENCE =====
    print(f"\n🔍 BƯỚC 4: Inference trên {len(selected_indices)} samples")

    results = []
    for i, idx in enumerate(selected_indices):
        print(f"   [{i+1}/{len(selected_indices)}] Sample index={idx}...", end=" ")
        result = inference_single(model, dataset, idx, device)
        results.append(result)
        print(f"IoU={result['iou']:.3f}  expr=\"{result['expression']}\"")

    # ===== BƯỚC 5: TẠO FIGURE GRID =====
    print(f"\n🖼️  BƯỚC 5: Tạo figure")
    create_qualitative_figure(
        results=results,
        img_dir=config.img_dir,
        img_size=config.img_size,
        output_path=args.output,
        ncols=args.ncols,
    )

    # ===== BƯỚC 6 (TÙY CHỌN): LƯU ẢNH RIÊNG LẺ =====
    if args.save_individual:
        print(f"\n📸 BƯỚC 6: Lưu ảnh riêng lẻ")
        # Lưu vào thư mục cùng tên với file output (bỏ extension)
        individual_dir = os.path.splitext(args.output)[0] + "_individual"
        save_individual_images(results, config.img_dir, config.img_size, individual_dir)

    # ===== TỔNG KẾT =====
    print("\n" + "=" * 60)
    print("✅ HOÀN TẤT!")
    print("=" * 60)
    print(f"   📄 Figure grid: {args.output}")
    if args.save_individual:
        print(f"   📂 Ảnh riêng:   {individual_dir}/")

    # In bảng tổng kết IoU
    print(f"\n   {'Index':>6}  {'IoU':>6}  Expression")
    print(f"   {'─'*6}  {'─'*6}  {'─'*40}")
    for idx, res in zip(selected_indices, results):
        expr_short = res['expression'][:40]
        print(f"   {idx:>6}  {res['iou']:>6.3f}  {expr_short}")

    avg_iou = np.mean([r['iou'] for r in results])
    print(f"\n   Average IoU: {avg_iou:.3f}")
    print("=" * 60)


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()
