import os
import re
import json
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

def clean_expression(expression):
    ''''
    Làm sạch câu : xoá các ký tự đặc biệt, lower, tách thành list các từ
    '''
    expression_cleaned = re.sub(r"([.,'!?\"()*#:;])", '', expression.lower())

    expression_cleaned = expression_cleaned.replace('-', ' ')

    expression_cleaned = expression_cleaned.replace('/',' ')

    return expression_cleaned.split()

def build_vocab(ann_file):
    """
    Xây dựng vocabulary (bảng từ vựng) từ file annotations.

    Duyệt qua TẤT CẢ các câu trong TẤT CẢ các split (train, val, testA, testB),
    thu thập mọi từ xuất hiện, gán cho mỗi từ 1 index duy nhất.

    Returns:
        token2idx (dict): Ánh xạ từ → số. Ví dụ: {"PAD": 0, "UNK": 1, "the": 3, "man": 4, ...}
        idx2token (dict): Ánh xạ ngược số → từ. Ví dụ: {0: "PAD", 1: "UNK", 3: "the", ...}
    """
    anns_all = json.load(open(ann_file, 'r'))
    token2idx = {
        "PAD": 0,
        "UNK": 1,
    
    }
    # Duyệt qua mọi split: "train", "val", "testA", "testB"
    for split_name in anns_all:
        # Mỗi split chứa list các annotation
        for ann in anns_all[split_name]:
            # Mỗi annotation có nhiều câu mô tả (expressions)
            for expression in ann['expressions']:
                # Làm sạch và tách câu thành các từ
                words = clean_expression(expression)
                for word in words:
                    # Nếu từ chưa có trong vocab, thêm vào với index mới
                    if word not in token2idx:
                        token2idx[word] = len(token2idx)

    # Tạo ánh xạ ngược: index → từ
    idx2token = {idx: token for token, idx in token2idx.items()}

    return token2idx, idx2token

def build_glove_matrix(token2idx, glove_model, glove_dim):
    """
    Tạo ma trận embedding GloVe cho vocabulary.

    Mỗi từ trong vocab sẽ được map sang vector GloVe 300 chiều.
    Từ nào không có trong GloVe → dùng vector ngẫu nhiên.

    Args:
        token2idx (dict): Vocabulary đã build. Ví dụ: {"PAD": 0, "the": 3, ...}
        glove_model: Model GloVe đã tải (từ gensim).
        glove_dim (int): Chiều của GloVe vector (300).

    Returns:
        weight_matrix (Tensor): [vocab_size, 300], dùng để khởi tạo nn.Embedding.

    Ví dụ:
        Vocab có 5000 từ → output shape = [5000, 300]
        weight_matrix[0] = vector của "PAD" = toàn số 0
        weight_matrix[3] = vector GloVe của "the" = [0.04, -0.2, ...]
    """
    vocab_size = len(token2idx)

    # Khởi tạo ma trận cho các từ không có trong glove (số nhỏ)
    weight_matrix = np.random.uniform(-0.01,0.01, (vocab_size, glove_dim)).astype(np.float32)
    weight_matrix[0] = np.zeros(glove_dim)

    # Đếm sô từ kiếm được trong glove
    found =0
    for word, idx in token2idx.items():
        if word in glove_model:
            weight_matrix[idx] = glove_model[word]
            found += 1
    print(f"Tìm được {found} trên {vocab_size} từ trong glove")

    return torch.from_numpy(weight_matrix)

def tokenize_expression(expression, token2idx, max_token) :
    """
    Chuyển 1 câu referring expression thành tensor các index.

    Args:
        expression (str): Câu mô tả. Ví dụ: "the man in red"
        token2idx (dict): Vocabulary.
        max_token (int): Độ dài cố định (padding/truncation).

    Returns:
        ref_inds (Tensor): [max_token], dtype=torch.long

    Ví dụ (max_token=8):
        "the man in red" → [3, 4, 5, 89, 0, 0, 0, 0]
                            ↑  ↑  ↑  ↑   ↑  ↑  ↑  ↑
                          the man in red PAD PAD PAD PAD
    """
    # Tạo tensor toàn 0 (PAD) với kích thước max_token
    ref_inds = torch.zeros(max_token, dtype=torch.long)

    # Tách câu thành các từ đã làm sạch
    words = clean_expression(expression)

    for i, word in enumerate(words):
        # Dừng nếu đã đủ max_token từ (truncation)
        if i >= max_token:
            break

        if word in token2idx:
            # Từ có trong vocab → dùng index của nó
            ref_inds[i] = token2idx[word]
        else:
            # Từ lạ → dùng UNK token (index 1)
            ref_inds[i] = token2idx['UNK']

    return ref_inds

def resize_image_keep_ratio(img, max_size):
    """
    Resize ảnh sao cho cạnh dài nhất = max_size, GIỮ NGUYÊN tỉ lệ.

    Args:
        img (np.ndarray): Ảnh gốc, shape [H, W, 3], dtype=uint8
        max_size (int): Kích thước tối đa (640)

    Returns:
        resized_img (np.ndarray): Ảnh đã resize, shape [new_H, new_W, 3]
        scale (float): Tỉ lệ scale. Ví dụ: 0.8 nghĩa là ảnh bị thu nhỏ 80%

    Ví dụ:
        Ảnh 800x600, max_size=640
        scale = 640 / max(800, 600) = 640/800 = 0.8
        new_size = (800*0.8, 600*0.8) = (640, 480)
    """

    h,w = img.shape[:2]

    scale = max_size / max(h,w)

    new_h , new_w = int(h*scale), int(w*scale)

    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((new_w, new_h))

    resized_img = np.array(pil_img)

    return resized_img, scale

def pad_image_to_square(img, target_size, pad_value =0):
     
        """
        Pad ảnh về kích thước target_size x target_size.
        Padding thêm ở bên PHẢI và bên DƯỚI.

        Args:
            img (np.ndarray): Ảnh đã resize, shape [H, W, 3]
            target_size (int): Kích thước đích (640)
            pad_value (int): Giá trị pixel để pad (0 = đen)

        Returns:
            padded_img (np.ndarray): Ảnh đã pad, shape [target_size, target_size, 3]

        Ví dụ:
            Ảnh 640x480 → pad thêm 160 dòng phía dưới → 640x640
            ┌──────────────┐
            │  Ảnh gốc     │ 480px
            │  640 x 480   │
            ├──────────────┤
            │  Padding (0) │ 160px
            └──────────────┘
                640px
        """
        h,w = img.shape[:2]

        padded = np.full((target_size,target_size,3), pad_value, dtype = img.dtype)

        padded[:h, :w, :] = img
        
        return padded

def normalize_image(img):
    """
    Chuyển các giá trị pixel về khoảng [0, 1]

    """

    return img.astype(np.float32)/255.0

def image_to_tensor(img):
    """
    Chuyển numpy image thành tensor, đổi thứ tự axes từ [H, W, C] → [C, H, W]
    """

    img = np.transpose(img, (2, 0, 1))
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)

    return img

def transform_bbox(bbox_xywh, scale, img_shape_after_resize):
    """
    Chuyển đổi bounding box cho phù hợp với ảnh đã resize.

    Annotations gốc lưu bbox dạng [x, y, w, h] (COCO format) -> chuyển sang [x1, y1, x2, y2] (corner format) rồi scale.

    Args:
        bbox_xywh (list): [x, y, w, h] từ annotation gốc
        scale (float): Tỉ lệ resize đã áp dụng lên ảnh
        img_shape_after_resize (tuple): (new_h, new_w) sau khi resize

    Returns:
        bbox (Tensor): [4], dạng [x1, y1, x2, y2], dtype=torch.float32

    Ví dụ:
        bbox_xywh = [100, 50, 200, 150]  → gốc: top-left (100,50), kích thước 200x150
        scale = 0.8
        → Sau scale: [80, 40, 160, 120]  → xywh format
        → Chuyển sang xyxy: [80, 40, 240, 160]  → x2 = x1 + w, y2 = y1 + h
        → Clip: đảm bảo tọa độ không vượt quá kích thước ảnh
    """
    x, y, w, h = bbox_xywh

    # Scale tọa độ theo tỉ lệ resize
    x1 = x * scale
    y1 = y * scale
    x2 = (x + w) * scale
    y2 = (y + h) * scale

    # Clip tọa độ để không vượt quá biên ảnh
    new_h, new_w = img_shape_after_resize
    x1 = np.clip(x1, 0, new_w - 1)
    y1 = np.clip(y1, 0, new_h - 1)
    x2 = np.clip(x2, 0, new_w - 1)
    y2 = np.clip(y2, 0, new_h - 1)

    return torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

class RefCOCODataset(Dataset):
    """
    Dataset class cho RefCOCO — dùng với PyTorch DataLoader.

    Mỗi sample gồm:
        - img: Tensor ảnh [3, 640, 640]
        - ref_inds: Tensor câu đã tokenize [max_token]
        - gt_bbox: Tensor bounding box [4] dạng [x1, y1, x2, y2]
        - img_meta: Dict chứa thông tin ảnh (dùng khi dequantize)
    """

    def __init__(self, ann_file, img_dir, split, token2idx, max_token=15, img_size=640):
        """
        Args:
            ann_file (str): Đường dẫn tới instances.json
            img_dir (str): Đường dẫn tới thư mục chứa ảnh COCO
            split (str): 'train', 'val', 'testA', hoặc 'testB'
            token2idx (dict): Vocabulary đã build
            max_token (int): Độ dài tối đa của câu
            img_size (int): Kích thước ảnh đích sau resize+pad
        """
        super().__init__()

        self.img_dir = img_dir
        self.split = split
        self.token2idx = token2idx
        self.max_token = max_token
        self.img_size = img_size

        # Đọc annotations cho split cụ thể
        # anns_all có cấu trúc: {"train": [...], "val": [...], "testA": [...], "testB": [...]}
        anns_all = json.load(open(ann_file, 'r'))
        self.anns = anns_all[split]

        print(f"[{split}] Loaded {len(self.anns)} samples")

    def __len__(self):
        """Trả về số lượng samples trong dataset."""
        return len(self.anns)

    def __getitem__(self, index):
        """
        Lấy 1 sample tại vị trí index.
        Đây là hàm QUAN TRỌNG NHẤT — DataLoader sẽ gọi hàm này cho mỗi sample.

        Returns:
            img (Tensor): [3, img_size, img_size] — ảnh đã xử lý
            ref_inds (Tensor): [max_token] — câu đã tokenize
            gt_bbox (Tensor): [4] — bounding box [x1, y1, x2, y2]
            img_meta (dict): Thông tin bổ sung để dequantize khi inference
        """
        ann = self.anns[index]

        # ====== 1. LOAD ẢNH ======
        # Tạo đường dẫn file ảnh từ image_id
        # Ảnh COCO train2014 có tên dạng: COCO_train2014_000000000072.jpg
        # %012d = pad image_id thành 12 chữ số: 72 → 000000000072
        img_path = os.path.join(
            self.img_dir,
            "COCO_train2014_%012d.jpg" % ann['image_id']
        )

        # Đọc ảnh bằng PIL → chuyển sang numpy array
        # PIL đọc ảnh dạng RGB, shape [H, W, 3]
        pil_img = Image.open(img_path).convert('RGB')
        img = np.array(pil_img)

        # Lưu kích thước gốc (cần cho evaluation sau này)
        ori_h, ori_w = img.shape[:2]

        # ====== 2. RESIZE ẢNH ======
        # Resize giữ tỉ lệ sao cho cạnh dài nhất = img_size (640)
        img, scale = resize_image_keep_ratio(img, self.img_size)
        resized_h, resized_w = img.shape[:2]

        # ====== 3. PAD ẢNH ======
        # Pad về img_size x img_size (640x640)
        img = pad_image_to_square(img, self.img_size)

        # ====== 4. NORMALIZE + TO TENSOR ======
        img = normalize_image(img)      # [H, W, 3] float32 [0, 1]
        img = image_to_tensor(img)      # [3, H, W] tensor

        # ====== 5. CHỌN VÀ TOKENIZE CÂU ======
        expressions = ann['expressions']
        if self.split == 'train':
            # Training: random chọn 1 câu (data augmentation cho text)
            expression = random.choice(expressions)
        else:
            # Val/Test: luôn chọn câu đầu tiên (để kết quả consistent)
            expression = expressions[0]

        ref_inds = tokenize_expression(expression, self.token2idx, self.max_token)

        # ====== 6. TRANSFORM BOUNDING BOX ======
        # ann['bbox'] lưu dạng [x, y, w, h] → chuyển sang [x1, y1, x2, y2] đã scale
        gt_bbox = transform_bbox(
            ann['bbox'],
            scale=scale,
            img_shape_after_resize=(resized_h, resized_w)
        )

        # ====== 7. TẠO IMG_META ======
        # Dict chứa thông tin cần thiết để dequantize tọa độ khi inference
        img_meta = {
            'image_id': ann['image_id'],
            'expression': expression,
            'ori_shape': (ori_h, ori_w, 3),         # Kích thước ảnh gốc
            'img_shape': (resized_h, resized_w, 3),  # Kích thước sau resize
            'pad_shape': (self.img_size, self.img_size, 3),  # Kích thước sau pad
            'scale_factor': np.array([scale, scale, scale, scale], dtype=np.float32),
        }

        return img, ref_inds, gt_bbox, img_meta
    

def collate_fn(batch):
    """
    Custom collate function cho DataLoader.

    DataLoader mặc định chỉ gom được tensors cùng shape.
    img_meta là dict nên cần xử lý riêng.

    Args:
        batch (list): List các tuples (img, ref_inds, gt_bbox, img_meta)
                      từ __getitem__, với len = batch_size

    Returns:
        imgs (Tensor): [B, 3, H, W] — batch ảnh đã stack
        ref_inds (Tensor): [B, max_token] — batch câu đã stack
        gt_bboxes (Tensor): [B, 4] — batch bboxes đã stack
        img_metas (list[dict]): List các img_meta dict
    """
    # zip(*batch) "unzip" list of tuples thành separate lists
    # batch = [(img1, ref1, bb1, meta1), (img2, ref2, bb2, meta2), ...]
    # → imgs = (img1, img2, ...), refs = (ref1, ref2, ...), ...
    imgs, ref_inds, gt_bboxes, img_metas = zip(*batch)

    # torch.stack gom list tensors thành 1 tensor với batch dimension
    # [img1, img2, ...] → [B, 3, H, W]
    imgs = torch.stack(imgs, dim=0)
    ref_inds = torch.stack(ref_inds, dim=0)
    gt_bboxes = torch.stack(gt_bboxes, dim=0)

    # img_metas giữ nguyên dạng list of dicts
    img_metas = list(img_metas)

    return imgs, ref_inds, gt_bboxes, img_metas

def build_dataloader(dataset, batch_size, shuffle=True, num_workers=2):
    """
    Tạo DataLoader từ dataset.  

    Args:
        dataset: RefCOCODataset instance
        batch_size (int): Số sample mỗi batch
        shuffle (bool): Có xáo trộn dữ liệu không (True cho train, False cho val)
        num_workers (int): Số process song song load data

    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,      # Dùng custom collate vì có img_meta
        pin_memory=True,             # Tăng tốc CPU→GPU transfer
        drop_last=(shuffle == True), # Drop batch cuối nếu không đủ (chỉ khi train)
    )

if __name__ == "__main__":
    """
    Chạy: python dataset.py
    Để test xem dataset hoạt động đúng chưa.
    Cần có file instances.json và ảnh COCO ở đúng đường dẫn trong config.
    """
    from config import Config

    print("=" * 60)
    print("TEST DATASET")
    print("=" * 60)

    # 1. Build vocabulary
    print("\n--- Building vocabulary ---")
    token2idx, idx2token = build_vocab(Config.ann_file)
    print(f"Vocabulary size: {len(token2idx)}")
    print(f"Sample words: {list(token2idx.items())[:10]}")

    # 2. Tạo dataset
    print("\n--- Creating dataset ---")
    train_dataset = RefCOCODataset(
        ann_file=Config.ann_file,
        img_dir=Config.img_dir,
        split='train',
        token2idx=token2idx,
        max_token=Config.max_token,
        img_size=Config.img_size,
    )

    # 3. Lấy 1 sample
    print("\n--- Getting 1 sample ---")
    img, ref_inds, gt_bbox, img_meta = train_dataset[0]
    print(f"Image shape: {img.shape}")            # [3, 640, 640]
    print(f"Image dtype: {img.dtype}")             # torch.float32
    print(f"Image range: [{img.min():.2f}, {img.max():.2f}]")  # [0, 1]
    print(f"Ref indices: {ref_inds}")              # [idx1, idx2, ..., 0, 0, 0]
    print(f"Ref words: {[idx2token.get(i.item(), '?') for i in ref_inds if i > 0]}")
    print(f"GT bbox: {gt_bbox}")                   # [x1, y1, x2, y2]
    print(f"Image meta: {img_meta}")

    # 4. Test DataLoader
    print("\n--- Testing DataLoader ---")
    loader = build_dataloader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    imgs, refs, bboxes, metas = batch
    print(f"Batch images: {imgs.shape}")    # [4, 3, 640, 640]
    print(f"Batch refs: {refs.shape}")      # [4, 15]
    print(f"Batch bboxes: {bboxes.shape}")  # [4, 4]
    print(f"Batch metas: {len(metas)} dicts")  # 4
