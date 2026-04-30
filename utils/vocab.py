import re
import json
import numpy as np
import torch

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