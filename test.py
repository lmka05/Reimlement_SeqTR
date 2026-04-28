# ==============================================================================
# test.py — Đánh giá model trên val / testA / testB từ checkpoint
# ==============================================================================
# Chạy:
#   python test.py --checkpoint work_dir/best.pth --splits val testA testB
# ==============================================================================

import os
import sys
import argparse

import torch

from config import Config
from dataset import build_vocab, build_glove_matrix, RefCOCODataset, build_dataloader
from model import SeqTRDet
from evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description='SeqTR Detection — Test')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Đường dẫn tới file checkpoint (.pth)')
    parser.add_argument('--splits', nargs='+', default=['val', 'testA', 'testB'],
                        help='Các split cần đánh giá (mặc định: val testA testB)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size cho evaluation (mặc định: 64)')
    args = parser.parse_args()

    config = Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Splits: {args.splits}")

    # 1. Build vocab + GloVe
    print("\n--- Building vocab ---")
    token2idx, idx2token = build_vocab(config.ann_file)
    print(f"Vocab size: {len(token2idx)}")

    try:
        import gensim.downloader as api
        glove_model = api.load("glove-wiki-gigaword-300")
        glove_matrix = build_glove_matrix(token2idx, glove_model, config.glove_dim)
    except ImportError:
        print("⚠️ gensim chưa cài. Dùng random embeddings.")
        glove_matrix = torch.randn(len(token2idx), config.glove_dim) * 0.01
        glove_matrix[0] = 0

    # 2. Build model + load checkpoint
    print("\n--- Loading model ---")
    model = SeqTRDet(config, glove_matrix).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Ưu tiên dùng EMA weights nếu có
    if 'ema_shadow' in ckpt:
        print("Using EMA weights")
        model.load_state_dict(ckpt['ema_shadow'], strict=True)
    else:
        model.load_state_dict(ckpt['model_state_dict'], strict=True)

    epoch = ckpt.get('epoch', '?')
    print(f"Loaded checkpoint from epoch {epoch}")

    # 3. Evaluate trên từng split
    print("\n" + "=" * 60)
    results = {}
    for split in args.splits:
        print(f"\n--- Evaluating on [{split}] ---")
        try:
            dataset = RefCOCODataset(
                config.ann_file, config.img_dir, split,
                token2idx, config.max_token, config.img_size
            )
            loader = build_dataloader(
                dataset, batch_size=args.batch_size,
                shuffle=False, num_workers=config.num_workers
            )
            acc, avg_iou = evaluate(model, loader, device, desc=split)
            results[split] = {'accuracy': acc, 'avg_iou': avg_iou}
        except KeyError:
            print(f"  ⚠️ Split '{split}' không tồn tại trong file annotations.")

    # 4. Tổng kết
    print("\n" + "=" * 60)
    print("TỔNG KẾT KẾT QUẢ")
    print("=" * 60)
    for split, res in results.items():
        print(f"  {split:8s}: Acc@IoU>=0.5 = {res['accuracy']:.2f}% | "
              f"Avg IoU = {res['avg_iou']:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
