
# Script này tự động chạy các thí nghiệm ablation study:
#   A. Language Feature Pooling: max / mean / last
#   B. Token Weight: uniform / x1-heavy / x1y1-heavy / decreasing
#
# Mỗi experiment train N epoch → evaluate trên val set → lưu kết quả.

import os
import sys
import copy
import time
import json
import random
import gc
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

# Import các module của project
from config import Config
from utils.vocab import build_vocab, build_glove_matrix
from datasets.dataset import RefCOCODataset, build_dataloader
from models.model import SeqTRDet
from evaluate import evaluate
from train import set_seed, EMA, build_scheduler, train_one_epoch


# ==============================================================================
# ĐỊNH NGHĨA CÁC ABLATION EXPERIMENTS
# ==============================================================================

ABLATION_CONFIGS = {
    # ─── Nhóm A: Language Feature Pooling ───
    # Giữ token_weights = None (đều), chỉ thay đổi pooling
    "A1": {
        "name": "A1_pooling_max",
        "desc": "Max pooling (baseline — mặc định)",
        "pooling": "max",
        "token_weights": None,
    },
    "A2": {
        "name": "A2_pooling_mean",
        "desc": "Mean pooling — trung bình hidden states",
        "pooling": "mean",
        "token_weights": None,
    },
    "A3": {
        "name": "A3_pooling_last",
        "desc": "Last hidden state — nối forward+backward",
        "pooling": "last",
        "token_weights": None,
    },

    # ─── Nhóm B: Token Weight ───
    # Giữ pooling = "max" (tốt nhất), chỉ thay đổi token_weights
    "B1": {
        "name": "B1_weight_uniform",
        "desc": "Uniform weights [1,1,1,1,1] (baseline)",
        "pooling": "max",
        "token_weights": None,  # None = đều
    },
    "B2": {
        "name": "B2_weight_x1_heavy",
        "desc": "x1 heavy [1.5,1,1,1,1] — theo paper gốc",
        "pooling": "max",
        "token_weights": [1.5, 1.0, 1.0, 1.0, 1.0],
    },
    "B3": {
        "name": "B3_weight_x1y1_heavy",
        "desc": "x1,y1 heavy [1.5,1.5,1,1,1]",
        "pooling": "max",
        "token_weights": [1.5, 1.5, 1.0, 1.0, 1.0],
    },
    "B4": {
        "name": "B4_weight_decreasing",
        "desc": "Decreasing [2,1.5,1,1,0.5]",
        "pooling": "max",
        "token_weights": [2.0, 1.5, 1.0, 1.0, 0.5],
    },
}


def get_experiments(exp_filter=None):
    """
    Lọc experiments dựa trên filter.
    
    Args:
        exp_filter: None (tất cả), "A" (nhóm A), "B" (nhóm B), 
                    hoặc "A1", "B2", ... (cụ thể)
    
    Returns:
        dict: {exp_id: config}
    """
    if exp_filter is None:
        return ABLATION_CONFIGS
    
    exp_filter = exp_filter.upper()
    
    # Filter cụ thể: "A1", "B2", ...
    if exp_filter in ABLATION_CONFIGS:
        return {exp_filter: ABLATION_CONFIGS[exp_filter]}
    
    # Filter nhóm: "A" hoặc "B"
    return {k: v for k, v in ABLATION_CONFIGS.items() if k.startswith(exp_filter)}


# CHẠY 1 EXPERIMENT

def run_experiment(exp_id, exp_config, config, glove_matrix, token2idx,
                   num_epochs=5, device=None):
    """
    Chạy 1 ablation experiment.
    
    Args:
        exp_id (str): ID experiment (e.g., "A1")
        exp_config (dict): Config cho experiment
        config: Config object gốc
        glove_matrix: Ma trận GloVe embeddings
        token2idx: Vocabulary mapping
        num_epochs (int): Số epoch train
        device: torch device
    
    Returns:
        dict: Kết quả {exp_id, name, desc, val_acc, val_iou, avg_loss, time}
    """
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT {exp_id}: {exp_config['desc']}")
    print(f"  Pooling: {exp_config['pooling']} | Token Weights: {exp_config['token_weights']}")
    print(f"  Training for {num_epochs} epochs")
    print(f"{'='*70}\n")

    # --- 1. Override config ---
    config.pooling = exp_config["pooling"]
    config.token_weights = exp_config["token_weights"]
    
    # Đặt work_dir riêng cho mỗi experiment
    base_work_dir = config.work_dir.rstrip("/")
    exp_work_dir = f"{base_work_dir}/ablation_{exp_config['name']}"
    config.work_dir = exp_work_dir
    os.makedirs(exp_work_dir, exist_ok=True)

    # --- 2. Reset seed ---
    set_seed(config.seed)

    # --- 3. Tạo datasets ---
    train_dataset = RefCOCODataset(
        config.ann_file, config.img_dir, 'train',
        token2idx, config.max_token, config.img_size
    )
    val_dataset = RefCOCODataset(
        config.ann_file, config.img_dir, 'val',
        token2idx, config.max_token, config.img_size
    )
    train_loader = build_dataloader(
        train_dataset, config.batch_size, shuffle=True, num_workers=config.num_workers
    )
    val_loader = build_dataloader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers
    )

    # --- 4. Build model (from scratch) ---
    model = SeqTRDet(config, glove_matrix).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {total_params:,} total, {train_params:,} trainable")

    # --- 5. Optimizer + Scheduler ---
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0,
        amsgrad=True,
    )
    scheduler = build_scheduler(optimizer, config)

    # --- 6. EMA ---
    ema = EMA(model, decay=config.ema_decay) if config.ema else None

    # --- 7. Multi-GPU ---
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model = nn.DataParallel(model)

    # --- 8. Training loop ---
    exp_start = time.time()
    best_accuracy = 0.0
    best_iou = 0.0
    final_loss = 0.0

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Train
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, config, ema
        )
        final_loss = avg_loss

        # Evaluate
        print(f"\n  --- [{exp_id}] Evaluating epoch {epoch+1}/{num_epochs} ---")
        if ema is not None:
            raw_model = model.module if hasattr(model, 'module') else model
            ema.apply(raw_model)
            val_acc, val_iou = evaluate(model, val_loader, device, 
                                         desc=f"{exp_id} val (EMA)")
            ema.restore(raw_model)
        else:
            val_acc, val_iou = evaluate(model, val_loader, device, 
                                         desc=f"{exp_id} val")

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_iou = val_iou

        # Step scheduler
        scheduler.step()

        # Giải phóng memory
        gc.collect()
        torch.cuda.empty_cache()

        epoch_time = time.time() - epoch_start
        print(f"  [{exp_id}] Epoch {epoch+1}/{num_epochs} | "
              f"Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"Best: {best_accuracy:.2f}% | Time: {epoch_time:.0f}s")

    exp_time = time.time() - exp_start

    # --- 9. Cleanup ---
    del model, optimizer, scheduler, ema
    del train_loader, val_loader, train_dataset, val_dataset
    gc.collect()
    torch.cuda.empty_cache()

    result = {
        "exp_id": exp_id,
        "name": exp_config["name"],
        "desc": exp_config["desc"],
        "pooling": exp_config["pooling"],
        "token_weights": str(exp_config["token_weights"]),
        "val_acc": best_accuracy,
        "val_iou": best_iou,
        "final_loss": final_loss,
        "time_seconds": exp_time,
        "num_epochs": num_epochs,
    }

    print(f"\n  ✅ [{exp_id}] Done! Best Acc: {best_accuracy:.2f}% | "
          f"mIoU: {best_iou:.4f} | Time: {exp_time:.0f}s\n")

    return result


# MAIN
def main():
    parser = argparse.ArgumentParser(description="SeqTR Ablation Study")
    parser.add_argument("--exp", type=str, default=None,
                        help="Filter experiments: A, B, A1, B2, etc. "
                             "Default: run all")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs per experiment (default: 5)")
    args = parser.parse_args()

    # Lấy danh sách experiments
    experiments = get_experiments(args.exp)
    if not experiments:
        print(f"❌ No experiments match filter: {args.exp}")
        print(f"   Available: {', '.join(ABLATION_CONFIGS.keys())}")
        sys.exit(1)

    print(f"\n{'#'*70}")
    print(f"  SEQTR ABLATION STUDY")
    print(f"  Experiments: {', '.join(experiments.keys())}")
    print(f"  Epochs per experiment: {args.epochs}")
    print(f"{'#'*70}\n")

    config = Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- 1. Build vocab (1 lần duy nhất) ---
    print("\n[SETUP] Building vocabulary...")
    token2idx, idx2token = build_vocab(config.ann_file)
    print(f"  Vocabulary size: {len(token2idx)}")

    # --- 2. Load GloVe (1 lần duy nhất) ---
    print("[SETUP] Loading GloVe embeddings...")
    try:
        import gensim.downloader as api
        glove_model = api.load("glove-wiki-gigaword-300")
        glove_matrix = build_glove_matrix(token2idx, glove_model, config.glove_dim)
        del glove_model
        gc.collect()
    except ImportError:
        print("  ⚠️ gensim not available, using random embeddings")
        glove_matrix = torch.randn(len(token2idx), config.glove_dim) * 0.01
        glove_matrix[0] = 0

    # --- 3. Lưu work_dir gốc ---
    original_work_dir = config.work_dir

    # --- 4. Chạy từng experiment ---
    all_results = []
    total_start = time.time()

    for exp_id, exp_config in experiments.items():
        # Reset work_dir về gốc trước mỗi experiment
        config.work_dir = original_work_dir

        result = run_experiment(
            exp_id, exp_config, config, glove_matrix, token2idx,
            num_epochs=args.epochs, device=device
        )
        all_results.append(result)

        # Lưu kết quả tạm sau mỗi experiment (phòng crash)
        config.work_dir = original_work_dir
        results_path = os.path.join(original_work_dir, "ablation_results.json")
        os.makedirs(original_work_dir, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

    total_time = time.time() - total_start

    # --- 5. In bảng kết quả ---
    print(f"\n\n{'='*70}")
    print(f"  ABLATION STUDY RESULTS")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"{'='*70}\n")

    # Header
    print(f"{'ID':<6} {'Name':<25} {'Pooling':<8} {'Weights':<22} "
          f"{'Acc@0.5':>8} {'mIoU':>8} {'Loss':>8} {'Time':>6}")
    print("-" * 100)

    for r in all_results:
        weights_str = r["token_weights"] if r["token_weights"] != "None" else "uniform"
        time_str = f"{r['time_seconds']/60:.1f}m"
        print(f"{r['exp_id']:<6} {r['name']:<25} {r['pooling']:<8} "
              f"{weights_str:<22} {r['val_acc']:>7.2f}% {r['val_iou']:>8.4f} "
              f"{r['final_loss']:>8.4f} {time_str:>6}")

    print(f"\n📊 Results saved to: {results_path}")
    print(f"🎉 Ablation study finished!")


if __name__ == "__main__":
    main()
