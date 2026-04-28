# ==============================================================================
# train.py — Training Loop cho SeqTR Detection
# ==============================================================================
# File này là ENTRY POINT chính — chạy file này để huấn luyện model.
#
# Luồng:
#   1. Build vocab + GloVe matrix
#   2. Tạo dataset + dataloader (train, val)
#   3. Tạo model + optimizer + scheduler
#   4. Training loop: forward → loss → backward → update
#   5. Mỗi epoch: evaluate trên val set → save checkpoint
#
# Chạy:
#   python train.py
# ==============================================================================

import os
import sys
import copy
import time
import json
import random
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

# Import các module của project
from config import Config
from dataset import (
    build_vocab, build_glove_matrix, RefCOCODataset, build_dataloader
)
from model import SeqTRDet
from evaluate import evaluate


# ==============================================================================
# PHẦN 1: TIỆN ÍCH
# ==============================================================================

def set_seed(seed):
    """
    Đặt random seed cho reproducibility.
    Đảm bảo chạy lại sẽ ra cùng kết quả.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Deterministic mode (chậm hơn 1 chút nhưng reproducible)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EMA:
    """
    Exponential Moving Average (EMA) cho model weights.

    EMA duy trì 1 bản sao "trung bình trượt" của model weights:
        shadow = decay * shadow + (1 - decay) * current_weights

    Khi evaluate, dùng shadow weights thay vì current weights
    → kết quả thường tốt hơn vì shadow ổn định hơn (ít nhiễu).

    Ví dụ với decay=0.999:
        Mỗi step, shadow giữ 99.9% giá trị cũ + 0.1% giá trị mới.
        → Shadow thay đổi rất chậm, mượt mà, giảm ảnh hưởng của noise.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.step_count = 0
        # Deep copy toàn bộ parameters của model
        self.shadow = {name: param.clone().detach()
                       for name, param in model.state_dict().items()}

    def update(self, model):
        """Cập nhật shadow weights sau mỗi training step."""
        # Warmup: decay tăng dần ở đầu training
        decay = min(self.decay, (self.step_count + 1) / (self.step_count + 10))
        with torch.no_grad():
            for name, param in model.state_dict().items():
                if name in self.shadow:
                    # shadow = decay * shadow + (1-decay) * param
                    self.shadow[name].mul_(decay).add_(param, alpha=1 - decay)
        self.step_count += 1

    def apply(self, model):
        """Thay weights của model bằng shadow weights (dùng khi evaluate)."""
        self.backup = {name: param.clone()
                       for name, param in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model):
        """Khôi phục weights gốc của model (sau khi evaluate xong)."""
        model.load_state_dict(self.backup, strict=True)


def build_scheduler(optimizer, config):
    """
    Tạo LR scheduler: warmup + multi-step decay.

    Giai đoạn 1 (epoch 0 → warmup_epochs-1):
        LR tăng dần: lr * (epoch+1) / (warmup_epochs+1)
        Giúp model ổn định ở đầu training.

    Giai đoạn 2 (epoch warmup_epochs → decay_epoch-1):
        LR giữ nguyên = lr

    Giai đoạn 3 (epoch >= decay_epoch):
        LR giảm: lr * decay_ratio (mặc định giảm 10 lần)
        Giúp model fine-tune tinh ở cuối training.
    """
    def lr_lambda(epoch):
        if epoch < config.warmup_epochs:
            return (epoch + 1) / (config.warmup_epochs + 1)
        elif epoch < config.decay_epoch:
            return 1.0
        else:
            return config.decay_ratio

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_checkpoint(model, ema, optimizer, scheduler, epoch, accuracy, best_accuracy, config):
    """Lưu checkpoint."""
    os.makedirs(config.work_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy': accuracy,
        'best_accuracy': best_accuracy,
    }
    if ema is not None:
        checkpoint['ema_shadow'] = ema.shadow

    # Luôn lưu latest
    latest_path = os.path.join(config.work_dir, 'latest.pth')
    torch.save(checkpoint, latest_path)

    # Lưu best nếu accuracy cao nhất
    if accuracy >= best_accuracy:
        best_path = os.path.join(config.work_dir, 'best.pth')
        torch.save(checkpoint, best_path)
        print(f"  ★ New best model saved! Acc: {accuracy:.2f}%")


# ==============================================================================
# PHẦN 2: TRAINING LOOP
# ==============================================================================

def train_one_epoch(model, dataloader, optimizer, device, epoch, config, ema=None):
    """
    Train model qua 1 epoch.

    Args:
        model: SeqTRDet
        dataloader: Train DataLoader
        optimizer: Adam optimizer
        device: 'cuda'
        epoch (int): Epoch hiện tại (0-indexed)
        config: Config object
        ema: EMA object (hoặc None)

    Returns:
        avg_loss (float): Loss trung bình của epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    start_time = time.time()

    for batch_idx, (imgs, ref_inds, gt_bboxes, img_metas) in enumerate(dataloader):
        # Chuyển dữ liệu sang GPU
        imgs = imgs.to(device)
        ref_inds = ref_inds.to(device)
        gt_bboxes = gt_bboxes.to(device)

        # Forward: tính loss
        loss = model(imgs, ref_inds, img_metas, gt_bbox=gt_bboxes)

        # Backward: tính gradient
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping: giới hạn gradient norm để tránh exploding
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        # Update weights
        optimizer.step()

        # Update EMA
        if ema is not None:
            ema.update(model)

        # Tracking
        total_loss += loss.item()
        num_batches += 1

        # Log
        if (batch_idx + 1) % config.log_interval == 0:
            avg = total_loss / num_batches
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Loss: {avg:.4f} | LR: {lr:.6f} | Time: {elapsed:.1f}s")

    avg_loss = total_loss / num_batches
    return avg_loss


# ==============================================================================
# PHẦN 3: MAIN
# ==============================================================================

def main():
    """Entry point chính cho training."""
    config = Config

    # 0. Seed
    set_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Build vocabulary
    print("\n" + "=" * 60)
    print("STEP 1: Building vocabulary")
    print("=" * 60)
    token2idx, idx2token = build_vocab(config.ann_file)
    print(f"Vocabulary size: {len(token2idx)}")

    # 2. Load GloVe embeddings
    print("\n" + "=" * 60)
    print("STEP 2: Loading GloVe embeddings")
    print("=" * 60)
    try:
        import gensim.downloader as api
        print("Downloading GloVe (lần đầu sẽ mất ~5 phút, sau đó dùng cache)...")
        glove_model = api.load("glove-wiki-gigaword-300")
        glove_matrix = build_glove_matrix(token2idx, glove_model, config.glove_dim)
        del glove_model  # Giải phóng ~2GB RAM
        import gc; gc.collect()
    except ImportError:
        print("⚠️ gensim chưa cài. Dùng random embeddings (kết quả sẽ kém hơn).")
        print("   Cài gensim: pip install gensim")
        glove_matrix = torch.randn(len(token2idx), config.glove_dim) * 0.01
        glove_matrix[0] = 0  # PAD = zero

    # 3. Create datasets
    print("\n" + "=" * 60)
    print("STEP 3: Creating datasets")
    print("=" * 60)
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
        val_dataset, batch_size=config.batch_size * 2,
        shuffle=False, num_workers=config.num_workers
    )

    # 4. Build model
    print("\n" + "=" * 60)
    print("STEP 4: Building model")
    print("=" * 60)
    model = SeqTRDet(config, glove_matrix).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {train_params:,}")

    # 5. Optimizer + Scheduler
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0,
        amsgrad=True,
    )
    scheduler = build_scheduler(optimizer, config)

    # 6. EMA
    ema = EMA(model, decay=config.ema_decay) if config.ema else None

    # 7. Resume from checkpoint (nếu có)
    start_epoch = 0
    best_accuracy = 0.0
    latest_ckpt = os.path.join(config.work_dir, 'latest.pth')
    if os.path.exists(latest_ckpt):
        print(f"\nResuming from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_accuracy = ckpt.get('best_accuracy', 0.0)
        if ema is not None and 'ema_shadow' in ckpt:
            ema.shadow = ckpt['ema_shadow']
        print(f"Resumed from epoch {start_epoch}, best acc: {best_accuracy:.2f}%")

    # 8. Training loop
    print("\n" + "=" * 60)
    print("STEP 5: Start training!")
    print("=" * 60)

    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()

        # Train
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, config, ema
        )

        # Evaluate
        print(f"\n  --- Evaluating epoch {epoch+1} ---")

        if ema is not None:
            # Evaluate với EMA weights (thường tốt hơn)
            ema.apply(model)
            val_acc, val_iou = evaluate(model, val_loader, device, desc="val (EMA)")
            ema.restore(model)
        else:
            val_acc, val_iou = evaluate(model, val_loader, device, desc="val")

        # Save checkpoint
        save_checkpoint(
            model, ema, optimizer, scheduler,
            epoch, val_acc, best_accuracy, config
        )
        best_accuracy = max(best_accuracy, val_acc)

        # Step scheduler
        scheduler.step()

        # Epoch summary
        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config.epochs} Summary:")
        print(f"  Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"Best: {best_accuracy:.2f}% | LR: {lr:.6f} | Time: {epoch_time:.0f}s")
        print(f"{'='*60}\n")

    print(f"\n🎉 Training finished! Best accuracy: {best_accuracy:.2f}%")
    print(f"Checkpoints saved at: {config.work_dir}")


if __name__ == "__main__":
    main()
