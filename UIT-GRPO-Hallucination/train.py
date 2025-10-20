# train.py
import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from collections import defaultdict
import logging

# Imports từ project
from src.config import get_training_args
from src.dataset import ViHalluSet, SmartCollator, create_weighted_sampler, _normalize_label, LABEL2ID
from src.model import build_model_and_tokenizer
from src.trainer import train_grpo_epoch, evaluate_grpo
from src.utils import plot_training_curves, plot_confusion_matrix, save_model_for_inference
from src.logging_utils import setup_logger # <-- THÊM MỚI

def set_seed(seed_value):
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def main(args):
    # Tạo thư mục save
    os.makedirs(args.save_dir, exist_ok=True)

    # --- THÊM MỚI: Thiết lập Logger ---
    log_file_path = os.path.join(args.save_dir, 'training.log')
    logger = setup_logger(log_file_path)
    logger.info("Logger initialized. Starting training...")
    logger.info(f"Saving logs to {log_file_path}")
    logger.info(f"Training arguments: {args}")
    # ---------------------------------

    # Set seed
    set_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")

    # Tải dữ liệu
    if not os.path.exists(args.train_file):
        logger.error(f"Train file not found: {args.train_file}") # <-- THAY ĐỔI
        raise FileNotFoundError(f"Train file not found: {args.train_file}")
    
    df_full = pd.read_csv(args.train_file)
    df_train, df_val = train_test_split(
        df_full, 
        test_size=args.val_split, 
        random_state=args.seed, 
        stratify=df_full['label']
    )

    logger.info(f"Data loaded: {len(df_train)} train, {len(df_val)} val") # <-- THAY ĐỔI

    # Build model
    logger.info(f"Building model and tokenizer from {args.model_name}...") # <-- THÊM MỚI
    model, ref_model, tok = build_model_and_tokenizer(args)
    logger.info("Model and tokenizer built successfully.") # <-- THÊM MỚI

    # Tạo dataset
    train_dataset = ViHalluSet(df_train)
    val_dataset = ViHalluSet(df_val)
    logger.info("Datasets created.") # <-- THÊM MỚI

    # Tạo collator và sampler
    collator = SmartCollator(tok, args.max_len, args)
    train_labels = [LABEL2ID[_normalize_label(row['label'])] for _, row in df_train.iterrows()]
    sampler = create_weighted_sampler(train_labels)
    logger.info("Collator and WeightedSampler created.") # <-- THÊM MỚI

    # Tạo DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        collate_fn=collator, num_workers=args.workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        collate_fn=collator, num_workers=args.workers, pin_memory=True
    )
    logger.info("DataLoaders created.") # <-- THÊM MỚI

    # Tạo Optimizer và Scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.wd
    )
    total_steps = (len(train_loader) // args.accum_steps) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    logger.info(f"Optimizer and Scheduler created. Total steps: {total_steps}, Warmup steps: {warmup_steps}") # <-- THÊM MỚI

    # Vòng lặp training
    history = defaultdict(list)
    best_f1 = 0
    epochs_no_improve = 0

    logger.info("--- Starting Training Loop ---") # <-- THÊM MỚI
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n--- Epoch {epoch}/{args.epochs} ---") # <-- THAY ĐỔI
        
        train_metrics = train_grpo_epoch(model, ref_model, train_loader, optimizer, scheduler, args, epoch)
        for k, v in train_metrics.items():
            history[f'train_{k}'].append(v)
        
        # Chuyển metrics thành string để log
        train_metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
        logger.info(f"Epoch {epoch} Train Metrics: {train_metrics_str}") # <-- THAY ĐỔI

        val_metrics = evaluate_grpo(model, val_loader, args, desc=f"Validating Epoch {epoch}")
        for k in ['loss', 'macro_f1', 'weighted_f1']:
            history[f'val_{k}'].append(val_metrics[k])
        
        logger.info(f"Validation Macro F1: {val_metrics['macro_f1']:.4f}") # <-- THAY ĐỔI
        
        cm_path = os.path.join(args.save_dir, f'confusion_matrix_epoch_{epoch}.png')
        plot_confusion_matrix(val_metrics['confusion_matrix'], save_path=cm_path)

        if val_metrics['macro_f1'] > best_f1:
            epochs_no_improve = 0
            best_f1 = val_metrics['macro_f1']
            logger.info(f">> New best model found with Macro F1: {best_f1:.4f}") # <-- THAY ĐỔI
            logger.info(f"Classification Report:\n{val_metrics['report']}") # <-- THAY ĐỔI
            
            save_path = os.path.join(args.save_dir, "best_model.pt")
            save_model_for_inference(model, tok, save_path)
        else:
            epochs_no_improve += 1
            logger.warning(f"No improvement for {epochs_no_improve}/{args.early_stop_limit} epochs") # <-- THAY ĐỔI
            if epochs_no_improve >= args.early_stop_limit:
                logger.info(f"Early stopping triggered after {args.early_stop_limit} epochs without improvement") # <-- THAY ĐỔI
                break
    
    logger.info("--- Training finished ---") # <-- THAY ĐỔI
    plot_training_curves(history, args.save_dir)

if __name__ == "__main__":
    args = get_training_args()
    main(args)