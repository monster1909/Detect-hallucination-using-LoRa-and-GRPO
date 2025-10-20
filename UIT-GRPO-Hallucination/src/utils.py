# src/utils.py
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from .dataset import ID2LABEL # Cần cho confusion matrix

logger = logging.getLogger(__name__)

def plot_training_curves(history, save_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics_to_plot = ['loss', 'accuracy', 'macro_f1', 'reward', 'kl_div']

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes.flatten()[idx]
        if f'train_{metric}' in history:
            ax.plot(history[f'train_{metric}'], label='Train', marker='o')
        if f'val_{metric}' in history:
            ax.plot(history[f'val_{metric}'], label='Val', marker='s')

        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Xóa trục cuối nếu không dùng
    if len(metrics_to_plot) < len(axes.flatten()):
        axes.flatten()[-1].axis('off')

    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=100)
    plt.show()
    logger.info(f"Training curves saved to {save_path}")

def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(ID2LABEL.values()),
                yticklabels=list(ID2LABEL.values()))
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.show()
    logger.info(f"Confusion matrix saved to {save_path}")

def save_model_for_inference(model, tokenizer, save_path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': tokenizer.name_or_path,
    }, save_path)
    logger.info(f"Inference model saved to {save_path}")