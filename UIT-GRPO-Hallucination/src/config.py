# src/config.py
import argparse
import torch

def get_training_args():
    parser = argparse.ArgumentParser(description="Train ViHallu GRPO Model")

    # Model & Tokenizer
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-4B", help="Base model name from Hugging Face")
    parser.add_argument('--trust_remote_code', action='store_true', default=True, help="Trust remote code for tokenizer/model")
    parser.add_argument('--max_len', type=int, default=1024 + 512, help="Max sequence length")

    # LoRA Config
    parser.add_argument('--lora_r', type=int, default=8, help="LoRA r")
    parser.add_argument('--lora_alpha', type=int, default=16, help="LoRA alpha")
    parser.add_argument('--lora_dropout', type=float, default=0.1, help="LoRA dropout")
    parser.add_argument('--target_modules', nargs='+', default=["q_proj", "k_proj", "v_proj", "o_proj"], help="LoRA target modules")

    # Training Params
    parser.add_argument('--lr', type=float, default=3e-5, help="Learning rate")
    parser.add_argument('--wd', type=float, default=0.01, help="Weight decay")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=4, help="Training batch size")
    parser.add_argument('--accum_steps', type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument('--warmup_ratio', type=float, default=0.15, help="Warmup ratio")

    # GRPO Params
    parser.add_argument('--grpo_beta', type=float, default=0.08, help="GRPO beta")
    parser.add_argument('--reward_scale', type=float, default=3.0, help="Reward scale")
    parser.add_argument('--warmup_epochs', type=int, default=5, help="Epochs for CE warmup before RL")

    # Data Params
    parser.add_argument('--context_ratio', type=float, default=0.6, help="Ratio of context length to total available length")
    parser.add_argument('--min_response_len', type=int, default=100, help="Minimum reserved length for response")
    parser.add_argument('--train_file', type=str, default="vihallu-train.csv", help="Path to training CSV")
    parser.add_argument('--val_split', type=float, default=0.1, help="Validation split size")

    # System
    parser.add_argument('--workers', type=int, default=8, help="Number of dataloader workers")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--save_dir', type=str, default="outputs", help="Directory to save checkpoints and plots")
    parser.add_argument('--early_stop_limit', type=int, default=3, help="Early stopping patience")


    return parser.parse_args()

def get_inference_args():
    parser = argparse.ArgumentParser(description="Inference with ViHallu GRPO Model")

    # Model & Checkpoint
    parser.add_argument('--repo_id', type=str, default="TTTam/UIT_2025", help="Hugging Face Repo ID or local path to checkpoint")
    parser.add_argument('--checkpoint_filename', type=str, default="best_model.pt", help="Filename of the checkpoint in the repo")
    parser.add_argument('--max_len', type=int, default=1024 + 512, help="Max sequence length")

    # LoRA Config (must match training)
    parser.add_argument('--lora_r', type=int, default=8, help="LoRA r")
    parser.add_argument('--lora_alpha', type=int, default=16, help="LoRA alpha")
    parser.add_argument('--target_modules', nargs='+', default=["q_proj", "k_proj", "v_proj", "o_proj"], help="LoRA target modules")

    # Data
    parser.add_argument('--test_file', type=str, default="vihallu-private-test.csv", help="Path to test CSV")
    parser.add_argument('--output_file', type=str, default="submit.csv", help="Path to save submission file")
    
    # System
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument('--context_ratio', type=float, default=0.6, help="Ratio of context length (dummy for collator)")
    parser.add_argument('--min_response_len', type=int, default=100, help="Min response length (dummy for collator)")


    return parser.parse_args()

def get_interactive_args():
    parser = argparse.ArgumentParser(description="Interactive Inference with ViHallu GRPO Model")

    # Model & Checkpoint
    parser.add_argument('--repo_id', type=str, default="TTTam/UIT_2025", help="Hugging Face Repo ID or local path to checkpoint")
    parser.add_argument('--checkpoint_filename', type=str, default="best_model.pt", help="Filename of the checkpoint in the repo")
    parser.add_argument('--max_len', type=int, default=1024 + 512, help="Max sequence length")

    # LoRA Config (must match training)
    parser.add_argument('--lora_r', type=int, default=8, help="LoRA r")
    parser.add_argument('--lora_alpha', type=int, default=16, help="LoRA alpha")
    parser.add_argument('--target_modules', nargs='+', default=["q_proj", "k_proj", "v_proj", "o_proj"], help="LoRA target modules")
    
    # System
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    
    # Dummy args for SmartCollator (lấy từ CFG cũ)
    parser.add_argument('--context_ratio', type=float, default=0.6, help="Ratio of context length (dummy for collator)")
    parser.add_argument('--min_response_len', type=int, default=100, help="Min response length (dummy for collator)")

    return parser.parse_args()