# Detect Hallucination using LoRA and GRPO  
**UIT 2025 Data Science Competition**

This project presents a solution for **hallucination detection in Large Language Models (LLMs)** for the **Vietnamese language**. The goal is to classify an AI-generated response into one of three labels based on a given **context** and **prompt**:

- **NO**: Accurate and faithful to the context (no hallucination).  
- **INTRINSIC**: Contains internal contradictions or distortions of information explicitly provided in the context.  
- **EXTRINSIC**: Includes external, fabricated, or unsubstantiated information not derivable from the context.  

The solution leverages efficient fine-tuning with **LoRA** and reinforcement learning via **GRPO (Generalized Reward-Policy Optimization)** to align the model with expert-level reasoning on hallucinations.

---

## Model Checkpoint

Download the fine-tuned model here:  
🔗 [https://huggingface.co/TTTam/UIT_2025](https://huggingface.co/TTTam/UIT_2025)

The checkpoint includes the LoRA-adapted weights, compatible with the base model [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B).

---

## Our Approach

We combine a strong base model with **efficient fine-tuning** and **reinforcement learning alignment** to achieve robust hallucination detection. The system is designed to handle Vietnamese text effectively, focusing on semantic consistency between context, prompt, and response.

### 1. Base Model & Fine-tuning Strategy

- **LLM Backbone**: [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) – a high-capacity model with strong semantic understanding, multilingual capabilities, and a hidden size of 2048. It excels at processing long sequences (up to 1024 + 512 tokens in our setup).
- **Efficient Adaptation**: We use **LoRA (Low-Rank Adaptation)** to fine-tune the model.
  - **Hyperparameters**: Rank (r=8), Alpha (16), Dropout (0.1), Target modules: ["q_proj", "k_proj", "v_proj", "o_proj"].
  - Only **~0.24%** of parameters are updated (trainable params: ~1.2M out of ~500M total).
  - Reduces memory usage and speeds up training significantly (e.g., batch size 4 with accumulation steps 8 on a single GPU).
  - Enables task-specific feature extraction from Vietnamese text without full model retraining.
- **Optimizer & Scheduler**: AdamW (LR=3e-5, Weight Decay=0.01) with linear warmup (15% of total steps) over 20 epochs.

### 2. Core Pipeline

Our method operates in **three key stages**:

#### **Step 1: Feature Extraction**
- **Input Formatting**: The input is constructed as a structured prompt:

  [Instruction Template]
  Context Information:  [Truncated Context]
  User Query:  [Prompt]
  AI Response to Analyze:  [Truncated Response]
- The instruction template provides detailed guidelines for classifying "NO", "INTRINSIC", or "EXTRINSIC" (see code in `SmartCollator` for full template).
- Smart truncation: Context limited to ~60% of max length (1024+512 tokens), response to the remainder, with ellipsis for overflow to preserve key parts.
- The **LoRA-adapted Qwen3-4B** processes the full sequence and outputs **hidden-state representations** for all tokens (last hidden state).

#### **Step 2: Response-Aware Pooling**
- Isolate hidden states corresponding to the **response segment** using a response mask.
- Apply **weighted average pooling** to aggregate them into a **single dense feature vector** (size: hidden_size=2048).
- Weights: Normalized response mask to emphasize response tokens.
- This vector encapsulates the semantic essence of the AI's answer in context, ignoring irrelevant prefix tokens.

#### **Step 3: RL-Aligned Classification**
- The pooled vector is fed into a **lightweight classification head** (MLP with layers: Linear(2048→2048) + LayerNorm + ReLU + Dropout(0.3) + Linear(2048→1024) + ReLU + Dropout(0.2) + Linear(1024→3)).
- A separate **value head** (Linear(2048→1024) + ReLU + Linear(1024→1)) estimates rewards for GRPO.
- **Training Losses**:
- **Cross-Entropy (CE)**: For supervised label prediction (weighted by class imbalance).
- **Policy Gradient (PG)**: From GRPO, with advantages computed from rewards minus values.
- **KL Penalty**: Clamps divergence from a reference model (beta=0.08).
- **Value Loss**: Smooth L1 between predicted values and rewards.
- **Warmup Phase**: First 5 epochs focus on CE; gradual introduction of PG/KL/Value over next 2 epochs.
- **Reward Function** (in `EnhancedRewardCalculator`):
- Combines metric-based rewards (class-weighted bonuses for correct predictions, penalties for errors) and F1-oriented rewards.
- Alpha=0.3 (metric), Beta=0.7 (F1); scaled by 3.0.
- Encourages high recall on hallucination classes (INTRINSIC/EXTRINSIC get higher weights).
- **Reference Model**: A frozen copy of the initial model to compute KL and log-prob ratios.
- **Data Handling**: WeightedRandomSampler for class imbalance; early stopping after 3 epochs without Macro F1 improvement.

---

## Why This Works

| Component              | Benefit |
|------------------------|--------|
| **Qwen3-4B + LoRA**    | Strong multilingual understanding + efficient adaptation for Vietnamese-specific nuances. |
| **Smart Collator & Truncation** | Handles long contexts/prompts without losing critical information; custom instruction boosts reasoning. |
| **Response Pooling**   | Focuses on answer semantics, ignores noise from context/prompt. |
| **GRPO Alignment**     | Enables nuanced, human-like judgment on hallucinations by optimizing for a custom reward that targets Macro F1. |
| **Reward Design**      | Balances accuracy with competition metrics; penalizes false positives/negatives on rare classes. |

The approach achieves strong generalization, as evidenced by training curves and confusion matrices (plotted during training).

---
**Team**: US-TTP
**Team Members**:  
  1. Tống Trọng Tâm
  2. Dương Thanh Triều
  3. Bùi Hồng Phúc

For questions or collaborations, contact [tongtrongtam1909@gmail.com](tongtrongtam1909@gmail.com).```
