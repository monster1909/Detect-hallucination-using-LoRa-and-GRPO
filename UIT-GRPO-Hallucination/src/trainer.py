# src/trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from typing import Any

from .dataset import ID2LABEL # Cần để report
import logging

logger = logging.getLogger(__name__)

class EnhancedRewardCalculator:
    def __init__(self, config: Any):
        self.config = config
        self.class_weights = torch.tensor([1.0, 2.0, 3.0]) # Default

    def set_class_weights(self, train_labels):
        """Tính class weights dựa trên phân bố labels trong training set"""
        class_counts = np.bincount(train_labels, minlength=3)
        total = class_counts.sum()
        self.class_weights = torch.tensor([
            total / (3 * (count + 1)) for count in class_counts
        ], dtype=torch.float32)
        logger.info(f"Updated class weights: {self.class_weights}")

    def calculate_combined_reward(self, predictions, labels, logits=None,
                                   alpha=0.5, beta=0.5):
        if logits is not None:
            probs = F.softmax(logits, dim=-1)
            confidences = probs.max(dim=-1)[0]
        else:
            confidences = torch.ones(len(predictions))

        r_metric = torch.zeros(len(predictions), dtype=torch.float32)
        r_f1 = torch.zeros(len(predictions), dtype=torch.float32)

        for i, (pred, label, conf) in enumerate(zip(predictions, labels, confidences)):
            pred_item = pred.item() if torch.is_tensor(pred) else pred
            label_item = label.item() if torch.is_tensor(label) else label
            conf_item = conf.item() if torch.is_tensor(conf) else conf

            if pred_item == label_item:
                base_reward = self.class_weights[label_item].item()
                confidence_bonus = 0.3 * conf_item
                reward = base_reward + confidence_bonus
                if label_item > 0:
                    reward *= 1.5
                r_metric[i] = reward
            else:
                base_penalty = -1.0
                confidence_penalty = -0.5 * conf_item
                if label_item > 0:
                    base_penalty *= 2.0
                if pred_item > 0 and label_item == 0:
                    base_penalty *= 1.5
                r_metric[i] = base_penalty + confidence_penalty

            if pred_item == label_item and pred_item > 0:
                r_f1[i] = 3.0
            elif pred_item == label_item and pred_item == 0:
                r_f1[i] = 1.0
            elif label_item > 0 and pred_item != label_item:
                r_f1[i] = -3.0
            elif pred_item > 0 and label_item == 0:
                r_f1[i] = -2.0
            else:
                r_f1[i] = -1.0

        combined = alpha * r_metric + beta * r_f1
        combined = combined * self.config.reward_scale

        return combined

class AdvancedGRPOTrainer:
    def __init__(self, model, ref_model, config: Any):
        self.model = model
        self.ref_model = ref_model
        self.config = config
        self.reward_calc = EnhancedRewardCalculator(config)

        self.warmup_epochs = config.warmup_epochs
        self.current_epoch = 0

        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def get_loss_weights(self):
        if self.current_epoch <= self.warmup_epochs:
            return {"ce": 1.0, "pg": 0.0, "kl": 0.0, "value": 0.0}
        elif self.current_epoch <= self.warmup_epochs + 2:
            progress = (self.current_epoch - self.warmup_epochs) / 3.0
            return {
                "ce": 0.8 - 0.3 * progress,
                "pg": 0.4 * progress,
                "kl": 0.03 * progress,
                "value": 0.2 * progress
            }
        else:
            return {"ce": 0.5, "pg": 0.4, "kl": 0.03, "value": 0.2}

    def compute_advantages(self, rewards, values):
        advantages = rewards.detach() - values.detach()
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-6
        advantages = (advantages - adv_mean) / adv_std
        advantages = torch.clamp(advantages, -5.0, 5.0)
        return advantages

    def compute_kl_penalty(self, logits, ref_logits):
        p = F.softmax(logits, dim=-1)
        ref_p = F.softmax(ref_logits.detach(), dim=-1)
        kl = F.kl_div(p.log(), ref_p, reduction='none').sum(-1)
        kl = torch.clamp(kl, max=5.0)
        return kl

    def compute_policy_gradient_loss(self, logits, ref_logits, actions, advantages):
        current_log_probs = F.log_softmax(logits, dim=-1)
        current_action_log_probs = current_log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            ref_action_log_probs = ref_log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        ratio = torch.exp(current_action_log_probs - ref_action_log_probs)
        ratio = torch.clamp(ratio, 0.1, 10.0)

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 0.8, 1.2)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        return pg_loss

    def grpo_step(self, batch, optimizer, scheduler, accumulation_step=0):
        self.model.train()
        input_ids = batch["input_ids"].to(self.config.device)
        attention_mask = batch["attention_mask"].to(self.config.device)
        resp_mask = batch["resp_mask"].to(self.config.device)
        labels = batch["labels"].to(self.config.device)

        weights = self.get_loss_weights()

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits, ce_loss, values = self.model(input_ids, attention_mask, resp_mask, labels)
            with torch.no_grad():
                ref_logits, _, _ = self.ref_model(input_ids, attention_mask, resp_mask)

        probs = F.softmax(logits / 1.15, dim=-1)
        if self.model.training and self.current_epoch > self.warmup_epochs:
            actions = torch.multinomial(probs, 1).squeeze(-1)
        else:
            actions = logits.argmax(-1)

        rewards = self.reward_calc.calculate_combined_reward(
                predictions=actions.cpu(),
                labels=labels.cpu(),
                logits=logits.cpu(),
                alpha=0.3,  # metric
                beta=0.7    # F1
            ).to(self.config.device)

        total_loss = 0
        loss_info = {}

        ce_loss_mean = ce_loss.mean()
        total_loss += weights["ce"] * ce_loss_mean
        loss_info["ce_loss"] = ce_loss_mean.item()

        if weights["pg"] > 0:
            advantages = self.compute_advantages(rewards, values)
            pg_loss = self.compute_policy_gradient_loss(logits, ref_logits, actions, advantages)
            total_loss += weights["pg"] * pg_loss
            loss_info["pg_loss"] = pg_loss.item()

            if weights["kl"] > 0:
                kl_penalty = self.compute_kl_penalty(logits, ref_logits)
                kl_loss = weights["kl"] * kl_penalty.mean()
                total_loss += kl_loss
                loss_info["kl_loss"] = kl_loss.item()
                loss_info["kl_div"] = kl_penalty.mean().item()

            if weights["value"] > 0:
                value_targets = rewards.detach()
                value_loss = F.smooth_l1_loss(values, value_targets)
                total_loss += weights["value"] * value_loss
                loss_info["value_loss"] = value_loss.item()

            loss_info["reward"] = rewards.mean().item()
        else:
            loss_info.update({
                "pg_loss": 0.0, "kl_loss": 0.0, "value_loss": 0.0,
                "reward": 0.0, "kl_div": 0.0
            })

        total_loss = total_loss / self.config.accum_steps
        total_loss.backward()

        if (accumulation_step + 1) % self.config.accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        preds = logits.argmax(-1)
        acc = (preds == labels).float().mean()

        loss_info.update({
            "loss": total_loss.item() * self.config.accum_steps,
            "accuracy": acc.item()
        })
        return loss_info

# --- Epoch Loops ---
def train_grpo_epoch(model, ref_model, loader, optimizer, scheduler, config, epoch):
    trainer = AdvancedGRPOTrainer(model, ref_model, config)
    trainer.set_epoch(epoch)

    all_labels = []
    for batch in loader:
        all_labels.extend(batch["labels"].cpu().numpy())
    trainer.reward_calc.set_class_weights(all_labels)

    metrics = defaultdict(list)
    model.train()
    stage = 'Warmup' if epoch <= config.warmup_epochs else 'RL'
    pbar = tqdm(loader, desc=f"Epoch {epoch} - GRPO Training (Stage: {stage})")

    for idx, batch in enumerate(pbar):
        step_metrics = trainer.grpo_step(batch, optimizer, scheduler, idx)
        for k, v in step_metrics.items():
            metrics[k].append(v)

        if len(metrics["loss"]) > 0:
            smoothed_metrics = {
                "loss": np.mean(metrics["loss"][-100:]),
                "acc": np.mean(metrics["accuracy"][-100:]),
                "reward": np.mean(metrics["reward"][-100:]),
                "ce": np.mean(metrics["ce_loss"][-100:])
            }
            pbar.set_postfix(smoothed_metrics)

    return {k: np.mean(v) for k, v in metrics.items()}

@torch.no_grad()
def evaluate_grpo(model, loader, config, desc="Evaluating"):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0
    num_batches = 0

    for batch in tqdm(loader, desc=desc):
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        resp_mask = batch["resp_mask"].to(config.device)
        labels = batch["labels"].to(config.device)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits, loss, _ = model(input_ids, attention_mask, resp_mask, labels)

        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(-1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().numpy())
        total_loss += loss.mean().item()
        num_batches += 1

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    per_class_f1 = f1_score(all_labels, all_preds, average=None)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=[ID2LABEL[i] for i in range(3)],
        digits=4,
        zero_division=0
    )
    return {
        "loss": total_loss / max(1, num_batches),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": {ID2LABEL[i]: f1 for i, f1 in enumerate(per_class_f1)},
        "confusion_matrix": cm,
        "report": report,
        "predictions": all_preds,
        "probabilities": all_probs
    }