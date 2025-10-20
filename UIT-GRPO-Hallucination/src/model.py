# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, TaskType
from typing import Any
import logging

from .dataset import LABEL2ID # Import để biết num_labels

logger = logging.getLogger(__name__) # <-- THÊM MỚI

class ImprovedViHalluGRPO(nn.Module):
    def __init__(self, backbone: AutoModel, hidden_size: int, num_labels: int = 3):
        super().__init__()
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_labels)
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

        self.crit = nn.CrossEntropyLoss(reduction='none')
        self._init_weights()

    def _init_weights(self):
        for module in [self.head, self.value_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, input_ids, attention_mask, resp_mask, labels=None):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        last = out.last_hidden_state[-1]

        weights = resp_mask / (resp_mask.sum(1, keepdim=True).clamp(min=1.0))
        weights = weights.unsqueeze(-1)
        pooled = (last * weights).sum(1)

        logits = self.head(pooled)
        values = self.value_head(pooled).squeeze(-1)

        loss = None
        if labels is not None:
            loss = self.crit(logits, labels)

        return logits, loss, values

    def get_policy_logprobs(self, logits, actions):
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        return action_log_probs

def create_reference_model(model_name, lora_config, hidden_size, trust_remote_code=True):
    base = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    base = get_peft_model(base, lora_config)
    ref_model = ImprovedViHalluGRPO(base, hidden_size=hidden_size, num_labels=len(LABEL2ID))
    return ref_model

def build_model_and_tokenizer(args: Any):
    tok = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, trust_remote_code=args.trust_remote_code
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token

    base = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    hidden = base.config.hidden_size

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        task_type=TaskType.FEATURE_EXTRACTION,
        bias="none"
    )
    base = get_peft_model(base, lora_cfg)

    trainable_params = sum(p.numel() for p in base.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in base.parameters())
    logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    model = ImprovedViHalluGRPO(base, hidden_size=hidden, num_labels=len(LABEL2ID)).to(args.device)
    ref_model = create_reference_model(
        args.model_name, lora_cfg, hidden, args.trust_remote_code
    ).to(args.device)
    ref_model.load_state_dict(model.state_dict())

    return model, ref_model, tok