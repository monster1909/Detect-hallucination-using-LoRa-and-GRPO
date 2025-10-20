# src/predictor.py
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, Any
from huggingface_hub import hf_hub_download
import logging
import os
import logging

from .dataset import SmartCollator, LABEL2ID, ID2LABEL
from .model import ImprovedViHalluGRPO

logger = logging.getLogger(__name__)
class InferenceModel:
    def __init__(self, repo_id_or_path: str, checkpoint_filename: str, lora_config: Dict, args: Any):
        self.device = torch.device(args.device)
        self.lora_config = lora_config
        self.collator = None
        self.args = args # Dùng để truyền max_len, context_ratio... cho collator
        self.load_model(repo_id_or_path, checkpoint_filename)

    def load_model(self, repo_id_or_path: str, checkpoint_filename: str):
        # Kiểm tra xem đây là đường dẫn local hay repo_id
        if os.path.exists(repo_id_or_path):
            checkpoint_path = repo_id_or_path
            logger.info(f"Loading local checkpoint: {checkpoint_path}")
        else:
            logger.info(f"Downloading checkpoint '{checkpoint_filename}' from repo '{repo_id_or_path}'...")
            checkpoint_path = hf_hub_download(
                repo_id=repo_id_or_path,
                filename=checkpoint_filename
            )
            logger.info(f"Checkpoint downloaded to: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_name = checkpoint['model_name']
        logger.info(f"Loading base model '{model_name}' for inference...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModel.from_pretrained(model_name)
        hidden_size = base_model.config.hidden_size
        logger.info("Base model loaded. Applying LoRA...")
        
        peft_config = LoraConfig(
            r=self.lora_config['r'],
            lora_alpha=self.lora_config['lora_alpha'],
            target_modules=self.lora_config['target_modules'],
            task_type=TaskType.FEATURE_EXTRACTION,
            bias="none"
        )
        peft_model = get_peft_model(base_model, peft_config)

        self.model = ImprovedViHalluGRPO(
            peft_model,
            hidden_size=hidden_size,
            num_labels=len(LABEL2ID)
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        logger.info("Model state dict loaded. Model is in eval mode.") 
        # Truyền args (chứa max_len,...) cho collator
        self.collator = SmartCollator(self.tokenizer, self.args.max_len, self.args)

    @torch.no_grad()
    def predict(self, context: str, prompt: str, response: str):
        sample = [{"context": context, "prompt": prompt, "response": response, "label": "NO"}]
        batch = self.collator(sample)

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        resp_mask = batch["resp_mask"].to(self.device)

        logits, _, _ = self.model(input_ids, attention_mask, resp_mask)
        probs = F.softmax(logits, dim=-1).flatten()

        pred_id = torch.argmax(probs).item()
        prediction = ID2LABEL[pred_id]
        confidence = probs[pred_id].item()

        return {"prediction": prediction, "confidence": confidence}