# src/dataset.py
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
from typing import List, Dict, Any

# --- Label Definitions ---
LABEL2ID = {"NO": 0, "INTRINSIC": 1, "EXTRINSIC": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def _normalize_label(x: str) -> str:
    s = ("" if x is None else str(x)).strip().lower()
    if s in {"no", "none", "0", "negative", "non-hallu", "non_hallu", "correct"}:
        return "NO"
    if s in {"intrinsic", "intra", "1", "internal"}:
        return "INTRINSIC"
    if s in {"extrinsic", "extra", "2", "external"}:
        return "EXTRINSIC"
    raise ValueError(f"Invalid label: {x}")

# --- Dataset Class ---
class ViHalluSet(Dataset):
    REQUIRED_COLS = ["id", "context", "prompt", "response", "label"]

    def __init__(self, data):
        self.samples = []
        if isinstance(data, pd.DataFrame):
            df = data.fillna("")
            self._load_from_df(df)

    def _load_from_df(self, df: pd.DataFrame):
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        for _, row in df.iterrows():
            self.samples.append({
                "id": str(row["id"]).strip(),
                "context": str(row["context"]).strip(),
                "prompt": str(row["prompt"]).strip(),
                "response": str(row["response"]).strip(),
                "label": _normalize_label(row["label"])
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]

# --- Collator Class ---
class SmartCollator:
    def __init__(self, tok: AutoTokenizer, max_len: int, cfg: Any):
        self.tok = tok
        self.max_len = max_len
        self.cfg = cfg
        self.instruction_template = """Bạn là một chuyên gia nhận diện ảo giác trong câu trả lời so với ngữ cảnh và câu hỏi, nhiệm vụ phân loại câu trả lời theo 3 loại:
Nhận diện "No":
- Response hoàn toàn nhất quán và đúng sự thật với thông tin được cung cấp trong context.
- Response không chứa bất kỳ thông tin nào sai lệch hoặc không thể suy luận trực tiếp từ context.
- Response trả lời đúng dựa trên context.

Nhận diện "Intrinsic":
- Mâu thuẫn trực tiếp trong phản hồi
- Bóp méo thông tin đã được cung cấp rõ ràng trong context
- Chứa thực thể hoặc khái niệm trong context nhưng thông tin chúng bị thay đổi, sai lệch
- Response sai lệch nhưng nghe có vẻ khá hợp lý (plausible) trong ngữ cảnh đó

Nhận diện "Extrinsic":
- Response bổ sung thông tin không có trong context
- Thông tin bổ sung không thể suy luận được từ context
- Thông tin bổ sung có thể đúng trong thế giới thực nhưng nó không được cung cấp trong context
"""

    def _smart_truncate(self, context: str, response: str, max_context_len: int, max_response_len: int):
        ctx_tokens = self.tok(context, add_special_tokens=False, truncation=False)["input_ids"]
        rsp_tokens = self.tok(response, add_special_tokens=False, truncation=False)["input_ids"]

        if len(ctx_tokens) <= max_context_len and len(rsp_tokens) <= max_response_len:
            return context, response

        if len(ctx_tokens) > max_context_len:
            keep_start = max_context_len * 2 // 3
            keep_end = max_context_len - keep_start
            ctx_start = self.tok.decode(ctx_tokens[:keep_start], skip_special_tokens=True)
            ctx_end = self.tok.decode(ctx_tokens[-keep_end:], skip_special_tokens=True)
            context = ctx_start + " [...] " + ctx_end

        if len(rsp_tokens) > max_response_len:
            keep_start = min(len(rsp_tokens), max_response_len * 3 // 4)
            keep_end = max_response_len - keep_start
            if keep_end > 0 and len(rsp_tokens) > keep_start:
                rsp_start = self.tok.decode(rsp_tokens[:keep_start], skip_special_tokens=True)
                rsp_end = self.tok.decode(rsp_tokens[-keep_end:], skip_special_tokens=True)
                response = rsp_start + " [...] " + rsp_end
            else:
                response = self.tok.decode(rsp_tokens[:max_response_len], skip_special_tokens=True)

        return context, response

    def __call__(self, batch: List[Dict[str, Any]]):
        input_ids, attention_mask, resp_mask, labels, metas = [], [], [], [], []

        for smp in batch:
            ctx = smp.get("context", "")
            prm = smp.get("prompt", "")
            rsp = smp.get("response", "")
            lbl = smp.get("label", "NO").upper()
            lbl_id = LABEL2ID.get(lbl, 0)

            instruction_len = len(self.tok(self.instruction_template, add_special_tokens=False)["input_ids"])
            prompt_len = len(self.tok(prm, add_special_tokens=False)["input_ids"])
            overhead = instruction_len + prompt_len + 50

            available_len = self.max_len - overhead
            max_context_len = int(available_len * self.cfg.context_ratio)
            max_response_len = available_len - max_context_len

            ctx, rsp = self._smart_truncate(ctx, rsp, max_context_len, max_response_len)

            prefix = f"""{self.instruction_template}

Context Information:
{ctx}

User Query:
{prm}

AI Response to Analyze:
"""

            enc_pre = self.tok(prefix, add_special_tokens=True, truncation=True,
                               max_length=self.max_len - self.cfg.min_response_len)
            enc_rsp = self.tok(rsp, add_special_tokens=False, truncation=True,
                               max_length=self.max_len - len(enc_pre["input_ids"]))

            ids = enc_pre["input_ids"] + enc_rsp["input_ids"]
            ids = ids[:self.max_len]
            attn = [1] * len(ids)

            pre_len = len(enc_pre["input_ids"])
            resp_len = len(ids) - pre_len
            rmask = [0] * pre_len + [1] * resp_len

            pad_id = self.tok.pad_token_id or self.tok.eos_token_id or 0
            if len(ids) < self.max_len:
                pad_n = self.max_len - len(ids)
                ids += [pad_id] * pad_n
                attn += [0] * pad_n
                rmask += [0] * pad_n

            input_ids.append(torch.tensor(ids, dtype=torch.long))
            attention_mask.append(torch.tensor(attn, dtype=torch.long))
            resp_mask.append(torch.tensor(rmask, dtype=torch.float))
            labels.append(lbl_id)
            metas.append({"id": smp.get("id", None), "label_text": lbl})

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "resp_mask": torch.stack(resp_mask),
            "labels": torch.tensor(labels, dtype=torch.long),
            "meta": metas
        }

# --- Sampler ---
def create_weighted_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
    return sampler