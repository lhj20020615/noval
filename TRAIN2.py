#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.optim import AdamW
import deepspeed

# ================ 设备检测 ================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[Device] {DEVICE}")

# ================ 超参 / 路径 ================
RESULT_DIR = ""
os.makedirs(RESULT_DIR, exist_ok=True)

TRAIN_PATH = ""
VAL_PATH   = ""

BATCH_SIZE = 8
ACCUM_STEPS = 2
LR = 1e-6
WEIGHT_DECAY = 0.05
SEED = 12
EPOCHS_PER_EXP = #
N_INDEPENDENT = #

MODEL_CANDIDATES = {
    "BERT": "",
    "MacBERT": "",
    "RoBERTa": ""
}

for m in MODEL_CANDIDATES.keys():
    os.makedirs(f"{RESULT_DIR}/{m}_最终预训练", exist_ok=True)

BEST_CSV_TEMPLATE = os.path.join(RESULT_DIR, "{}_best.csv")
ALL_CSV_TEMPLATE  = os.path.join(RESULT_DIR, "{}_all_epochs.csv")

# ================ 随机种子 ================
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True

# ================ 数据加载 ================
ERP_COLS = ["N400_peak", "N400_trough", "LPC_peak", "LPC_trough"]

def load_excel_data(path):
    df = pd.read_excel(path)
    data = []
    for _, row in df.iterrows():
        erp_values = row[ERP_COLS].astype(float).values
        erp = torch.tensor(erp_values, dtype=torch.float32)
        data.append({
            "text": row["Sentence"],
            "erp": erp,
            "tag": int(row["Score"]) - 1
        })
    return data

def collate_fn(batch, tokenizer):
    texts = [x["text"] for x in batch]
    erps = torch.stack([x["erp"] for x in batch]).to(DEVICE)
    tags = torch.tensor([x["tag"] for x in batch]).to(DEVICE)
    enc = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    return tags, erps, enc

# ================ 模型定义 ================
class LateFusionClassifier(nn.Module):
    def __init__(self, model_dir, hidden_size):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_dir,
            config=AutoConfig.from_pretrained(model_dir, local_files_only=True),
            local_files_only=True,
            trust_remote_code=True,
            low_cpu_mem_usage=False
        )
        self.text_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0)
        )
        self.erp_proj = nn.Linear(len(ERP_COLS), hidden_size)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.classifier = nn.Linear(hidden_size * 2, 3)

    def forward(self, **enc):
        erp_feats = enc.pop("erp_feats")
        # FP32 前向，关闭 BF16
        out = self.backbone(**enc)
        pooled = self.text_proj(out.last_hidden_state[:, 0, :])
        pooled = torch.cat([pooled, self.erp_proj(erp_feats) * self.alpha], dim=1)
        logits = self.classifier(pooled)
        return logits

# ================ 指标计算 ================
def metrics(y_true, y_pred):
    y_hat = np.argmax(y_pred, axis=1)
    return (
        accuracy_score(y_true, y_hat),
        precision_score(y_true, y_hat, average="macro", zero_division=0),
        recall_score(y_true, y_hat, average="macro", zero_division=0),
        f1_score(y_true, y_hat, average="macro", zero_division=0)
    )

# ================ 训练/验证 ================
def run_stage(train, loader, model_engine, loss_fn):
    model_engine.train(train)
    ys, ps, total_loss = [], [], 0

    with torch.set_grad_enabled(train):
        for tags, erps, enc in loader:
            out = model_engine(**enc, erp_feats=erps)
            loss = loss_fn(out, tags)
            if train:
                model_engine.backward(loss)
                model_engine.step()
            total_loss += loss.item() * len(tags)
            ys.extend(tags.cpu().numpy())
            ps.extend(out.detach().cpu().numpy())

    acc, p, r, f1 = metrics(np.array(ys), np.array(ps))
    return total_loss / len(ys), acc, p, r, f1

# ================ 主程序 ================
if __name__ == "__main__":

    for model_name, model_dir in MODEL_CANDIDATES.items():
        print(f"\n===== Running Model: {model_name} =====")

        BEST_CSV = BEST_CSV_TEMPLATE.format(model_name)
        ALL_CSV  = ALL_CSV_TEMPLATE.format(model_name)

        # 写表头
        with open(BEST_CSV, "w", newline="") as f:
            csv.writer(f).writerow([
                "Model", "Exp_ID", "Best_Epoch",
                "Train_ACC","Train_P","Train_R","Train_F1",
                "Val_ACC","Val_P","Val_R","Val_F1",
                "Alpha","Train_Set","Val_Set"
            ])
        with open(ALL_CSV, "w", newline="") as f:
            csv.writer(f).writerow([
                "Model","Exp_ID","Epoch",
                "Train_ACC","Train_P","Train_R","Train_F1",
                "Val_ACC","Val_P","Val_R","Val_F1",
                "Alpha","Train_Set","Val_Set"
            ])

        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        config = AutoConfig.from_pretrained(model_dir, local_files_only=True)

        train_data = load_excel_data(TRAIN_PATH)
        val_data   = load_excel_data(VAL_PATH)
        train_loader = DataLoader(train_data, BATCH_SIZE, shuffle=True,
                                  collate_fn=lambda x: collate_fn(x, tokenizer))
        val_loader = DataLoader(val_data, BATCH_SIZE, shuffle=False,
                                collate_fn=lambda x: collate_fn(x, tokenizer))

        for exp_id in range(1, N_INDEPENDENT+1):
            print(f"\n--- Experiment {exp_id} ---")
            model = LateFusionClassifier(model_dir, config.hidden_size)
            optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

            # DeepSpeed 初始化，但关闭 bf16
            model_engine, _, _, _ = deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                config={
                    "train_batch_size": BATCH_SIZE*ACCUM_STEPS,
                    "gradient_accumulation_steps": ACCUM_STEPS,
                    "zero_optimization": {"stage": 0},
                    "bf16": {"enabled": False}  # 关闭 bf16
                }
            )

            loss_fn = nn.CrossEntropyLoss()
            best_val_acc, best_epoch, best_record, best_alpha = -1, -1, None, 0
            train_set_id = "train"
            val_set_id   = "val"

            for epoch in range(1, EPOCHS_PER_EXP+1):
                train_metrics = run_stage(True, train_loader, model_engine, loss_fn)
                val_metrics   = run_stage(False, val_loader, model_engine, loss_fn)

                print(f"[Exp {exp_id} | Epoch {epoch}] "
                      f"Train Acc {train_metrics[1]*100:.2f} | Train P {train_metrics[2]*100:.2f} | "
                      f"Train R {train_metrics[3]*100:.2f} | Train F1 {train_metrics[4]*100:.2f}")
                print(f"[Exp {exp_id} | Epoch {epoch}] "
                      f"Val   Acc {val_metrics[1]*100:.2f} | Val   P {val_metrics[2]*100:.2f} | "
                      f"Val   R {val_metrics[3]*100:.2f} | Val   F1 {val_metrics[4]*100:.2f}")

                with open(ALL_CSV, "a", newline="") as f:
                    csv.writer(f).writerow([
                        model_name, exp_id, epoch,
                        train_metrics[1]*100, train_metrics[2]*100, train_metrics[3]*100, train_metrics[4]*100,
                        val_metrics[1]*100, val_metrics[2]*100, val_metrics[3]*100, val_metrics[4]*100,
                        model.alpha.item(), train_set_id, val_set_id
                    ])

                if val_metrics[1] > best_val_acc:
                    best_val_acc = val_metrics[1]
                    best_epoch   = epoch
                    best_record  = {"Train": train_metrics, "Val": val_metrics}
                    best_alpha   = model.alpha.item()

            print(f"[Exp {exp_id} | Best Epoch {best_epoch}] "
                  f"Train Acc {best_record['Train'][1]*100:.2f} | Train P {best_record['Train'][2]*100:.2f} | "
                  f"Train R {best_record['Train'][3]*100:.2f} | Train F1 {best_record['Train'][4]*100:.2f}")
            print(f"[Exp {exp_id} | Best Epoch {best_epoch}] "
                  f"Val   Acc {best_record['Val'][1]*100:.2f} | Val   P {best_record['Val'][2]*100:.2f} | "
                  f"Val   R {best_record['Val'][3]*100:.2f} | Val   F1 {best_record['Val'][4]*100:.2f}")

            with open(BEST_CSV, "a", newline="") as f:
                csv.writer(f).writerow([
                    model_name, exp_id, best_epoch,
                    best_record['Train'][1]*100, best_record['Train'][2]*100, best_record['Train'][3]*100, best_record['Train'][4]*100,
                    best_record['Val'][1]*100, best_record['Val'][2]*100, best_record['Val'][3]*100, best_record['Val'][4]*100,
                    best_alpha, train_set_id, val_set_id
                ])
