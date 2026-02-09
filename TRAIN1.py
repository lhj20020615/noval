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
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
import deepspeed
import json

# ================= GPU =================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 配置 =================
BASE_RESULT_DIR = "/home/lihaojia/1101/最终预训练/llm2"
ABLATION_DIR = os.path.join(BASE_RESULT_DIR, "大语言消融")
os.makedirs(ABLATION_DIR, exist_ok=True)

TRAIN_FILE = "/home/lihaojia/1101/130data/train.xlsx"
VAL_FILE   = "/home/lihaojia/1101/130data/VAL.xlsx"

EPOCHS = 15
BATCH_SIZE = 8
ACCUM_STEPS = 2
LR = 1e-6
WEIGHT_DECAY = 0.02
SEED = 12
N_INDEPENDENT = 1  # 消融实验只跑一轮

MODEL_LIST = [
    ("", "Qwen3-4B-Thinking"),
    ##("", "Meta-Llama-3.1-8B")
]

# 消融实验配置
ABLATION_LIST = [
   # "仅文本",
    #"消融所有波谷",
    #"消融所有波峰",
    #"消融N400",
    #"消融LPC",
    #"只保留N400_peak",
   # "只保留N400_trough",
    #"只保留LPC_peak",
   # "只保留LPC_trough",
    "完整ERP"
]

# ================= 随机种子 =================
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# ================= 数据处理 =================
ERP_COLS = ["N400_peak", "N400_trough", "LPC_peak", "LPC_trough"]

def load_excel_data(file_path, ablation=None, debug=False):
    df = pd.read_excel(file_path)
    data_list = []

    for idx, row in df.iterrows():
        erp = row[ERP_COLS].astype(float).values

        # ===== 消融逻辑 =====
        #if ablation == "仅文本":
          #  erp[:] = 0
      #  elif ablation == "消融所有波谷":
            erp[[1,3]] = 0
     #   elif ablation == "消融所有波峰":
          ####elif ablation == "消融LPC":
            erp[[2,3]] = 0
      #  elif ablation == "只保留N400_peak":
            erp[[1,2,3]] = 0
      #  elif ablation == "只保留N400_trough":
            erp[[0,2,3]] = 0
       # elif ablation == "只保留LPC_peak":
            erp[[0,1,3]] = 0
       # elif ablation == "只保留LPC_trough":
            erp[[0,1,2]] = 0
        elif ablation == "完整ERP":
            pass
            # debug 可用 -999 测试消融效果
        # erp[erp == 0] = -999.0

        erp_tensor = torch.tensor(erp, dtype=torch.float32)
        tag = int(row["Score"]) - 1
        data_list.append({
            "guid": idx,
            "text": row["Sentence"],
            "erp": erp_tensor,
            "tag": tag
        })

    if debug:
        print(f">>> Debug: ablation = {ablation}")
        for i, d in enumerate(data_list[:3]):
            print(f" sample {i} erp:", d['erp'].numpy())

    return data_list

# ================= DataLoader =================
def collate_fn_latefusion(data_list, tokenizer, device):
    texts = [d["text"] for d in data_list]
    erps  = torch.stack([d["erp"] for d in data_list]).to(device)
    tags  = torch.tensor([d["tag"] for d in data_list], dtype=torch.long).to(device)

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    return tags, erps, encoded

def get_data_loader(data_list, tokenizer, batch_size, device, shuffle=True):
    return DataLoader(
        data_list,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: collate_fn_latefusion(x, tokenizer, device)
    )

# ================= 模型 =================
class LateFusionClassifier(nn.Module):
    def __init__(self, model_path):
        super().__init__()

        self.config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )

        backbone = AutoModel.from_pretrained(
            model_path,
            config=self.config,
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True
        )

        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            task_type="SEQ_CLS",
            target_modules=[
                "q_proj","k_proj","v_proj","o_proj",
                "gate_proj","up_proj","down_proj"
            ]
        )

        self.backbone = get_peft_model(backbone, lora_config)

        hidden_size = self.config.hidden_size
        self.text_pool = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.2)
        )
        self.erp_fc = nn.Linear(4, hidden_size)
        self.classifier = nn.Linear(hidden_size * 2, 3)

    def forward(self, input_ids=None, attention_mask=None, erp_feats=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.text_pool(pooled)

        erp_feats = erp_feats.to(self.erp_fc.weight.dtype)
        erp_emb = self.erp_fc(erp_feats)

        fusion = torch.cat([pooled, erp_emb], dim=1)
        logits = self.classifier(fusion)
        return logits

# ================= 指标 =================
def calc_metrics(target, pred):
    pred_label = np.argmax(pred, axis=1)
    return (
        accuracy_score(target, pred_label),
        precision_score(target, pred_label, average="macro", zero_division=0),
        recall_score(target, pred_label, average="macro", zero_division=0),
        f1_score(target, pred_label, average="macro", zero_division=0)
    )

# ================= 消融实验训练 =================
def run_ablation_experiment(model_name, model_path, ablation_name):
    print(f"\n===== {model_name} | 消融实验: {ablation_name} =====")

    ablation_result_dir = os.path.join(ABLATION_DIR, ablation_name)
    os.makedirs(ablation_result_dir, exist_ok=True)

    all_results_file = os.path.join(ablation_result_dir, f"{model_name}_all_rounds.csv")
    if os.path.exists(all_results_file):
        print(f">>> {ablation_name} 已经完成，跳过...")
        return

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if not deepspeed.comm.is_initialized():
        deepspeed.init_distributed(dist_backend="nccl")

    # ====== 加 debug=True 验证消融 ======
    train_data = load_excel_data(TRAIN_FILE, ablation=ablation_name, debug=True)
    val_data   = load_excel_data(VAL_FILE, ablation=ablation_name, debug=True)

    train_loader = get_data_loader(train_data, tokenizer, BATCH_SIZE, DEVICE)
    val_loader   = get_data_loader(val_data, tokenizer, BATCH_SIZE, DEVICE, shuffle=False)

    exp_id = 1
    print(f"\n===== Round {exp_id} =====")
    model = LateFusionClassifier(model_path).to(DEVICE)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        model_parameters=trainable_params,
        config={
            "train_batch_size": BATCH_SIZE * ACCUM_STEPS,
            "gradient_accumulation_steps": ACCUM_STEPS,
            "bf16": {"enabled": True},
            "zero_optimization": {"stage": 0}
        },
        dist_init_required=False
    )

    criterion = nn.CrossEntropyLoss()
    best_val_acc = -1
    best_epoch_result = None
    round_results = []

    for epoch in range(1, EPOCHS + 1):
        model_engine.train()
        y_true, y_pred = [], []

        for step, (tags, erp_feats, batch) in enumerate(train_loader, start=1):
            outputs = model_engine(**batch, erp_feats=erp_feats)
            loss = criterion(outputs, tags)
            model_engine.backward(loss)
            model_engine.step()

            y_true.extend(tags.cpu().numpy())
            y_pred.extend(outputs.detach().float().cpu().numpy())

        acc, pre, rec, f1 = calc_metrics(np.array(y_true), np.array(y_pred))

        model_engine.eval()
        y_true_val, y_pred_val = [], []
        with torch.no_grad():
            for tags, erp_feats, batch in val_loader:
                outputs = model_engine(**batch, erp_feats=erp_feats)
                y_true_val.extend(tags.cpu().numpy())
                y_pred_val.extend(outputs.detach().float().cpu().numpy())

        acc_v, pre_v, rec_v, f1_v = calc_metrics(np.array(y_true_val), np.array(y_pred_val))

        print(
            f"[Round {exp_id} | Epoch {epoch:02d}] "
            f"Train -> ACC: {acc*100:.2f}%, PRE: {pre*100:.2f}%, REC: {rec*100:.2f}%, F1: {f1*100:.2f}% | "
            f"Val -> ACC: {acc_v*100:.2f}%, PRE: {pre_v*100:.2f}%, REC: {rec_v*100:.2f}%, F1: {f1_v*100:.2f}%"
        )

        row = {
            "Round": exp_id,
            "Epoch": epoch,
            "Train_ACC": acc*100,
            "Train_PRE": pre*100,
            "Train_REC": rec*100,
            "Train_F1": f1*100,
            "Val_ACC": acc_v*100,
            "Val_PRE": pre_v*100,
            "Val_REC": rec_v*100,
            "Val_F1": f1_v*100
        }
        round_results.append(row)

        if acc_v > best_val_acc:
            best_val_acc = acc_v
            best_epoch_result = row

    print(f">>> Best Round {exp_id} Epoch: {best_epoch_result['Epoch']} Val_ACC: {best_epoch_result['Val_ACC']:.2f}%")

    # 保存 CSV
    df_all = pd.DataFrame(round_results)
    df_all.to_csv(all_results_file, index=False)
    print(f">>> Saved {ablation_name} results to {all_results_file}")

    # ===== 保存 ERP 权重统计 =====
    erp_w = model.erp_fc.weight.detach().cpu().numpy()
    feat_imp = np.mean(np.abs(erp_w), axis=0)
    with open(os.path.join(ablation_result_dir, f"{model_name}_erp_weight_stats.json"), "w") as f:
        json.dump({"feat_mean_abs": feat_imp.tolist()}, f, indent=2)
    print(">>> Saved ERP weight stats:", feat_imp)

    # ================= 手动释放显存 =================
    del model_engine, model, optimizer
    torch.cuda.empty_cache()
    print(f">>> GPU memory cleared after {ablation_name}")

# ================= 主程序 =================
if __name__ == "__main__":
    for model_path, model_name in MODEL_LIST:
        for ablation_name in ABLATION_LIST:
            run_ablation_experiment(model_name, model_path, ablation_name)
