import pandas as pd
import torch
import transformers
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import string
import numpy as np

######################################
# Qwen / GPT-style tokenizer do NOT use Ġ
BOW_SYMBOL = "Ġ"   # 保留，不影响 Qwen（mask 会自动为空）
######################################

# ===== Helper functions =====
def blank_target(s: str, start: int, end: int, replacement: str) -> str:
    if start < 0: start = 0
    if end > len(s): end = len(s)
    return s[:start] + replacement + s[end:]

def subtoken_indices_for_char_span(offset_mapping, start_char, end_char):
    idxs = []
    for i, (s, e) in enumerate(offset_mapping):
        if s == 0 and e == 0:
            continue
        if s < end_char and e > start_char:
            idxs.append(i)
    return idxs

def build_pimentel_masks(tokenizer, model, bow_symbol="Ġ"):
    device = next(model.parameters()).device
    V = model.get_output_embeddings().out_features
    vocab = tokenizer.get_vocab()

    bow_ids = [i for tok, i in vocab.items()
               if i < V and tok and tok.startswith(bow_symbol)]
    punct_ids = [i for tok, i in vocab.items()
                 if i < V and tok and tok[0] in string.punctuation]

    eos_id = tokenizer.eos_token_id

    bow_mask   = torch.zeros(V, device=device)
    punct_mask = torch.zeros(V, device=device)
    eos_mask   = torch.zeros(V, device=device)

    if bow_ids:
        bow_mask[torch.tensor(bow_ids, device=device)] = 1.0
    if punct_ids:
        punct_mask[torch.tensor(punct_ids, device=device)] = 1.0
    if eos_id is not None and eos_id < V:
        eos_mask[eos_id] = 1.0
        punct_mask[eos_id] = 0.0

    useless_mask = torch.zeros(V, device=device)
    if tokenizer.vocab_size < V:
        useless_mask[tokenizer.vocab_size:] = 1.0

    mid_mask = (1.0 - bow_mask - punct_mask - useless_mask - eos_mask).clamp(0.0, 1.0)
    return {"bow": bow_mask, "punct": punct_mask, "eos": eos_mask, "mid": mid_mask}

def compute_pimentel_fixes_from_logits(logits, masks):
    probs = torch.softmax(logits.float(), dim=-1).to(dtype=logits.dtype)
    eps = torch.finfo(probs.dtype).tiny

    bow_vocab = (masks["bow"] + masks["eos"]).clamp(max=1.0)
    bos_vocab = (masks["mid"] + masks["punct"] + masks["eos"]).clamp(max=1.0)

    p_bow = (probs * bow_vocab).sum(-1).clamp_min(eps)
    p_bos = (probs * bos_vocab).sum(-1).clamp_min(eps)

    bow_fix = -torch.log(p_bow)
    bos_fix = -torch.log(p_bos)

    eow_fix = torch.empty_like(bow_fix)
    eow_fix[:, :-1] = bow_fix[:, 1:]
    eow_fix[:, -1] = bow_fix[:, -1]

    return bow_fix, bos_fix, eow_fix

def rank_tercile(values):
    values = np.array(values)
    idx = np.argsort(values)
    labels = np.zeros(len(values), dtype=int)
    n = len(values) // 3
    labels[idx[:n]] = 1
    labels[idx[n:2*n]] = 2
    labels[idx[2*n:]] = 3
    return labels.tolist()

def compute_metrics(true, pred):
    return {
        "Accuracy": accuracy_score(true, pred),
        "Precision": precision_score(true, pred, average="macro", zero_division=0),
        "Recall": recall_score(true, pred, average="macro", zero_division=0),
        "Macro-F1": f1_score(true, pred, average="macro", zero_division=0),
    }

# ===== Paths =====
INPUT_FILE = r""
OUTPUT_DIR = Path(r"")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== Qwen-2.5 models only (4个) =====
MODEL_LIST = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
]

# ===== Load data =====
df = pd.read_excel(INPUT_FILE)
df.columns = ["sentence", "label"]
df["target"] = df["sentence"].str[-2:]

all_metrics = []

# ===== Main loop =====
for model_name in MODEL_LIST:
    print(f"\n==== Processing model: {model_name} ====")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)
    model.eval()

    masks = build_pimentel_masks(tokenizer, model)

    surprisal_direct = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        sent = row["sentence"]
        target = row["target"]

        enc = tokenizer(
            sent,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=False
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        input_ids = enc["input_ids"]
        offsets = enc["offset_mapping"][0].tolist()

        eos = torch.full((1, 1), tokenizer.eos_token_id, device=DEVICE)
        labels = torch.cat([input_ids, eos], dim=1)

        bos_id = tokenizer.bos_token_id
        if bos_id is None:
            bos_id = tokenizer.eos_token_id

        bos = torch.full((1, 1), bos_id, device=DEVICE)
        input_ids_bos = torch.cat([bos, labels[:, :-1]], dim=1)

        with torch.no_grad():
            logits = model(input_ids_bos).logits

        bow_fix, bos_fix, eow_fix = compute_pimentel_fixes_from_logits(logits, masks)
        log_probs = torch.log_softmax(logits, dim=-1)
        tok_lp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        surprisal_tok = -tok_lp

        start = sent.rfind(target)
        end = len(sent)
        span = subtoken_indices_for_char_span(offsets + [(0, 0)], start, end)
        span = torch.tensor(span, device=DEVICE)

        s = (
            surprisal_tok[0, span]
            - bos_fix[0, span]
            - bow_fix[0, span]
            + eow_fix[0, span]
        ).sum().item()

        surprisal_direct.append(s)

    preds = rank_tercile(surprisal_direct)

    metrics = compute_metrics(df["label"], preds)
    metrics["Model"] = model_name
    all_metrics.append(metrics)

    out_df = df.copy()
    out_df["surprisal"] = surprisal_direct
    out_df["pred"] = preds
    out_df.to_excel(OUTPUT_DIR / f"{model_name.replace('/','_')}.xlsx", index=False)

# ===== Save metrics =====
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_excel(OUTPUT_DIR / "qwen25_metrics.xlsx", index=False)
print(metrics_df)
