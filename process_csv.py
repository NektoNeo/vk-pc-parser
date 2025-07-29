import csv
import io
import re
from pathlib import Path

import pandas as pd
import regex
import requests
import torch
import open_clip
from PIL import Image
from tqdm import tqdm

# Input and output paths
# The script expects to live in the same directory as the CSV file
# (for example ``C:\Parser`` on Windows). The result ``vk_items_parsed.xlsx``
# will also be saved there.
BASE_DIR = Path(__file__).resolve().parent
SRC = BASE_DIR / "vk.barkov.net-marketitems-2025-07-28_13-22-22.csv"
DEST = BASE_DIR / "vk_items_parsed.xlsx"

# --- CLIP model for case color ---------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
model = model.to(device)
_tokenizer = open_clip.get_tokenizer("ViT-B-32")
with torch.no_grad():
    text_features = model.encode_text(_tokenizer(["white pc case", "black pc case"]).to(device)).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

def clip_color(url: str) -> str:
    try:
        img = Image.open(io.BytesIO(requests.get(url, timeout=10).content)).convert("RGB")
        img = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feats = model.encode_image(img).float(); feats /= feats.norm(dim=-1, keepdim=True)
            sims = (feats @ text_features.T)[0]
            return "White" if sims[0] > sims[1] else "Black"
    except Exception:
        return ""

# --- Regular expressions ----------------------------------------------------
CPU_RX = regex.compile(r"(?i)\b(?:i[3579]|i[1-9]|r[3579]|r5)\s*-?\d{4,5}[a-z]?\b")
GPU_RX = regex.compile(r"(?i)\b(?:rtx|gtx|rx|arc)\s*-?\d{3,4}(?:\s?(?:super|ti))?\b")
RAM_OK = regex.compile(r"(оператив|озу|ddr|ram)", regex.I)
RAM_BAD = regex.compile(r"(rtx|gtx|rx|gpu|видеокарт)", regex.I)
RAM_RX = regex.compile(r"\b(8|16|32|48|64|96|128)\s?(?:gb|гб)\b", regex.I)
COL_RX = regex.compile(r"корпус[^:]*:\s*(white|wh|белый|black|bk|ч[её]рный|черный)", regex.I)
COLORS = {"white": "White", "wh": "White", "белый": "White", "black": "Black", "bk": "Black", "чёрный": "Black", "черный": "Black"}

# --- Helpers ---------------------------------------------------------------

def find_index(header, *keywords):
    low = [h.lower() for h in header]
    for kw in keywords:
        for i, h in enumerate(low):
            if kw in h:
                return i
    raise KeyError(f"column not found: {keywords}")

def get_cpu(desc: str, name: str) -> str:
    m = CPU_RX.search(desc)
    if not m:
        m = CPU_RX.search(name)
    return m.group(0).upper() if m else ""

def get_gpu(text: str) -> str:
    m = GPU_RX.search(text)
    return m.group(0).upper() if m else ""

def get_ram(desc: str) -> str:
    for part in re.split(r"[;\n|,]", desc):
        if RAM_OK.search(part) and not RAM_BAD.search(part):
            m = RAM_RX.search(part)
            if m:
                return f"{m.group(1)} GB"
    return ""

def get_color(desc: str, photo: str) -> str:
    m = COL_RX.search(desc)
    if m:
        return COLORS[m.group(1).lower()]
    return clip_color(photo) if photo else ""

# --- Read CSV and process --------------------------------------------------
rows = []
skipped = 0
with open(SRC, newline="", encoding="utf-8") as f:
    reader = csv.reader(
        (
            line for line in f
            if line.strip() and not line.lstrip("\ufeff").startswith("-")
        ),
        delimiter=";",
    )
    header = next(reader)
    try:
        idx_name = find_index(header, "назван")
        idx_price = find_index(header, "цена")
        idx_desc = find_index(header, "описан")
        idx_group = find_index(header, "название группы")
        idx_photo = find_index(header, "фото")
    except KeyError:
        raise SystemExit("Не найдены нужные столбцы")

    for row in reader:
        if len(row) < 8:
            skipped += 1
            continue
        name = row[idx_name]
        price = row[idx_price]
        desc = row[idx_desc] or ""
        group = row[idx_group]
        photo = row[idx_photo]
        text = f"{desc} {name}"
        rows.append([
            group.strip(),
            get_cpu(desc, name),
            get_gpu(text),
            get_ram(desc),
            get_color(desc, photo),
            price.strip(),
            photo.strip(),
        ])

cols = [
    "Название компании/мастерской",
    "Модель процессора (CPU)",
    "Модель видеокарты (GPU)",
    "Объём ОЗУ",
    "Цвет корпуса",
    "Финальная цена",
    "Фото",
]

pd.DataFrame(rows, columns=cols).to_excel(DEST, index=False)
print(f"✔ Готово: {len(rows)} товаров сохранено в {DEST}")
