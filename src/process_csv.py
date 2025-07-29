import csv, re, pandas as pd, regex, requests, io, torch, open_clip
from PIL import Image
from tqdm import tqdm

SRC  = r"C:\Parser\vk.barkov.net-marketitems-2025-07-28_13-22-22.csv"
DEST = r"C:\Parser\vk_items_parsed.xlsx"

# ─── CLIP (цвет корпуса) ─────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tok   = open_clip.get_tokenizer("ViT-B-32")
txt_f = model.encode_text(tok(["white pc case", "black pc case"]).to(device)).float()
txt_f /= txt_f.norm(dim=-1, keepdim=True)

def clip_color(url):
    try:
        img = Image.open(io.BytesIO(requests.get(url, timeout=10).content))
        img = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            imf = model.encode_image(img).float(); imf /= imf.norm(dim=-1, keepdim=True)
            return "White" if (imf @ txt_f.T)[0,0] > (imf @ txt_f.T)[0,1] else "Black"
    except Exception:
        return "не указано"

# ─── паттерны ─────────────────────────────────────────────────
CPU_RX  = regex.compile(r"(?i)\b(i[3579]-?\d{3,5}[KF]?|i\d-\d{3,4}|r[3579]\d{3,4}[xX]?|r[3579]-?\d{4}|i[57]-\d{4}|xeon\s?\w+\d+|pentium\s?\w+\d*|celeron\s?\w+\d*)\b")
GPU_RX  = regex.compile(r"(?i)\b(rt[rx]|gtx|rx|arc)\s?-?\d{3,4}(?:\s?super|ti)?\b")
RAM_RX  = re.compile(r"\b(4|6|8|12|16|24|32|48|64|96|128)\s?(GB|ГБ)\b", re.I)
RAM_OK  = re.compile(r"(оператив|озу|ddr|ram)",re.I)
RAM_BAD = re.compile(r"(rtx|gtx|rx|gpu|видеокар)",re.I)
COL_TXT = re.compile(r"корпус[^:]*:\s*[\w\s-]*?(white|wh|белый|black|bk|ч[её]рный)",re.I)
C_MAP   = {"white":"White","wh":"White","белый":"White",
           "black":"Black","bk":"Black","чёрный":"Black","черный":"Black"}

BRAND_MAP = {"vapcbuild":"VA-PC","va-pc":"VA-PC","vavpc":"VA-PC",
             "maxxpc":"MaxxPC","compshop":"CompShop","bzone":"bzone54","ofosters":"ofosters"}

# ─── helpers ──────────────────────────────────────────────────
def get_brand(row):
    g = row.get("НАЗВАНИЕ ГРУППЫ","") or ""
    if g: 
        for k,v in BRAND_MAP.items():
            if k in g.lower(): return v
        return g.strip()
    url = row.get("АДРЕС ГРУППЫ","") or ""
    for k,v in BRAND_MAP.items():
        if k in url.lower(): return v
    return "не указано"

def get_cpu(text): m=CPU_RX.search(text); return m.group(0).upper() if m else "не указан"
def get_gpu(text): m=GPU_RX.search(text); return m.group(0).upper() if m else "не указан"

def get_ram(desc):
    for frag in re.split(r"[;\n|,]", desc):
        if RAM_OK.search(frag) and not RAM_BAD.search(frag):
            m = RAM_RX.search(frag)
            if m: return f"{m.group(1)} GB"
    return "не указан"

def get_color(desc, photo):
    m = COL_TXT.search(desc)
    if m: return C_MAP[m.group(1).lower()]
    return clip_color(photo) if photo else "не указано"

# ─── чтение CSV ───────────────────────────────────────────────
rows=[]; skipped=0
with open(SRC, newline='', encoding="utf-8") as f:
    reader = csv.DictReader(
        (r for r in f if r.strip() and not r.startswith("-")),
        delimiter=";")
    for row in tqdm(reader, desc="парсинг"):
        try:
            name  = row["НАЗВАНИЕ"]
            price = row["ЦЕНА"]
            desc  = (row["ОПИСАНИЕ"] or "").replace("\t"," ")
            photo = row.get("ФОТО-ПРЕВЬЮ","")
            brand = get_brand(row)

            rows.append([
                brand,
                get_cpu(desc+" "+name),
                get_gpu(desc+" "+name),
                get_ram(desc),
                get_color(desc,photo),
                price,
                photo
            ])
        except KeyError:  # если нет нужных колонок
            skipped += 1

cols=["Название компании/мастерской","Модель процессора (CPU)",
      "Модель видеокарты (GPU)","Объём ОЗУ","Цвет корпуса",
      "Финальная цена","Фото"]

pd.DataFrame(rows, columns=cols).to_excel(DEST,index=False)
print(f"✔ сохранено {len(rows)} товаров (пропущено {skipped}) → {DEST}")
