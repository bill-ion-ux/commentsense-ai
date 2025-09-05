import pandas as pd
import re
from transformers import pipeline
import torch
import math
import os

# ----------------------------
# 0. Paths
# ----------------------------
ASSETS_DIR = "./src/commentsense/assets"
os.makedirs(ASSETS_DIR, exist_ok=True)  # Ensure folder exists

input_file = os.path.join(ASSETS_DIR, "comments2.csv")
output_file = os.path.join(ASSETS_DIR, "comments2_analyzed.csv")
summary_file = os.path.join(ASSETS_DIR, "per_video_summary.csv")

# ----------------------------
# 1. GPU Check
# ----------------------------
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ----------------------------
# 2. LOAD DATA
# ----------------------------
df = pd.read_csv(input_file, encoding='utf-8', on_bad_lines='skip')
print(f"Loaded {len(df)} records")
print("Columns:", df.columns)

# ----------------------------
# 3. CLEAN TEXT
# ----------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.encode('latin1', 'ignore').decode('utf-8', 'ignore')
    text = re.sub(r"[^\w\s.,!?@#%&:;()\-\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+", "", text)
    return text.strip()

df["cleaned_comment"] = df["textOriginal"].apply(clean_text)

# ----------------------------
# 4. SENTIMENT ANALYSIS (LIMITED)
# ----------------------------
print("Loading sentiment model...")
sentiment_pipeline = pipeline("sentiment-analysis", device=0 if torch.cuda.is_available() else -1)

BATCH_SIZE = 5000
MAX_BATCHES = 2
sentiments = []

total_batches = math.ceil(len(df) / BATCH_SIZE)
print(f"Processing only {MAX_BATCHES} out of {total_batches} batches...")

for batch_num, i in enumerate(range(0, len(df), BATCH_SIZE)):
    if batch_num >= MAX_BATCHES:
        break
    batch_texts = df["cleaned_comment"].iloc[i:i+BATCH_SIZE].tolist()
    print(f"Processing batch {batch_num + 1}/{MAX_BATCHES} ...")
    results = sentiment_pipeline(batch_texts, batch_size=32, truncation=True)
    sentiments.extend([r["label"] for r in results])

if len(sentiments) < len(df):
    sentiments.extend(["not_processed"] * (len(df) - len(sentiments)))

df["sentiment"] = sentiments

# ----------------------------
# 5. SPAM DETECTION
# ----------------------------
def detect_spam(text):
    text = text.lower()
    if "http" in text or "subscribe" in text or "follow me" in text:
        return "spam"
    return "not_spam"

df["spam_flag"] = df["cleaned_comment"].apply(detect_spam)

# ----------------------------
# 6. RELEVANCE & CATEGORY
# ----------------------------
RELEVANCE_KEYWORDS = [
    "loreal","l'oréal","serum","retinol","hyaluronic","lipstick","mascara",
    "foundation","sunscreen","spf","moisturizer","fragrance","perfume",
    "toner","cleanser","concealer","hair","shampoo","conditioner","skincare","makeup"
]

def is_relevant(t):
    t = (t or "").lower()
    return "relevant" if any(k in t for k in RELEVANCE_KEYWORDS) else "not_relevant"

df["relevance"] = df["cleaned_comment"].apply(is_relevant)

CATEGORY_MAP = {
    "Skincare": ["skin","skincare","serum","retinol","hyaluronic","toner","cleanser","moisturizer","sunscreen","spf","acne"],
    "Fragrance": ["fragrance","perfume","eau de parfum","edp","eau de toilette","edt","cologne","scent"],
    "Makeup": ["makeup","foundation","concealer","mascara","lipstick","blush","eyeliner","palette","brow"]
}

def categorize(t):
    t = (t or "").lower()
    for cat, kws in CATEGORY_MAP.items():
        if any(k in t for k in kws):
            return cat
    return "Other"

df["category"] = df["cleaned_comment"].apply(categorize)

# ----------------------------
# 7. QUALITY SCORE
# ----------------------------
def sentiment_to_score(label):
    l = (label or "").lower()
    if "pos" in l:
        return 1.0
    if "neg" in l:
        return -0.5
    return 0.2

def substance_score(t):
    n = len((t or "").split())
    return 1.0 if n>=12 else 0.6 if n>=6 else 0.3 if n>=3 else 0.1

def engagement_score(row):
    try:
        lc = float(row.get("likeCount", 0))
    except:
        lc = 0.0
    return min(1.0, math.log1p(lc)/math.log1p(50))

W_REL, W_SNT, W_LEN, W_ENG = 0.40, 0.35, 0.15, 0.10

def quality_row(row):
    if row["spam_flag"] == "spam":
        return 0.0
    rel = 1.0 if row["relevance"] == "relevant" else 0.0
    snt01 = (sentiment_to_score(row["sentiment"]) + 1.0)/2.0
    sub = substance_score(row["cleaned_comment"])
    eng = engagement_score(row)
    return round(W_REL*rel + W_SNT*snt01 + W_LEN*sub + W_ENG*eng, 3)

df["quality_score"] = df.apply(quality_row, axis=1)
df["is_quality"] = df["quality_score"] >= 0.65

# ----------------------------
# 8. SAVE RESULTS
# ----------------------------
df.to_csv(output_file, index=False, encoding="utf-8")
print("Saved", output_file)

per_video = df.groupby("videoId").agg(
    quality_ratio=("is_quality","mean"),
    avg_quality=("quality_score","mean"),
    comments=("commentId","count")
).reset_index()
per_video["quality_ratio"] = (per_video["quality_ratio"]*100).round(1)
per_video.to_csv(summary_file, index=False, encoding="utf-8")
print("Saved", summary_file)

print("Analysis complete!")
print("Quality Ratio Overall:", f"{df['is_quality'].mean()*100:.2f}%")
