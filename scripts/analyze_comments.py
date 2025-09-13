import pandas as pd
import re
from transformers import pipeline
import torch
import math
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------------
# 0. Paths
# ----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # parent of script

ASSETS_DIR = os.path.join(PROJECT_ROOT, "src", "commentsense", "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

input_file = os.path.join(ASSETS_DIR, "comments2.csv")
output_file = os.path.join(ASSETS_DIR, "comments2_analyzed.csv")
summary_file = os.path.join(ASSETS_DIR, "per_video_summary.csv")

print("Looking for:", input_file)


print("Current working directory:", os.getcwd())

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
try:
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        truncation=True,
        device=0 if torch.cuda.is_available() else -1)
except Exception as e:
    print(f"Error loading sentiment model: {e}")
    # Fallback to a simpler model
    sentiment_pipeline = pipeline("sentiment-analysis", device=0 if torch.cuda.is_available() else -1)


def analyze_batch(batch_id, batch_texts):

    print(f"[Batch {batch_id}] Starting ({len(batch_texts)} comments)...")
    results = sentiment_pipeline(batch_texts, batch_size=128, truncation=True)
    print(f"[Batch {batch_id}] Done")
    return batch_id, [r["label"] for r in results]

BATCH_SIZE = 5000
MAX_BATCHES = 2
texts = df["cleaned_comment"].tolist()
futures = []

with ThreadPoolExecutor(max_workers=2) as executor:  # 2 workers safer for GPU
    for batch_num, i in enumerate(range(0, len(texts), BATCH_SIZE)):
        if batch_num >= MAX_BATCHES:
            break
        batch_texts = texts[i:i+BATCH_SIZE]
        futures.append(executor.submit(analyze_batch, batch_num, batch_texts))

# Collect results in correct order
batch_results = {}
for f in as_completed(futures):
    batch_id, labels = f.result()
    batch_results[batch_id] = labels

# Rebuild sentiment list in batch order
sentiments = []
for batch_num in range(len(batch_results)):
    sentiments.extend(batch_results[batch_num])

# Pad if incomplete
if len(sentiments) < len(df):
    sentiments.extend(["not_processed"] * (len(df) - len(sentiments)))

df["sentiment"] = sentiments

# ----------------------------
# 5. SPAM DETECTION
# ----------------------------
print("Loading spam detection model...")
spam_detector = pipeline(
    "text-classification",
    model="valurank/distilroberta-spam-comments-detection",  # Better for social media spam
    tokenizer="valurank/distilroberta-spam-comments-detection",
    truncation=True,
    padding=True,
    max_length=512,
    device=0 if torch.cuda.is_available() else -1
)
del sentiment_pipeline
gc.collect()
print("Cleaned up sentiment analysis memory")
# Process in smaller batches to avoid memory issues
spam_results = []
for i in range(0, len(df), 256):
    batch_texts = df["cleaned_comment"].iloc[i:i+256].astype(str).tolist()
    print(f"Processing batch {i//256 + 1}...")
    batch_results = spam_detector(batch_texts, batch_size=32)
    spam_results.extend(batch_results)

df["spam_flag"] = [1 if r["label"] == "spam" else 0 for r in spam_results]
df["spam_confidence"] = [r["score"] for r in spam_results]

print(f"Spam detection complete. Found {df['spam_flag'].sum()} spam comments.")
del spam_detector
gc.collect()
print("Cleaned up spam detection memory")


# 6. RELEVANCE & CATEGORY ANALYSIS
# ----------------------------
# Enhanced keyword lists for beauty industry analysis
BEAUTY_KEYWORDS = {
    "loreal_brand": ["loreal", "l'oréal", "lóreal", "l'oreal", "loreal paris"],
    "skincare": [
        "skin", "skincare", "serum", "retinol", "hyaluronic", "moisturizer", 
        "sunscreen", "spf", "toner", "cleanser", "acne", "wrinkle", "glow",
        "dark spot", "pigmentation", "hydration", "dry skin", "oily skin",
        "anti-aging", "anti aging", "brightening", "exfoliate", "peel"
    ],
    "makeup": [
        "makeup", "foundation", "concealer", "mascara", "lipstick", "blush",
        "eyeliner", "palette", "brow", "eyebrow", "lip gloss", "primer",
        "highlighter", "contour", "make up", "cosmetics", "powder", "compact"
    ],
    "fragrance": [
        "fragrance", "perfume", "scent", "eau de parfum", "edp", 
        "eau de toilette", "edt", "cologne", "aroma", "smell", "notes"
    ],
    "haircare": [
        "hair", "shampoo", "conditioner", "haircare", "hair care",
        "smooth", "frizz", "dye", "color", "coloring", "hair mask",
        "hair oil", "scalp", "dandruff", "hair fall", "hair loss",
        "keratin", "straight", "curl", "volum", "shiny hair"
    ],
    "products": [
        "product", "item", "buy", "purchase", "price", "cost", "expensive",
        "cheap", "affordable", "value", "worth it", "recommend", "repurchase"
    ],
    "application": [
        "use", "apply", "how to", "routine", "morning", "night", "daily",
        "step", "method", "technique", "tip", "tutorial", "demo"
    ]
}

def is_beauty_relevant(t):
    """
    Determine if comment is relevant to beauty products with confidence scoring
    """
    if not isinstance(t, str):
        return "not_relevant", 0.0
    
    text_lower = t.lower()
    relevance_score = 0.0
    
    # Check for brand mentions (high weight)
    brand_mentions = sum(1 for kw in BEAUTY_KEYWORDS["loreal_brand"] if kw in text_lower)
    if brand_mentions > 0:
        relevance_score += 0.6
    
    # Check for product category mentions (medium weight)
    category_mentions = 0
    for category in ["skincare", "makeup", "fragrance", "haircare"]:
        if any(kw in text_lower for kw in BEAUTY_KEYWORDS[category]):
            category_mentions += 1
            relevance_score += 0.2
    
    # Check for product-related terms (low weight)
    if any(kw in text_lower for kw in BEAUTY_KEYWORDS["products"]):
        relevance_score += 0.1
    
    # Check for application terms (low weight)
    if any(kw in text_lower for kw in BEAUTY_KEYWORDS["application"]):
        relevance_score += 0.1
    
    # Determine relevance based on threshold
    if relevance_score >= 0.3:
        return "relevant", min(1.0, relevance_score)
    else:
        return "not_relevant", relevance_score

def extract_key_products(text):
    """
    Extract specific products mentioned in comments
    """
    if not isinstance(text, str):
        return []
    
    text_lower = text.lower()
    products_mentioned = []
    
    # Common L'Oréal product lines
    loreal_products = [
        "revitalift", "age perfect", "collagen", "hyaluron", "glycolic",
        "true match", "infallible", "telescopic", "voluminous", "colour riche",
        "elseve", "elvive", "professionnel", "everpure", "everstrong"
    ]
    
    for product in loreal_products:
        if product in text_lower:
            products_mentioned.append(product)
    
    return products_mentioned
def categorize_comment(text):
    """
    Multi-category classification for beauty comments
    Returns primary category and all detected categories
    """
    if not isinstance(text, str):
        return "Other", []
    
    text_lower = text.lower()
    detected_categories = []
    
    # Check each category
    for category, keywords in {
        "Skincare": BEAUTY_KEYWORDS["skincare"],
        "Makeup": BEAUTY_KEYWORDS["makeup"], 
        "Fragrance": BEAUTY_KEYWORDS["fragrance"],
        "Haircare": BEAUTY_KEYWORDS["haircare"]
    }.items():
        if any(keyword in text_lower for keyword in keywords):
            detected_categories.append(category)
    
    # Brand-specific category (highest priority)
    if any(kw in text_lower for kw in BEAUTY_KEYWORDS["loreal_brand"]):
        detected_categories.append("L'Oréal Brand")
    
    # Determine primary category
    if not detected_categories:
        return "Other", []
    
    # Priority order for primary category
    priority_order = ["L'Oréal Brand", "Skincare", "Makeup", "Haircare", "Fragrance"]
    for category in priority_order:
        if category in detected_categories:
            return category, detected_categories
    
    return detected_categories[0], detected_categories
# Apply enhanced analysis


# Apply relevance analysis with confidence scoring
relevance_results = df["cleaned_comment"].apply(is_beauty_relevant)
df["relevance"] = [result[0] for result in relevance_results]
df["relevance_confidence"] = [result[1] for result in relevance_results]

# Apply category analysis
category_results = df["cleaned_comment"].apply(categorize_comment)
df["primary_category"] = [result[0] for result in category_results]
df["all_categories"] = [result[1] for result in category_results]

# Extract product mentions
df["products_mentioned"] = df["cleaned_comment"].apply(extract_key_products)

# Brand mention flag
df["mentions_loreal"] = df["cleaned_comment"].apply(
    lambda x: any(kw in x.lower() for kw in BEAUTY_KEYWORDS["loreal_brand"]) if isinstance(x, str) else False
)
# ----------------------------
# 7. QUALITY SCORE
# ----------------------------
def sentiment_to_score(label):
    # Map new sentiment labels to scores
    label_str = str(label).upper()
    if "LABEL_2" in label_str:  # Positive
        return 1.0
    elif "LABEL_0" in label_str:  # Negative
        return -0.5
    elif "LABEL_1" in label_str:  # Neutral
        return 0.2
    else:  # not_processed or unknown
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
    if row["spam_flag"] == 1:
        return 0.0
    rel = 1.0 if row["relevance"] == "relevant" else 0.0
    snt01 = (sentiment_to_score(row["sentiment"]) + 1.0)/2.0
    sub = substance_score(row["cleaned_comment"])
    eng = engagement_score(row)
    return round(W_REL*rel + W_SNT*snt01 + W_LEN*sub + W_ENG*eng, 3)

df["quality_score"] = df.apply(quality_row,axis=1)
df["is_quality"] = df["quality_score"] >= 0.65

# ----------------------------
# 8. SAVE RESULTS
# ----------------------------

print("Preparing to save results...")

# Convert list columns to strings for CSV compatibility
df["all_categories_str"] = df["all_categories"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
df["products_mentioned_str"] = df["products_mentioned"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))

# Define safe columns for CSV output
safe_columns = [
    "commentId", "videoId", "textOriginal", "cleaned_comment", "likeCount",
    "sentiment", "spam_flag", "spam_confidence", "relevance", 
    "relevance_confidence", "primary_category", "all_categories_str",
    "products_mentioned_str", "mentions_loreal", "quality_score", "is_quality"
]

# Keep only columns that actually exist
existing_columns = [col for col in safe_columns if col in df.columns]
existing_columns.extend([col for col in df.columns if col not in safe_columns and col not in ['all_categories', 'products_mentioned']])

try:
    df[existing_columns].to_csv(output_file, index=False, encoding="utf-8")
    print(f"Successfully saved {output_file}")
except Exception as e:
    print(f"Error saving main CSV: {e}")
    # Try a minimal version
    minimal_cols = ["commentId", "videoId", "cleaned_comment", "sentiment", "relevance", "quality_score"]
    df[minimal_cols].to_csv(output_file, index=False, encoding="utf-8")
    print("Saved minimal version")

# Create per-video summary
try:
    # Find the correct comment count column
    comment_count_col = None
    for col in ['commentId', 'id', 'comment_id']:
        if col in df.columns:
            comment_count_col = col
            break
    
    if comment_count_col:
        per_video = df.groupby("videoId").agg(
            quality_ratio=("is_quality", "mean"),
            avg_quality=("quality_score", "mean"),
            comments=(comment_count_col, "count")
        ).reset_index()
        per_video["quality_ratio"] = (per_video["quality_ratio"] * 100).round(1)
        per_video.to_csv(summary_file, index=False, encoding="utf-8")
        print(f"Saved {summary_file}")
    else:
        print("Could not find comment count column for per_video summary")
        
except Exception as e:
    print(f"Error creating per-video summary: {e}")

print("Analysis complete!")
print("Quality Ratio Overall:", f"{df['is_quality'].mean()*100:.2f}%")
print("Relevant comments:", f"{(df['relevance'] == 'relevant').mean()*100:.2f}%")
print("L'Oréal mentions:", f"{df['mentions_loreal'].mean()*100:.2f}%")
