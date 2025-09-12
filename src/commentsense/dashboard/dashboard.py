import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
df = pd.read_csv(os.path.join(ASSETS_DIR, "comments2_analyzed.csv"))

st.set_page_config(page_title="CommentSense Dashboard", layout="wide")
st.title("CommentSense – AI-Powered Comment Analysis")

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Quality Ratio", f"{df['is_quality'].mean()*100:.1f}%")
col2.metric("Spam Rate", f"{df['spam_flag'].mean()*100:.1f}%")
col3.metric("Avg Quality Score", f"{df['quality_score'].mean():.2f}")
col4.metric("# Comments", f"{len(df):,}")

# Filters
cats = ["All"] + sorted(df["category"].unique().tolist())
cat = st.selectbox("Category Filter", cats)
if cat != "All":
    df = df[df["category"]==cat]

# Sentiment Chart
st.subheader("Sentiment Breakdown")
st.bar_chart(df["sentiment"].value_counts())

# Category Chart
st.subheader("Category Distribution")
st.bar_chart(df["category"].value_counts())

# Per-video summary
st.subheader("Per Video Quality Metrics")
per_video = pd.read_csv(os.path.join(ASSETS_DIR, "per_video_summary.csv"))
st.dataframe(per_video.sort_values("quality_ratio", ascending=False))

# Top quality comments
st.subheader("Top Quality Comments")
topq = df[df["is_quality"]].sort_values(["quality_score","likeCount"], ascending=False)[
    ["videoId","likeCount","category","sentiment","quality_score","textOriginal"]
].head(50)
st.dataframe(topq)

# Spam review
st.subheader("Spam Comments (Sample)")
spam_df = df[df["spam_flag"]=="spam"].head(50)
st.dataframe(spam_df[["videoId","textOriginal","likeCount"]])

# Category Pie Chart
st.subheader("Postive Category Breakdown")

df["sentiment"] = df["sentiment"].str.lower()
positive_df = df[df["sentiment"] == "positive"]
cat_counts = positive_df["category"].value_counts(normalize=True) * 100

fig, ax = plt.subplots()

# plot pie without autopct
wedges, texts = ax.pie(
    cat_counts,
    startangle=90
)

# add legend with percentages
labels = [f"{cat} - {val:.1f}%" for cat, val in zip(cat_counts.index, cat_counts.values)]
ax.legend(
    wedges,
    labels,
    title="Categories",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1)  # outside chart
)

ax.set_title("Comment Breakdown by Category")
st.pyplot(fig)