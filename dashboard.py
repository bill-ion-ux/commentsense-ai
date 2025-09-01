import streamlit as st
import pandas as pd

st.set_page_config(page_title="CommentSense Dashboard", layout="wide")

# Load analyzed data
df = pd.read_csv("comments2_analyzed.csv")

st.title("CommentSense – AI-Powered Comment Analysis")

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Quality Ratio", f"{df['is_quality'].mean()*100:.1f}%")
col2.metric("Spam Rate", f"{(df['spam_flag']=='spam').mean()*100:.1f}%")
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
per_video = pd.read_csv("per_video_summary.csv")
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
