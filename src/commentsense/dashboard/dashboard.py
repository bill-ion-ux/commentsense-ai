import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config first
st.set_page_config(page_title="CommentSense Dashboard", layout="wide")

# Set style
plt.style.use('default')
sns.set_palette("husl")

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")

try:
    df = pd.read_csv(os.path.join(ASSETS_DIR, "comments2_analyzed.csv"))
    per_video = pd.read_csv(os.path.join(ASSETS_DIR, "per_video_summary.csv"))
except FileNotFoundError:
    st.error("Data files not found! Please run the analysis first.")
    st.stop()

st.title("CommentSense – AI-Powered Comment Analysis for L'Oréal")
st.markdown("### Advanced analytics for beauty product video comments")

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Quality Ratio", f"{df['is_quality'].mean()*100:.1f}%")
col2.metric("Spam Rate", f"{df['spam_flag'].mean()*100:.1f}%")
col3.metric("Avg Quality Score", f"{df['quality_score'].mean():.2f}")
col4.metric("Total Comments", f"{len(df):,}")

# Additional KPIs
col5, col6, col7, col8 = st.columns(4)
col5.metric("Relevant Comments", f"{(df['relevance'] == 'relevant').mean()*100:.1f}%")
col6.metric("L'Oréal Mentions", f"{df['mentions_loreal'].mean()*100:.1f}%")
positive_sentiment = (df['sentiment'].str.contains('LABEL_2', case=False, na=False)).mean()
col7.metric("Positive Sentiment", f"{positive_sentiment*100:.1f}%")
col8.metric("High Quality Count", f"{df['is_quality'].sum():,}")

# Filters
st.sidebar.header("Filters")
cats = ["All"] + sorted(df["primary_category"].unique().tolist())
cat = st.sidebar.selectbox("Category Filter", cats)

sentiments = ["All"] + sorted(df["sentiment"].unique().tolist())
sentiment_filter = st.sidebar.selectbox("Sentiment Filter", sentiments)

quality_threshold = st.sidebar.slider("Minimum Quality Score", 0.0, 1.0, 0.0, 0.1)

# Apply filters
filtered_df = df.copy()
if cat != "All":
    filtered_df = filtered_df[filtered_df["primary_category"] == cat]
if sentiment_filter != "All":
    filtered_df = filtered_df[filtered_df["sentiment"] == sentiment_filter]
filtered_df = filtered_df[filtered_df["quality_score"] >= quality_threshold]

st.sidebar.markdown(f"**Filtered:** {len(filtered_df):,} comments")

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Category Analysis", "Quality Insights", "Raw Data"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sentiment Analysis")
        sentiment_counts = filtered_df["sentiment"].value_counts()
        # Map to readable names
        sentiment_mapping = {
            "LABEL_0": "Negative", 
            "LABEL_1": "Neutral", 
            "LABEL_2": "Positive", 
            "not_processed": "Not Processed"
        }
        sentiment_readable = sentiment_counts.rename(index=sentiment_mapping)
        fig, ax = plt.subplots(figsize=(8, 6))
        sentiment_readable.plot(kind='bar', ax=ax, color=['#ff6b6b', '#ffd93d', '#6bcf7f', '#95a5a6'])
        ax.set_title("Sentiment Distribution")
        ax.set_ylabel("Number of Comments")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Relevance Analysis")
        relevance_counts = filtered_df["relevance"].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        relevance_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax, 
                             colors=['#6bcf7f', '#ff6b6b'], startangle=90)
        ax.set_title("Comment Relevance to Beauty Products")
        ax.set_ylabel("")
        st.pyplot(fig)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Category Distribution (Bar)")
        category_counts = filtered_df["primary_category"].value_counts()
        fig, ax = plt.subplots(figsize=(10, 6))
        category_counts.plot(kind='bar', ax=ax, color='#4ecdc4')
        ax.set_title("Comments by Product Category")
        ax.set_ylabel("Number of Comments")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Category Distribution (Pie)")
        category_counts = filtered_df["primary_category"].value_counts(normalize=True) * 100
        fig, ax = plt.subplots(figsize=(8, 6))
        wedges, _ = ax.pie(category_counts, startangle=90)  # no autopct
        # legend outside with percentages
        labels = [f"{cat} - {val:.1f}%" for cat, val in zip(category_counts.index, category_counts.values)]
        ax.legend(wedges, labels, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        ax.set_title("Category Share (%)")
        st.pyplot(fig)

    # Keep your existing stacked sentiment chart below
    st.subheader("Sentiment by Category")
    sentiment_by_cat = pd.crosstab(
        filtered_df["primary_category"], 
        filtered_df["sentiment"].replace({
            "LABEL_0": "Negative",
            "LABEL_1": "Neutral",
            "LABEL_2": "Positive"
        })
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sentiment_by_cat.plot(kind='bar', ax=ax, stacked=True)
    ax.set_title("Sentiment Distribution by Category")
    ax.set_ylabel("Number of Comments")
    plt.xticks(rotation=45, ha='right')
    ax.legend(title="Sentiment")
    st.pyplot(fig)


with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Quality Score Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(filtered_df["quality_score"], bins=20, color='#45b7d1', alpha=0.7, edgecolor='black')
        ax.axvline(x=0.65, color='red', linestyle='--', label='Quality Threshold (0.65)')
        ax.set_title("Distribution of Quality Scores")
        ax.set_xlabel("Quality Score")
        ax.set_ylabel("Number of Comments")
        ax.legend()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Top Product Mentions")
        # Flatten the products_mentioned list
        all_products = []
        for products in filtered_df["products_mentioned"].dropna():
            if isinstance(products, str):
                # Convert string representation of list to actual list
                try:
                    products = eval(products)
                except:
                    products = [products]
            if isinstance(products, list):
                all_products.extend(products)
        
        if all_products:
            from collections import Counter
            product_counts = Counter(all_products).most_common(10)
            products, counts = zip(*product_counts)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(products, counts, color='#ff9a8b')
            ax.set_title("Top 10 Mentioned L'Oréal Products")
            ax.set_xlabel("Number of Mentions")
            ax.invert_yaxis()
            st.pyplot(fig)
        else:
            st.info("No product mentions found in filtered data")

with tab4:
    st.subheader("Per Video Quality Metrics")
    st.dataframe(
        per_video.sort_values("quality_ratio", ascending=False),
        use_container_width=True
    )
    
    st.subheader("Top Quality Comments")
    top_comments = filtered_df.nlargest(20, "quality_score")[[
        "videoId", "likeCount", "primary_category", "sentiment", 
        "quality_score", "cleaned_comment", "mentions_loreal"
    ]]
    st.dataframe(top_comments, use_container_width=True)
    
    st.subheader("Sample Comments Needing Review")
    low_quality = filtered_df[filtered_df["quality_score"] < 0.4].sample(min(10, len(filtered_df)), random_state=42)[[
        "videoId", "likeCount", "primary_category", "sentiment", 
        "quality_score", "cleaned_comment", "spam_flag"
    ]]
    st.dataframe(low_quality, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**CommentSense** - AI-powered beauty comment analysis for L'Oréal")