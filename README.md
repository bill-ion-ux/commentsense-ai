# 🧠 CommentSense AI  
*AI-powered comment quality analysis for the L’Oréal x Monash Datathon 2025*

---

## 📌 Project Overview  
CommentSense is an **AI-powered solution** designed to analyze the **quality and relevance of social media comments at scale**.  
It introduces a new metric — **Comment Quality Index (CQI)** — that goes beyond likes and shares by evaluating:  
- **Relevance** of comments to the post/product.  
- **Sentiment strength** (positive/neutral/negative).  
- **Category tagging** (skincare, makeup, fragrance, etc.).  
- **Spam detection & filtering**.  

The solution provides **actionable insights** via a **Streamlit dashboard**, helping L’Oréal’s **marketing & product innovation teams** identify which campaigns drive meaningful engagement.  

---

## 🎯 Features  
- ✅ Sentiment Analysis (positive / neutral / negative).  
- ✅ Category Tagging (skincare, makeup, fragrance, etc.).  
- ✅ Spam Detection (filter out bots & irrelevant noise).  
- ✅ **Comment Quality Index (CQI)** scoring.  
- ✅ Interactive Dashboard with KPIs and visual insights.  

## 🛠 Tech Stack  
- **Language**: Python 3.9+  
- **Data Processing**: pandas, numpy  
- **NLP**: spaCy, NLTK, Hugging Face Transformers, sentence-transformers  
- **ML**: scikit-learn, LightGBM  
- **Visualization/Dashboard**: Streamlit, Plotly  

## 🚀 Setup Instructions

1. Clone the repo
   ```bash
   git clone https://github.com/bill-ion-ux/commentsense-ai.git
   cd commentsense-ai
    python -m venv venv
    source venv/bin/activate   # Mac/Linux
    venv\Scripts\activate      # Windows