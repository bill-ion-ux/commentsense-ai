import pandas as pd
import re
import emoji

#-----------------------------------
#|       LOAD THE CSV FILE          |
#-----------------------------------

# df = pd.read_csv("comments5.csv", encoding="utf-8", on_bad_lines="skip")
df = pd.read_csv("/src/commentsense/assets/comments5.csv", encoding="utf-8", on_bad_lines="skip")

print(df.head())
#print(df.info())
#print(df.duplicated().sum())
#print(df.isna().sum())

#-----------------------------------
#|       CLEAN DATA                 |
#-----------------------------------

#   drop missing value in textOriginal column
df = df.dropna(subset=["textOriginal"])
# drop the parentCommentId
df = df.drop(columns=["parentCommentId"])
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)            # remove URLs
    text = re.sub(r"@\w+", "", text)               # remove mentions
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)     # reduced repeated characters
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text) # keep letters, numbers, basic punctuation
    return text.strip()

df["cleaned_text"] = df["textOriginal"].apply(clean_text)
df["cleaned_text"] = df["cleaned_text"].str.replace(r"\s+", " ", regex=True)

