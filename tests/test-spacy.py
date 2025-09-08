import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK resources
# run once
# nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Text to process
# text = "SpaCy is great for fast NLP, while NLTK provides a wide range of linguistic tools."
text = "Dear Amazon, last week I ordered an Optimus Prime action figure from your online store in Germany. Unfortunately, when I opened the package, I discovered to my horror that I had been sent an action figure of Megatron instead! As a lifelong enemy of the Decepticons, I hope you can understand my dilemma. To resolve the issue, I demand an exchange of Megatron for the Optimus Prime figure I ordered. Enclosed are copies of my records concerning this purchase. I expect to hear from you soon. Sincerely, Bumblebee."


# spaCy processing
doc = nlp(text)
tokens = [token.text for token in doc]

# NLTK stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

# NLTK stemming
stemmer = PorterStemmer()
stemmed = [stemmer.stem(word) for word in filtered_tokens]

print("Tokens:", tokens)
print("Filtered:", filtered_tokens)
print("Stemmed:", stemmed)

# Token is basically splitting text by every spaces
# Good for full context NLP (dependency parsing)

# Filter is basically keyword-based work.
# It just removes the common stopwords("is", "for", "a")

# Stemmed is basically the root word.
# Not necessarily correct, but root
# Good for ML model that only need word roots.
