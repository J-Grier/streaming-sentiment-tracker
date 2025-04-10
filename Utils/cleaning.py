# utils/cleaning.py

import re
import gensim
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Basic text cleaner
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)  # Remove links
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)      # Remove numbers
    return text.lower().strip()

# Tokenizer + lemmatizer for LDA
def preprocess(text, custom_stopwords=None):
    lemmatizer = WordNetLemmatizer()
    if custom_stopwords is None:
        custom_stopwords = set(stopwords.words('english')) | {
            'like', 'get', 'one', 'really', 'even', 'know', 'would', 'ive', 'yeah',
            'he', 'she', 'it', 'they', 'we', 'you', 'him', 'her', 'them', 'us', 'our', 'his', 'their',
            'season', 'episode', 'show', 'im', 'hes', 'shes', 'thats', 'is', 'think'
        }

    tokens = gensim.utils.simple_preprocess(text)
    return [lemmatizer.lemmatize(token) for token in tokens if token not in custom_stopwords]
