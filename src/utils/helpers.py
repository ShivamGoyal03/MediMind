"""
Contains helper functions for cleaning and preprocessing data. Perfoms the following functions:
- Clean text data
- Lemmatize text data
- Tokenize text data
- Remove stopwords from text data
- Remove punctuation from text data
- Removing URLs from text data
"""

import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    """
    Cleans the input text by performing the following operations:
    1. Lowercasing
    2. Removing URLs
    3. Removing punctuation
    4. Removing numbers
    5. Tokenization
    6. Removing stop words
    7. Lemmatization

    Args:
        text: str: input text

    Returns:
        str: cleaned text
    """
    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)