# Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from pathlib import Path
import os
import re
import html
import string
import unicodedata
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer



######Load Data###### 
dataset = pd.read_csv('udemy_tech.csv')

dataset['Summary'] = dataset['Summary'].fillna('')
dataset['new'] = dataset['Title'] + ' ' + dataset['Summary']

rec = input("enter : ")
rec = [rec]



stop_words = stopwords.words('english')


import unicodedata
def remove_non_ascii(text):
    """Remove non-ASCII characters from list of tokenized words"""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')


def to_lowercase(text):
    return text.lower()


import string
def remove_punctuation(text):
    """Remove punctuation from list of tokenized words"""
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


def replace_numbers(text):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    return re.sub(r'\d+', '', text)


def remove_whitespaces(text):
    return text.strip()


def remove_stopwords(words, stop_words):
   
    return [word for word in words if word not in stop_words]


def stem_words(words):
    """Stem words in text"""
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in words]

from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
def lemmatize_words(words):
    """Lemmatize words in text"""

    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]

def lemmatize_verbs(words):
    """Lemmatize verbs in text"""

    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word, pos='v') for word in words])

def text2words(text):
  return word_tokenize(text)

def normalize_text( text):
    text = remove_non_ascii(text)
    text = remove_punctuation(text)
    text = to_lowercase(text)
    text = replace_numbers(text)
    words = text2words(text)
    words = remove_stopwords(words, stop_words)
    #words = stem_words(words)# Either stem ovocar lemmatize
    words = lemmatize_words(words)
    words = lemmatize_verbs(words)

    return ''.join(words)

def normalize_corpus(corpus):
  return [normalize_text(t) for t in corpus]

nor_new = normalize_corpus(dataset['new'])
nor_input = normalize_corpus(rec)

from keras.preprocessing.text import Tokenizer
tok = Tokenizer(num_words=10000, oov_token='UNK')
tok.fit_on_texts(nor_new+nor_input)
tfidf_ind = tok.texts_to_matrix(nor_new , mode='tfidf')
tfidf_input = tok.texts_to_matrix(nor_input, mode='tfidf')

#####recommender function########
def get_recommendations(title):
    cosine_sim = linear_kernel(tfidf_ind, tfidf_input)
    
    sim_scores = list(enumerate(cosine_sim))

    # Sort the courses based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    sim_scores = sim_scores[0:11]

    # Get the cources indices
    courses_indices = [i[0] for i in sim_scores]
    return courses_indices
   


a = get_recommendations(rec)
print('this is a Title for courses we are recommended for you : \n\n')
for i in a:
    print(dataset['Title'].iloc[i])
print('-----------------------------------------------')
print('this is a Links for courses we are recommended for you : \n\n')
for i in a:
    print(dataset['Link'].iloc[i])

print('-----------------------------------------------')










