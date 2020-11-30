import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.ensemble import RandomForestClassifier
stop_words = set(stopwords.words('english'))

#reading in the new fold data
df = pd.read_csv('depression.csv')

features = df[['depression', 'text']].columns

pickle.dump(features, open('features.pickle', 'wb'))


# remove noise from text
def text_preprocess(tweet):
    tweet = tweet.lower()
    # remove hyperlinks
    tweet = re.sub(r"http?:\S+|www\S+|https?:\S+", '', tweet)
    # remove user @ mentions
    tweet = re.sub(r'@[a-z0-9]+', '', tweet)
    # remove # symbols
    tweet = re.sub(r'#', '', tweet)
    # remove rt
    tweet = re.sub(r'rt[\s]+', '', tweet)
    # remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # remove stopwords
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens if not w in stop_words]
    # word normalization
    # stemmer = PorterStemmer()
    # stemmed_words = [stemmer.stem(w) for w in filtered_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(w, pos='a') for w in filtered_words]

    return " ".join(lemmatized_words)

pre_text = df['text'].apply(text_preprocess)
pickle.dump(pre_text, open('pre_text.pickle', 'wb'))

tfidf = TfidfVectorizer()
vector = tfidf.fit_transform(pre_text)
pickle.dump(vector, open('vector.pickle', 'wb'))

X, y = vector, df.loc[:, 'depression']
clf = RandomForestClassifier()
clf.fit(X,y)
pickle.dump(clf, open('model.pickle', 'wb'))