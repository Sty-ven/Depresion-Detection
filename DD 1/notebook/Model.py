import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import wordcloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle


#global parameter
stop_words = set(stopwords.words('english'))


# Load data and save indices of columns
data = pd.read_csv("C:/Users/USER/Desktop/project/Depression_Detection/dep_api/depression.csv")
data.drop(["time", "tweet_id"], axis = 1, inplace = True)
features = data['text']
pickle.dump(features, open('features.pickle', 'wb'))

def text_preprocess(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"http?:\S+|www\S+|https?:\S+", '', tweet)
    tweet = re.sub(r'@[a-z0-9]+','', tweet)
    tweet = re.sub(r'#', '', tweet)
    tweet = re.sub(r'rt[\s]+', '', tweet)
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens if not w in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(w, pos='a') for w in filtered_words]
    
    return " ".join(lemmatized_words)
data.text = data['text'].apply(text_preprocess)

def get_feature_vector(data_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit(data_fit)
    return vector
tf_vector = get_feature_vector(np.array(data.iloc[:, 1]).ravel())
pickle.dump(tf_vector, open('vector.pickle', 'wb'))

X = tf_vector.transform(np.array(data.iloc[:, 1]).ravel())
y = np.array(data.iloc[:, 0]).ravel()
model = RandomForestClassifier(n_estimators=200)
clf = model.fit(X,y)
pickle.dump(clf, open('model.pickle', 'wb'))
