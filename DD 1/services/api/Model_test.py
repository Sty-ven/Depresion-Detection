import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

stop_words = set(stopwords.words('english'))

data = pd.read_csv("depression.csv")
data.head()

data.drop(["time", "tweet_id"], axis = 1, inplace = True)

def text_preprocess(tweet):
    tweet = tweet.lower()
    #remove hyperlinks
    tweet = re.sub(r"http?:\S+|www\S+|https?:\S+", '', tweet)
    #remove user @ mentions
    tweet = re.sub(r'@[a-z0-9]+','', tweet)
    #remove # symbols
    tweet = re.sub(r'#', '', tweet)
    #remove rt
    tweet = re.sub(r'rt[\s]+', '', tweet)
    #remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    #remove stopwords
    tweet_tokens = word_tokenize(tweet)  #list of tokens
    filtered_words = [w for w in tweet_tokens if not w in stop_words]   #list of tokens
    #word normalization
    #stemmer = PorterStemmer()
    #stemmed_words = [stemmer.stem(w) for w in filtered_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(w, pos='a') for w in filtered_words]  #list of tokens
    result = " ".join(lemmatized_words)   #return a clean sentence
    
    return result

data.text = data['text'].apply(text_preprocess)

def get_feature_vector(data_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit(data_fit)
    joblib.dump(vector, open('vector.plk', 'wb'))
    return vector

tf_vector = get_feature_vector(data['text'])
X = tf_vector.transform(data['text'])
y = data['depression']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                        random_state=10)

model = RandomForestClassifier(n_estimators=200)
clf = model.fit(X_train,y_train)
joblib.dump(clf, open('model.plk', 'wb'))
y_pred = model.predict(X_test)
print(accuracy_score(y_pred, y_test))