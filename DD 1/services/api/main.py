#main framework for the app
#Data handling
import pandas as pd
import numpy as np
import joblib
from pydantic import BaseModel
import pickle

#Server
from fastapi import FastAPI
import uvicorn

#Model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

model = joblib.load(open('model.plk', 'rb'))
vector = joblib.load(open('vector.plk', 'rb'))


app = FastAPI()


@app.get('/')
async def index():
    return {'text': 'Hello there, welcome to Depression Detection!'}

@app.get('/items')
async def get_items(name:str):
    return {'name': name}


#@app.get('/predict/')
#async def predict_depression(tweet:str = Query(None, min_length = 3, max_length = 25)):
    #vectorized_tweet = vector.transform([tweet]).toarray()
    #prediction = model.predict(vectorized_tweet)
    #if prediction[1] == 1:
        #result = 'Depressed'
    #else:
        #result = 'Not depressed'
    #return {'User is': result}


@app.post("/predict/{tweet}")
async def predict_depression(tweet:str):

    vectorized_tweet = vector.transform([tweet]).toarray()
    prediction = model.predict(vectorized_tweet)
    if prediction == 0:
        result = 'Not Depressed'
    else:
        result = 'Depressed'
    return {'User is': result}


    #data = data.dict()
    #data extraction
    #to_predict = data[feature]

    #vector transformer
    #to_predict = vector.transform(to_predict.ravel())
    #prediction = clf.predict(to_predict.ravel())
    #return {"prediction": int(prediction)}

if __name__=='__Main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
