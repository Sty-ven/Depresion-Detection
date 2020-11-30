import pickle
import numpy as np
import pandas
from pydantic import BaseModel

import uvicorn
from fastapi import FastAPI

from sklearn.ensemble import RandomForestClassifier



# initialize files
clf = pickle.load(open('model.pickle', 'rb'))
pre_preprocess = pickle.load(open('pre_text.pickle', 'rb'))
vector = pickle.load(open('vector.pickle', 'rb'))
features = pickle.load(open('features.pickle', 'rb'))


# @app.get("/")
# def index():
#     return{"message": "Hello World"}

# class Data(BaseModel):
#     text: str

app = FastAPI()
@app.post("/predict")
def predict(data: str):

    to_predict = [data[feature] for feature in features]

    tokenized_features = pre_preprocess.transform(to_predict)
    vectorized_features = vector.transform(tokenized_features)
    to_predict = np.array(to_predict + vectorized_features)

    prediction = clf.predict(to_predict)

    return {"prediction": int(prediction[0])}
#
#
# text = 'the boy is sad'
#
# predict(text)