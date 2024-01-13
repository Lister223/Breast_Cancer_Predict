# -*- coding: UTF-8 -*-
import pickle
import gzip
import pandas


# 載入模型
with gzip.open('./model/Logistic_breastcancer.pgz', 'rb') as f:
    Logistic = pickle.load(f)
# 載入標準化參數
with open('./model/scaler.pkl', 'rb') as g:
    scaler = pickle.load(g)

def predict(input):
    data = scaler.transform(input)
    pred = Logistic.predict(data)
    return pred