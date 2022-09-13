from email import header
from re import I
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model
from itertools import zip_longest
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from nltk.stem import WordNetLemmatizer 
import random 
import string
import nltk
import tensorflow as tf 
from fuzzywuzzy import fuzz
import json
import streamlit as st
import pandas as pd

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text): 
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in string.punctuation]
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

data_points = open("cell_dataset.json", "r")
data_points = json.loads(data_points.read())
#matching_score = fuzz.token_set_ratio(row, product) 
input_v = st.text_input('search HS Code here :') 
cleaned_value = clean_text(input_v)
c_x = ", ".join(cleaned_value).replace(",","")
#print(c_x)
st.title('HS Code Output')
hs_data = []
for item,value in data_points.items():
    if type(value) == dict:
        for i,v in value.items():
            if type(v) == dict:
                for i2,v2 in v.items():
                    for q in v2:
                        matching_score = fuzz.token_set_ratio(c_x, q) 
                        if matching_score>=70:
                            #print(item,i,"-------->>",q)
                            hs_data.append({"HS CODE":item, "HS Code Description":i, "HS Code Heading":i2})



super_dict = {}
for d in hs_data:
    for k, v in d.items():  # d.items() in Python 3+
        super_dict.setdefault(k, []).append(v)


df = pd.DataFrame.from_dict(super_dict)

if len(df) >1:
    test = df.iloc[0]
    new = pd.DataFrame(test)
    new = new.T
    new.set_index("HS CODE",inplace=True)
    print(new)
    st.table(new)

