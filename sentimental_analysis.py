# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 15:53:09 2021

@author: OKOK PROJECTS
"""

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.loads(l)
    
import pandas as pd
import gzip
import json

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF('Cell_Phones_and_Accessories_5.json.gz')

df100k = df.head(100000)    

df100k = df100k.rename({'reviewText': 'review'}, axis=1)

df100k['label'] = df100k['overall'].apply(lambda c: 1 if c >3 else (0 if c<3 else 'neut'))

df100k = df100k.drop(['overall', 'reviewTime', 'reviewerID', 'asin', 'reviewerName', 'summary', 'unixReviewTime'], axis=1)

df100k = df100k[df100k['label'] != 'neut']
df100k

df100k['review'].isnull().values.any()

import numpy as np
df100k = df100k.replace(np.nan, '', regex=True)
df100k['review'].isnull().values.any()

df100k['label'].value_counts()

def get_top_data(top_n):
    top_data_df_positive = df100k[df100k['label'] == 1].head(top_n)
    top_data_df_negative = df100k[df100k['label'] == 0].head(top_n)
    top_data_df_small = pd.concat([top_data_df_positive, top_data_df_negative])
    return top_data_df_small

# Function call to get the top 10000 from each sentiment
top_data_df_small = get_top_data(top_n=15000)

# After selecting top few samples of each sentiment
print("After segregating and taking equal number of rows for each sentiment:")
print(top_data_df_small['label'].value_counts())
top_data_df_small.head(10)

df30k = top_data_df_small

import re
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import spacy

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

df30k['new_reviews'] = df30k['review'].apply(lambda x: " ".join(x.lower() for x in x.split()))

df30k['new_reviews'] = df30k['new_reviews'].str.replace('[^\w\s]','')

def space(comment):
    doc = nlp(comment)
    return " ".join([token.lemma_ for token in doc])
df30k['new_reviews']= df30k['new_reviews'].apply(space)
df30k.head(20)

df30k = df30k.drop(['review'], axis=1)
df30k = df30k.rename({'new_reviews': 'review'}, axis=1)
df30k

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df30k.review, df30k.label, test_size=0.2, random_state=32)

from sklearn.feature_extraction.text import TfidfVectorizer
# Create feature vectors
vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)
train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)

import time
from sklearn import svm
from sklearn.metrics import classification_report
# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(train_vectors, y_train)
t1 = time.time()
prediction_linear = classifier_linear.predict(test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1
# results
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))

print(classification_report(y_test,prediction_linear))