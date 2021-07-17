from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
from sklearn.model_selection import train_test_split
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
df100k = df100k.drop(['overall',  'reviewTime', 'reviewerID', 'asin', 'reviewerName', 'summary', 'unixReviewTime'], axis=1)
df100k = df100k[df100k['label'] != 'neut']

df100k
df100k['review'].isnull().values.any()
import numpy as np
df100k = df100k.replace(np.nan, '', regex=True)
df100k['review'].isnull().values.any()
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
X = df30k.review.values
Y = df30k.label.values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=32)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectorizer.fit(x_train)
X_train = vectorizer.transform(x_train)
X_test= vectorizer.transform(x_test)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print("Accuracy: ", accuracy_score(y_test, predictions))

model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model.summary()
def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.loads(l)
    
import pandas as pd
import json
import gzip

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



df.head()

df100k = df.head(100000)
df100k

df100k = df100k.rename({'reviewText': 'review'}, axis=1)

df100k['label'] = df100k['overall'].apply(lambda c: 1 if c >3 else (0 if c<3 else 'neut'))

df100k = df100k.drop(['overall',  'reviewTime', 'reviewerID', 'asin',  'reviewerName', 'summary', 'unixReviewTime'], axis=1)

df100k = df100k[df100k['label'] != 'neut']

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
print(top_data_df_small.head(10))
train, test = train_test_split(top_data_df_small, test_size=0.3, random_state=42, shuffle=True)
train = train.rename({'review': 'DATA_COLUMN', 'label': 'LABEL_COLUMN'}, axis=1)
print(train.head())
train['DATA_COLUMN'].isnull().sum().sum()
train['DATA_COLUMN'].fillna("", inplace = True)

train['DATA_COLUMN'].isnull().sum().sum()
test = test.rename({'review': 'DATA_COLUMN', 'label': 'LABEL_COLUMN'}, axis=1)
print(test.head())
test['DATA_COLUMN'].isnull().sum().sum()

def convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN): 
  train_InputExamples = train.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this case
                                                          text_a = x[DATA_COLUMN], 
                                                          text_b = None,
                                                          label = x[LABEL_COLUMN]), axis = 1)

  validation_InputExamples = test.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this case
                                                          text_a = x[DATA_COLUMN], 
                                                          text_b = None,
                                                          label = x[LABEL_COLUMN]), axis = 1)
  
  return train_InputExamples, validation_InputExamples

  train_InputExamples, validation_InputExamples = convert_data_to_examples(train, 
                                                                           test, 
                                                                           'DATA_COLUMN', 
                                                                           'LABEL_COLUMN')

import tensorflow as tf
def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = [] # -> will hold InputFeatures to be converted later

    for e in examples:
        # Documentation is really strong for this method, so please take a look at it
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length, # truncates if len(s) > max_length
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True, # pads to the right by default # CHECK THIS for pad_to_max_length
            truncation=True
        )

        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],
            input_dict["token_type_ids"], input_dict['attention_mask'])

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )


DATA_COLUMN = 'DATA_COLUMN'
LABEL_COLUMN = 'LABEL_COLUMN'

train_InputExamples, validation_InputExamples = convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN)

train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
train_data = train_data.shuffle(100).batch(32).repeat(2)

validation_data = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)
validation_data = validation_data.batch(32)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

model.fit(train_data, epochs=2, validation_data=validation_data)

model.save('bert_model')

from tensorflow import keras

new_model = keras.models.load_model('bert_model')
new_model.summary()