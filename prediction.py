import pickle

'''
from google.cloud import language_v1
from google.cloud.language_v1 import enums
from google.cloud.language_v1 import types
from google.cloud.speech_v1 import enums
'''

def predictFromModel(text):
    vectorizer = pickle.load(open('models/vectorizer.sav', 'rb'))
    classifier = pickle.load(open('models/classifier.sav', 'rb'))
    text_vector = vectorizer.transform([text])
    result = classifier.predict(text_vector)
    return result[0]
