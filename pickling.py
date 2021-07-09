import pickle

def saveModel(vectorizer, classifier_linear):
    pickle.dump(vectorizer, open('models/vectorizer.sav', 'wb'))
    pickle.dump(classifier_linear, open('models/classifier.sav', 'wb'))
    print("Models are Saved")
