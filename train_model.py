from sklearn.feature_extraction.text import TfidfVectorizer
import time
from sklearn import svm
from sklearn.metrics import classification_report
import pickling
import pandas as pd
'''
def getData():
    trainDataFile = pd.read_csv("data/train.csv")
    testDataFile = pd.read_csv("data/test.csv")
    return trainDataFile, testDataFile
'''

def getData():
    #            Training data
    #trainDataFile = pd.read_csv("data/electronics_train.csv")
    df = pd.read_csv(r"data\electronics_train.csv")
    #print(a)
    #a.dropna(axis="columns", how="any")
    #a = a.dropna(axis="Label", how="any")
    df = df.dropna(axis=0, subset=['Label'])
    trainDataFile = df
    
    #             Test data
    #testDataFile = pd.read_csv("data/electronics_test.csv")
    df = pd.read_csv(r"data\electronics_test.csv")
    #print(a)
    #a.dropna(axis="columns", how="any")
    #a = a.dropna(axis="Label", how="any")
    df = df.dropna(axis=0, subset=['Label'])
    testDataFile = df
    print('file read was completed')
    return trainDataFile, testDataFile



trainData, testData = getData()

vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)

train_vectors = vectorizer.fit_transform(trainData['Content'])
test_vectors = vectorizer.transform(testData['Content'])
print('Now doing the training')
classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(train_vectors, trainData['Label'])
t1 = time.time()
print('Now doing the testing')
prediction_linear = classifier_linear.predict(test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1
pickling.saveModel(vectorizer, classifier_linear)

print("Results for SVC(kernel=linear)")
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
report = classification_report(testData['Label'], prediction_linear, output_dict=True)
print('positive: ', report['pos'])
print('negative: ', report['neg'])
