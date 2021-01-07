import numpy as np
import pandas as pd
import nltk
import string

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC


''' Read in the data. '''
filepath = '/Users/JasmineW/Desktop/Projects/HumorDetection/data/master_dataset.csv'
data = pd.read_csv(filepath)

from sklearn.model_selection import train_test_split

train_x_ngram, test_x_ngram, train_y_ngram, test_y_ngram = train_test_split(data.Line, data.Funny, test_size = 0.2, random_state = 2)

from sklearn.feature_extraction.text import TfidfVectorizer


tfidf_ngram = TfidfVectorizer(ngram_range = (1, 2))
train_1_ngram = tfidf_ngram.fit_transform(train_x_ngram)
test_1_ngram = tfidf_ngram.transform(test_x_ngram)


train_arr_ngram = train_1_ngram.toarray()
test_arr_ngram = test_1_ngram.toarray()


# SVM w/ngram
svc_ngram = LinearSVC()
svc_ngram.fit(train_arr_ngram, train_y_ngram)
svc_ngram_predicted = svc_ngram.predict(test_arr_ngram)
svc_accuracy_ngram = accuracy_score(test_y_ngram, svc_ngram_predicted) * 100
#print(svc_accuracy_ngram)




''' Returns whether or not a user inputted string is funny. '''
def is_funny(str):
    test = tfidf_ngram.transform([str])
    predicted_val = svc_ngram.predict(test.toarray())
    return predicted_val


print(is_funny("I like to eat"))
