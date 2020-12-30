import numpy as np
import pandas as pd
import nltk
import string
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style


''' Read in the data. '''
data = pd.read_csv('data/master_dataset.csv')


from sklearn.model_selection import train_test_split

train_x_tfidf, test_x_tfidf, train_y_tfidf, test_y_tfidf = train_test_split(data.Line, data.Funny, test_size = 0.2, random_state = 2)

from sklearn.feature_extraction.text import TfidfVectorizer

# without ngram
tfidf = TfidfVectorizer()
train_1_tfidf = tfidf.fit_transform(train_x_tfidf)
test_1_tfidf = tfidf.transform(test_x_tfidf)

# with ngram
tfidf_ngram = TfidfVectorizer(ngram_range = (1, 2))
train_1_ngram = tfidf_ngram.fit_transform(train_x_tfidf)
test_1_ngram = tfidf_ngram.transform(test_x_tfidf)



# without ngram
train_arr_tfidf = train_1_tfidf.toarray()
test_arr_tfidf = test_1_tfidf.toarray()

# with ngram
train_arr_ngram = train_1_ngram.toarray()
test_arr_ngram = test_1_ngram.toarray()



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Logistic Regression w/out ngram
logreg_tfidf = LogisticRegression()
logreg_tfidf.fit(train_arr_tfidf, train_y_tfidf)
logreg_tfidf_predicted = logreg_tfidf.predict(test_arr_tfidf)
logreg_accuracy_tfidf = accuracy_score(test_y_tfidf, logreg_tfidf_predicted) * 100
print("Accuracy: ", logreg_accuracy_tfidf)

print("------------------------------------------")

# Logistic Regression w/ngram
logreg_ngram = LogisticRegression()
logreg_ngram.fit(train_arr_ngram, train_y_tfidf)
logreg_ngram_predicted = logreg_ngram.predict(test_arr_ngram)
logreg_accuracy_ngram = accuracy_score(test_y_tfidf, logreg_ngram_predicted) * 100
print("w/ngram Accuracy: ", logreg_accuracy_ngram)



from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Naive Bayes w/out ngram
nb_tfidf = MultinomialNB()
nb_tfidf.fit(train_arr_tfidf, train_y_tfidf)
nb_tfidf_predicted = nb_tfidf.predict(test_arr_tfidf)
nb_accuracy_tfidf = accuracy_score(test_y_tfidf, nb_tfidf_predicted) * 100
print("Accuracy: ", nb_accuracy_tfidf)

print("------------------------------------------")

# Naive Bayes w/ngram\
nb_ngram = MultinomialNB()
nb_ngram.fit(train_arr_ngram, train_y_tfidf)
nb_ngram_predicted = nb_ngram.predict(test_arr_ngram)
nb_accuracy_ngram = accuracy_score(test_y_tfidf, nb_ngram_predicted) * 100
print("w/ngram Accuracy: ", nb_accuracy_ngram)




from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import f1_score, accuracy_score

#SVM w/out ngram
svc_tfidf = LinearSVC()
svc_tfidf.fit(train_arr_tfidf, train_y_tfidf)
svc_tfidf_predicted = svc_tfidf.predict(test_arr_tfidf)
svc_accuracy_tfidf = accuracy_score(test_y_tfidf, svc_tfidf_predicted) * 100
print("Accuracy: ", svc_accuracy_tfidf)

print("------------------------------------------")

# SVM w/ngram
svc_ngram = LinearSVC()
svc_ngram.fit(train_arr_ngram, train_y_tfidf)
svc_ngram_predicted = svc_ngram.predict(test_arr_ngram)
svc_accuracy_ngram = accuracy_score(test_y_tfidf, svc_ngram_predicted) * 100
print("w/ngram Accuracy: ", svc_accuracy_ngram)


results = pd.DataFrame({'Model': ['Logsitic Regression', 'Naive Bayes', 'SVC'],
                        'TF-IDF': [logreg_accuracy_tfidf, nb_accuracy_tfidf, svc_accuracy_tfidf],
                        'w/ N-Grams': [logreg_accuracy_ngram, nb_accuracy_ngram, svc_accuracy_ngram],
})
results = results.sort_values(by = 'Model', ascending = False)
print(results)
