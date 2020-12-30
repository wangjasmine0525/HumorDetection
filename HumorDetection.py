import numpy as np
import pandas as pd
import nltk
import string
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style


''' Read in the data. '''
oneliners = pd.read_pickle('data/humorous_oneliners.pickle') # supposed to be funny
proverbs = pd.read_pickle('data/proverbs.pickle') # not funny
reuters_headlines = pd.read_pickle('data/reuters_headlines.pickle') # not funny
wiki_sentences = pd.read_pickle('data/wiki_sentences.pickle') # not funny
shortjokes = pd.read_csv('data/shortjokes.csv') # supposed to be funny
shortjokes_df = shortjokes.drop(['ID'], axis = 1).rename(columns = {"Joke": "Line"}).sample(frac = 0.02, replace = False)
shortjokes_df['Funny'] = 1


''' Print length of data. '''
#print("oneliners length: ", len(oneliners))
#print("shortjokes length: ", len(shortjokes_df))
#print("proverbs length: ", len(proverbs))
#print("reuters length: ", len(reuters_headlines))
#print("wiki length: ", len(wiki_sentences))


''' Create the master pandas dataframe. '''
oneliner_df = pd.DataFrame(data = {'Line': oneliners, 'Funny': 1})
proverbs_df = pd.DataFrame(data = {'Line': proverbs, 'Funny': 0})
reuters_headlines_df = pd.DataFrame(data = {'Line': reuters_headlines, 'Funny': 0})
wiki_sentences_df = pd.DataFrame(data = {'Line': wiki_sentences, 'Funny': 0})

# data is a 2 column dataframe with one column containing the joke and the other denoting whether it is funny(1) or not(0)
data = pd.concat([oneliner_df, proverbs_df, reuters_headlines_df, wiki_sentences_df, shortjokes_df])

# we only want to randomly sample half of our dataframe
data = data.sample(frac = 0.25, replace = False)
#print(data.head(5))

#sns.catplot(x='Funny', kind = 'count', data = data)

# Tokenize
''' Splits elements in the Line series into lists of words. '''
def Tokenize(string):
    lst = string.split()
    return lst

data["Line"] = data["Line"].apply(Tokenize)


# lower case
''' Takes in a tokenized list and makes all words lower case. '''
def lower_case(lst):
    lowered = []
    for i in lst:
        lowered.append(i.lower())
    return lowered

data["Line"] = data["Line"].apply(lower_case)


# remove punctuation and symbols
''' Removed punctuation from tokenized list. '''
def remove_punc(lst):
    removed = []
    for i in lst:
        stripped = i.strip(string.punctuation)
        if stripped is not '':
            removed.append(stripped)
    return removed

data["Line"] = data["Line"].apply(remove_punc)


# remove numbers
''' Removed numbers from tokenized list. '''
def remove_numbers(lst):
    numbers = "0123456789"
    no_numbers = []
    for i in lst:
        i = ''.join([j for j in i if not j.isdigit()])
        no_numbers.append(i)
    return no_numbers

data["Line"] = data["Line"].apply(remove_numbers)


# remove stopwords
from nltk.corpus import stopwords
nltk.download('stopwords')
english_stopwords = stopwords.words("english")

''' Removes stopwords from tokenized list. '''
def remove_stopwords(lst):
    return [i for i in lst if not i in english_stopwords]

data["Line"] = data["Line"].apply(remove_stopwords)


# Lemmatization
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
wordnet_lemmatizer = WordNetLemmatizer()

''' Lemmatizes words in a tokenized list'''
def lemmatize(lst):
    lemmatized_words = []
    for i in lst:
        lemmatized_words.append(wordnet_lemmatizer.lemmatize(i, pos = "v"))
    return lemmatized_words

data["Line"] = data["Line"].apply(lemmatize)



data['Line'] = data['Line'].apply(lambda x: ''.join(i+' ' for i in x))


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
