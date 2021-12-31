# HumorDetection
In this individual passion project, we trained a machine learning model to accurately detect humor. We created a **10000+ custom dataset** that includes a variety of funny and not funny statements ranging from jokes to wikipedia sentences. To prepare the dataset, we utilized **TFIDF Vectorizer** in order to represent the words as features, as well as, **N-Grams** to account for linguistic patterns in speech that may affect the sentiment of a certain phrase or sentence. Using **Support Vector Classification Model w/ Ngrams**, we were able to achieve a **92% test accuracy**. We also created an interactive web app so you can test out your own funny statements! 

#### Test out the app [here](http://Humor-detection.herokuapp.com)

A possible extension to this project is to utilize the NLP package (NLTK) to better model human speech as well as to train on a more diverse dataset that includes a larger variety of statements (e.i. slang, differing syntax, etc ...)

#### 1. Data Acquisition + Preparation
We chose 5 different datasets: oneliners, short jokes, headlines, wikipedia sentences and proverbs. We labeled the oneliners and short jokes as "funny" and the other three datasets as "not funny" so that there is around 10000 statements for each category (funny and not funny).
#### 2. Data Cleaning + EDA
To clean the data, we got rid of any null values, tokenized each statement, changed all words to lowercase and removed punctuation, symbols or numbers. We also removed any stopwords and lemmatized each word using the **NLTK package**. This was only necessary for the non-ngram models that we trained. In the end, the n-gram models ended up performing better so data cleaning was not super necessary.  
#### 3. Feature Engineering
We used **Term Frequency - Inverse Document Frequency (TF-IDF)**, which is a form of text vectorization, to turn all the words into features with weights depending on its importance in the document. More on TF-IDF [here](https://towardsdatascience.com/text-vectorization-term-frequency-inverse-document-frequency-tfidf-5a3f9604da6d).
#### 4. Train Model 
Using **sci-kit learn** we split the data into into training and test set. Then, we trained the model a total of six times on three different models (w/out n-gram and w/gram for each): **Naive Bayes, Logistic Regression and Support Vector Classification**.  
#### 5. Evaluate Model 
From the table at the bottom of the Jupyter Notebook, the **Support Vector Classification w/ Ngram model** achieved the highest performance with a **test accuracy of 92%**.
#### 6. Host as Web App
We used the **Flask** to connect ML model and utilized **HTML/CSS** to develop and design web app. 

#### Main Files: 
- HumorAnalysis.ipynb (experimenting with different models)
- HumorDetection.py (final SVC model saved to pkl file)
- flask_humor.py (connecting ml model into web app)


