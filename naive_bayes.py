import logging
import pandas as pd
import numpy as np
from numpy import random
import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
import csv


df = pd.read_csv('resultcrawl70.csv', sep=',', header=None)



df.rename(columns={0: 'tweet', 1: 'score'}, inplace=True)

print(df['tweet'].apply(lambda x: len(x.split(' '))).sum())


df.loc[df.score > 0.0, 'score'] = 1
df.loc[df.score == 0.0, 'score'] = 0
df.loc[df.score < 0.0, 'score'] = -1

df.loc[df.score == 1 , 'score'] = "positif"
df.loc[df.score == 0, 'score'] = "netral"
df.loc[df.score == -1, 'score'] = "negatif"

df = df[pd.notnull(df['score'])]
my_tags = ['positif', 'netral','negatif']

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english')) #adjective

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text


    
df['tweet'] = df['tweet'].apply(clean_text)
df['tweet'].apply(lambda x: len(x.split(' '))).sum()
df['tweet'] = df.tweet.str.replace('^b','')

df

X = df.tweet
y = df.score
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)



from sklearn.model_selection import KFold 
from sklearn.model_selection import RepeatedKFold
kf = KFold(n_splits=10, random_state=42, shuffle=True)

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score


loop=0
for train_index, test_index in kf.split(X):
    loop+=1
    print("============================ NUMBER of FOLD", loop," ============================")
    X_train, X_test = X[train_index], X[test_index] 
    y_train, y_test = y[train_index], y[test_index]

    nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)

    print('accuracy %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred,target_names=my_tags))
    print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
    print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))
    
# confusion matrix and classification report(precision, recall, F1-score)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))