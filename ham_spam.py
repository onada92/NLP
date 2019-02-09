import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier

from sklearn.svm import SVC
from wordcloud import WordCloud

data = pd.read_csv('spam.csv', encoding = 'ISO-8859-1')
data = data[['v1','v2']]

data.columns = ['labels','data']
data['b_label'] = data['labels'].map({'ham': 0, 'spam' : 1})

Y = data.b_label

# tfidf = TfidfVectorizer()
# X = tfidf.fit_transform(data.data)

count_vectorizer = CountVectorizer()
X = count_vectorizer.fit_transform(data['data'])


Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y, test_size = 0.33)

model = AdaBoostClassifier()
model.fit(Xtrain,Ytrain)

print("Train score: ", model.score(Xtrain, Ytrain))
print("Test score: ", model.score(Xtest,Ytest))





