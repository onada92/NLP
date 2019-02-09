from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('spambase.data').as_matrix() # use pandas for convenience

X = data[1:,:48]
Y = data[1:,-1]

Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

# model = MultinomialNB()
# model.fit(Xtrain,Ytrain)
# print("Classification rate for NB :", model.score(Xtest,Ytest))

model = AdaBoostClassifier()
model.fit(Xtrain,Ytrain)
print("Classification rate for AdaBoostClassifier :", model.score(Xtest,Ytest))
