# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 16:51:17 2021

@author: annai
"""
import pandas as pd

#recompile data from the file to make pandas see all the tokens&tags in it
list1 = list()
list2 = list()
list3 = list()

fil = open('dataset_fine_tuning.csv', 'r', encoding='utf8')
lines = fil.readlines()
for line in lines:
  line = line.split()
  list1.append(line[0])
  list2.append(line[1])
  list3.append(line[2])
  
data = pd.DataFrame(list(zip(list1, list2, list3)),
                columns =['id', 'token', 'tag'])

#convert data to numpy arrays
token_column = data.loc[:,'token']
token = token_column.values
tag_column = data.loc[:,"tag"]
tag = tag_column.values
print(type(tag))

#train five dummy classifiers and print their f1-score
from sklearn.dummy import DummyClassifier
dummy1 = DummyClassifier(strategy='stratified', random_state = 100, constant = None)
dummy1.fit(token, tag)
tag_predicted1 = dummy1.predict(token)
from sklearn.metrics import f1_score
print("Dummy Model 1, strategy: stratified, f1-score {0:.2f}".format(f1_score(tag, tag_predicted1, pos_label = 'YES')))

dummy2 = DummyClassifier(strategy='most_frequent', random_state = 100, constant = None)
dummy2.fit(token, tag)
tag_predicted2 = dummy2.predict(token)
print("Dummy Model 2, strategy: most_frequent, f1-score {0:.2f}".format(f1_score(tag, tag_predicted2, pos_label = 'YES')))

dummy3 = DummyClassifier(strategy='prior', random_state = 100, constant = None)
dummy3.fit(token, tag)
tag_predicted3 = dummy3.predict(token)
print("Dummy Model 3, strategy: prior, f1-score {0:.2f}".format(f1_score(tag, tag_predicted3, pos_label = 'YES')))

dummy4 = DummyClassifier(strategy='uniform', random_state = 100, constant = None)
dummy4.fit(token, tag)
tag_predicted4 = dummy4.predict(token)
print("Dummy Model 4, strategy: uniform, f1-score {0:.2f}".format(f1_score(tag, tag_predicted4, pos_label = 'YES')))

dummy5 = DummyClassifier(strategy='constant', random_state = 100, constant = 'YES')
dummy5.fit(token, tag)
tag_predicted5 = dummy5.predict(token)
print("Dummy Model 5, strategy: constant, f1-score {0:.2f}".format(f1_score(tag, tag_predicted5, pos_label = 'YES')))
