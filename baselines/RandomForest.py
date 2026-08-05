#code for this baseline was partially taken from repository by Shoumik (2018) at: <https://www.kaggle.com/shoumikgoswami/ner-using-random-forest-and-crf/notebook> 

import nltk
import pandas as pd
import numpy as np

# we use cross validation because the data set is small and imbalanced
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
data = pd.read_csv('dataset_fine_tuning.csv', delimiter = '\t')

# load data to lists first to access all the tokens with no problems
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

# pos tag for feature representation
new_str = [str(x) for x in token]
new_str = nltk.pos_tag(new_str)

lst_for_merge = list()

for el in new_str:
    lst_for_merge.append(el[1])
    
se = pd.Series(lst_for_merge)
data['POS'] = se.values

# use one-hot encoding for feature (pos tag) represenation
one_hot = pd.get_dummies(data['POS'])

# drop column POS as it is now encoded
data = data.drop('POS',axis = 1)

# join the encoded data
data = data.join(one_hot)

# join digits in encoding in one column via joining their headers 
data["new_pos"] = data["$"].map(str)+ data["''"].map(str)+data["("].map(str)+data[")"].map(str)+data["("].map(str)+data[")"].map(str)+data[","].map(str)+data["."].map(str)+data[":"].map(str)+data["CC"].map(str)+data["CD"].map(str)+data["DT"].map(str)+data["EX"].map(str)+data["FW"].map(str)+data["IN"].map(str)+data["JJ"].map(str)+data["JJR"].map(str)+data["JJS"].map(str)+data["MD"].map(str)+data["NN"].map(str)+data["NNP"].map(str)+data["NNPS"].map(str)+data["NNS"].map(str)+data["PDT"].map(str)+data["POS"].map(str)+data["PRP"].map(str)+data["PRP$"].map(str)+data["RB"].map(str)+data["RBR"].map(str)+data["RBS"].map(str)+data["RP"].map(str)+data["SYM"].map(str)+data["TO"].map(str)+data["UH"].map(str)+data["VB"].map(str)+data["VBD"].map(str)+data["VBG"].map(str)+data["VBN"].map(str)+data["VBP"].map(str)+data["VBZ"].map(str)+data["WDT"].map(str)+data["WP"].map(str)+data["WP$"].map(str)+data["WRB"].map(str)+data["``"].map(str)

# make new data frame with one-hot ecoding as one column and write to file
header = ["id", "token", "tag", "new_pos"]
data.to_csv('PV_DATASET_COLAB1.csv', columns = header, index = False, sep = '\t')

#open new file and load data
list1 = list()
list2 = list()
list3 = list()
list4 = list()
fil = open('PV_DATASET_COLAB1.csv', 'r', encoding='utf8')
lines = fil.readlines()
for line in lines:
  line = line.split()
  list1.append(line[0])
  list2.append(line[1])
  list3.append(line[2])
  list4.append(line[3])
  
data = pd.DataFrame(list(zip(list1, list2, list3, list4)),
                columns =['id', 'token', 'tag', 'POS'])

# convert column with parts of speech to nparray - 
# previously POS were encoded as one-hot encoding. 
# The format turned out to be too long for RandomForests 
# classifier, so we will shorten the values 
pos_column = data.loc[:,"POS"]
pos_column = pos_column.values

# introduce this to avoid adding header 'POS' to the numeric data for further processing
lst = [1000000]

# turn one-hot encoding (str) into int and add to list
for i in pos_column[1:]:
    i = int(i, base = 2)
    lst.append(i)


# add list with shortened values as extra column to the dataframe
se = pd.Series(lst)
data['POS2'] = se.values    

# define features for the classifier: format of the token (letters/digits) + use POS as a feature    
def feature_map(word, pos):
    return np.array([int(word.isdigit()),  int(word.isalpha()), pos])

words = [feature_map(w, p) for w, p in zip(data["token"].values.tolist(),
                                            data["POS2"].values.tolist())]


tags = data["tag"].values.tolist()
tags = tags[1:]



# define model
pred = cross_val_predict(RandomForestClassifier(n_estimators=20),X=words[1:], y=tags, cv=5)
print("RF model: f1-score {0:.2f}".format(f1_score(tags, pred, pos_label = 'YES')))


