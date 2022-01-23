

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
 
main_data = pd.read_csv('card_data.csv')

#put every data point into split()
#find the gini of the column
#find the gini of the true array
#if information gain is more the best gain, 
#      the data entry becomes the question
#      the best_gain is updates

headers = ["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","Amount"]

main_data.drop(["V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28"], inplace = True, axis = 1)
main_data["Index"] = range(len(main_data.Class))


Y = main_data["Class"]
X = main_data.drop(["Class"], axis= 1)
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size= 0.3, random_state= 5)

data = X_test



#CALCULATE LOWEST WEIGHTED GINI INDEX
# Ginie Index formula = 1 - SUM(p^2)
# p = (number of entrys belonging to a class in that branch)/(number of entrys in all classes in that branch)
# weighted gini index = SUM((entrys in branch)/(entrys of node) * (gini index of branch))

def gini(series):
    counts = class_counts(series)
    result = 1
    for instance in counts:
        probability = counts[instance] / float(len(series))
        result -= probability**2
    return(result)
    

def class_counts(series):
    counts = {}
    for entry in series:
        label = entry
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return(counts)

#generate questions to ask
class Question:
    def __init__(self, col_name, row_index):
        self.col_name = col_name
        self.row_index = row_index

    def match(self, example):
        val = example[self.col_name]
        if isinstance(val, int) or isinstance(val, float):
            print(headers[self.col_name])
            print("more than")
            print(self.row_index)
            return(val >= self.row_index)
        else:
            print(headers[self.col_name])
            print("less than")
            print(self.row_index)
            return val == self.row_index

def partition(data_point, column):
    # is entry thing more than the data_point
    true_data = data[data[column] >= data_point]
    false_data = data[data[column] < data_point]
    return(true_data, false_data)

def partition2(data_point, column):
    # is entry thing more than the data_point
    
    true_data = []
    false_data = []
    for entry in column:
        if entry >= data_point:
            true_data.append(entry)
        else:
            false_data.append(entry)

    return(true_data, false_data)

#for column in headers:
 #   for entry in data[column]:
  #      partition(entry, column)

def split(column):
    best_gain = 0
    entry_question = None
    curr_gini_index = gini(data[column])
    for entry in data[column]:
        true_data, false_data = partition(entry, column)
        if(len(true_data) > 0) and (len(false_data) > 0):

            gain = information_gain(true_data, false_data, curr_gini_index)

            if gain >= curr_gini_index:
                cur_gini_index = gain
                entry_question = entry
    return(best_gain, entry_question)

def split2(column):
    best_gain = 0
    entry_question = None
    cur_gini_index = gini(column)
    for entry in column:
        true_data, false_data = partition2(entry, column)
        if(len(true_data) > 0) and (len(false_data) > 0):

            gain = information_gain(true_data, false_data, cur_gini_index)

            if gain >= cur_gini_index:
                cur_gini_index = gain
                entry_question = entry
    return(best_gain, entry_question)



def information_gain(true, false, score):
    prob = float(len(true)) / (len(true) + len(false))
    new_score = score - prob * gini(true) - (1-prob) * gini(false)
    return(new_score)

print(split("Amount"))

# create Decision tree using recursion

def Tree(Column):
    if isinstance(Column, str):
  
        gain, question = split(Column)
        if gain == 0:
            return Leaf_node(Column)
        true_data, false_data = partition(Column, question)

        true = Tree(true_data) #!change name input to array input
        false = Tree(false_data) #!change name input to array input

    if isinstance(Column, list):
        gain, question = split(Column)
        if gain == 0:
            return Leaf_node(Column)
        true_data, false_data = partition(Column, question)

        true = Tree(true_data) 
        false = Tree(false_data) 

    return(Decision_node(question, true, false))

class Leaf_node:
    def __init__(self, Column):
        self.predictions = class_counts(data[Column])

class Decision_node:
    def __init__(self, question, true, false):
        self.question = question
        self.true = true
        self.false = false

for name in headers:
    Tree(name)

#train decision tree against X_train, Y-train
train_data = X_train

def sort(row, node):
    if isinstance(node, Leaf_node):
        return node.predictions
    
