import pandas as pd
from collections import Counter
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

data = pd.read_csv('card_data.csv')

(data.head(10)) 

(data.isnull().values.any()) 

(data["Amount"].describe()) 

is_fraud = len(data[data.Class != 0]) 

not_fraud = len(data[data.Class == 0])

pfraud = (is_fraud / len(data)) * 100

count_classes = data.value_counts(data["Class"])

count_classes.plot(kind = "bar", rot = 0)
plt.title("Genuine and Fraud Occurances")
plt.ylabel("# of Occurances")
plt.xticks(range(2), ["Genuine", "Fraud"])
#plt.show()


scaler = StandardScaler()
data["NormalizedAmount"] = scaler.fit_transform(data["Amount"].values.reshape(-1, 1))
data.drop(["Amount", "Time"], inplace= True, axis= 1)
Y = data["Class"]
X = data.drop(["Class"], axis= 1)
# fit - gives model mean and variance 
# transform - scales model using information from fit
# preform on sample data and have model apply to test data

(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size= 0.3, random_state= 5)

d_tree = DecisionTreeClassifier()

d_tree.fit(X_train, Y_train)
d_tree_results = d_tree.predict(X_test)
d_score = d_tree.score(X_test, Y_test) * 100

r_forest = RandomForestClassifier()

r_forest.fit(X_train, Y_train)
r_forest_results = r_forest.predict(X_test)
r_score = r_forest.score(X_test, Y_test) * 100
#Random Forest Score:  99.9602073897218
#Decision Tree Score:  99.916903666772

dt_matrix = confusion_matrix(Y_test, d_tree_results.round())
print("Confusion Matrix - Decision Tree")
print(dt_matrix)
plot_confusion_matrix(d_tree, X_test, Y_test)
#plt.show()


rf_matrix = confusion_matrix(Y_test, r_forest_results.round())
print("Confusion Matrix - Random Forest")
print(rf_matrix)
plot_confusion_matrix(r_forest, X_test, Y_test)
#plt.show()
