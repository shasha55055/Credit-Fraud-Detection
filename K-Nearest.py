


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.cluster as cluster 
from sklearn.model_selection import train_test_split



data = pd.read_csv('card_data.csv')



#Calculate k using elbow method

intertia_list = []
Clusters = [1,2,3,4,5,6,7,8,9,10]
for i in range(1,11):
    K = cluster.KMeans(n_clusters = i, init = "k-means++" )
    K.fit(data[['Amount', "V1", "V2", "V3", "V4", "V5","V6", "V7","V8", "V9"]])
    entry = (K.inertia_)
    intertia_list.append(entry)
plt.plot(Clusters, intertia_list)
plt.show()

K = 6

K = cluster.KMeans(n_clusters = 6, init = "k-means++" )

kmeans = K.fit(data[['Amount', "V1", "V2", "V3", "V4", "V5","V6", "V7","V8", "V9"]])
data["Clusters"] = kmeans.labels_

Y = data["Clusters"]
X = data['Amount', "V1", "V2", "V3", "V4", "V5","V6", "V7","V8", "V9"]
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size= 0.3, random_state= 5)



(kmeans.cluster_centers_)





print(X.head(5))
print("DONE   DONE   DONE")

