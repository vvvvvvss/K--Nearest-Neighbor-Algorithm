import numpy as np
from collections import Counter
def euclidean(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))
class KNN:
    def __init__(self,k=2):
        self .k=k

    def fit(self,x,y):
        self.x_train = x
        self.y_train = y

    def predict(self,x):
        predicted_labels = [self._predict(i) for i in x]
        return np.array(predicted_labels)

    def _predict(self,x):
        distances = [euclidean(x,x_train) for x_train in self.x_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearst_labels = [self.y_train[i] for i in k_indices]
        most_common =  Counter(k_nearst_labels).most_common()
        return most_common[0][0]
 from sklearn import datasets
iris = datasets.load_iris()
x = iris.data
y = iris.target 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=12)
knn = KNN(3)
knn.fit(x_train,y_train)
pred =  knn.predict(x_test)                   
accuracy= np.sum(pred==y_test)/len(y_test)
accuracy
