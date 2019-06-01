from __future__ import print_function
import numpy as np 
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split # for splitting data
from sklearn.metrics import accuracy_score # for evaluating result

np.random.seed(7)
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
print("labels:", np.unique(iris_y))

# split train and test 
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=130)
print('Train size:', X_train.shape[0], ', test size:', X_test.shape[0])
# danh gia trong so khac nhau cho 7 diem gan nhat.
# diem cang gan kiem thu phai duoc danh trong so cang cao
# model = neighbors.KNeighborsClassifier(n_neighbors=1, p=2, weight='distance')
model = neighbors.KNeighborsClassifier(n_neighbors=1, p=2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy of 1NN: %.2f %%" %(100 * accuracy_score(y_test, y_pred)))

def myweight(distances):
    sigma2 = .4 # we can change this number
    return np.exp(-distances**2/sigma2)

# model = neightbors.KNeighborsClassifier(n_neighbors=7, p=2, weights=myweight)