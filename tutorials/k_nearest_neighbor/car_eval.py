import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd 
import numpy as np 
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
print(data.head())

# transform all string data into numerals
le = preprocessing.LabelEncoder()
buying  = le.fit_transform(list(data["buying"]))
maint  = le.fit_transform(list(data["maint"]))
doors  = le.fit_transform(list(data["doors"]))
persons  = le.fit_transform(list(data["persons"]))
lug_boot  = le.fit_transform(list(data["lug_boot"]))
safety  = le.fit_transform(list(data["safety"]))
clazz  = le.fit_transform(list(data["class"]))
print(buying)

# feature to predict
predict = "class"

X = list(zip(buying,maint,doors,persons,lug_boot,safety))
y = list(clazz)

# split data into training set and testing set
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

model = KNeighborsClassifier(n_neighbors=7)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 9, True)
    print("N: ", n)