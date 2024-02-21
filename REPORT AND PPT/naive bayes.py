[11:24 pm, 05/12/2023] Yadunandan: import pandas as pd
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
path="C:\\Users\\dell8\\OneDrive\\Documents\\onedrive\\Desktop\\project\\data\\train_dataset.csv"
data = pd.read_csv(path)
print(data.info())
print(data)
inputs = data.drop(['class'],axis=1)
outputs = data.drop(['fLength','fWidth','fSize','fConc','fConc1','fAsym','fM3Long','fM3Trans','fAlpha','fDist'],axis=1)
print("input")
print(inputs)
print("output")
print(outputs)
x_train,x_test,y_train,y_test=train_test_split(inputs,outputs,test_size=0.2)
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
model=GaussianNB()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Predicted values:")
print(y_pred)
print(y_test)

new=np.array([[66.8366,8.663,5.333,0.252,0.825,-1.8999,-5.325,-9.652,75.256,250.366]])
new=sc.transform(new)
result=model.predict(new)
print(result)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
[11:24 pm, 05/12/2023] Yadunandan: import pandas as pd
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
path="C:\\Users\\dell8\\OneDrive\\Documents\\onedrive\\Desktop\\project\\data\\train_dataset.csv"
data = pd.read_csv(path)
print(data.info())
print(data)
inputs = data.drop(['class'],axis=1)
outputs = data.drop(['fLength','fWidth','fSize','fConc','fConc1','fAsym','fM3Long','fM3Trans','fAlpha','fDist'],axis=1)
print("input")
print(inputs)
print("output")
print(outputs)
x_train,x_test,y_train,y_test=train_test_split(inputs,outputs,test_size=0.2)
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
model=tree.DecisionTreeClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Predicted values:")
print(y_pred)
print(y_test)

new=np.array([[66.8366,8.663,5.333,0.252,0.825,-1.8999,-5.325,-9.652,75.256,250.366]])
new=sc.transform(new)
result=model.predict(new)
print(result)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy