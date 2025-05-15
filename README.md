# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: RAMYA S
RegisterNumber: 212224040268 
*/
```
```
import pandas as pd
df=pd.read_csv("/content/Employee.csv")
print("data.head():")
df.head()

print("data.info()")
df.info()

print("data.isnull().sum()")
df.isnull().sum()

print("data value counts")
df["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
print("data.head() for Salary:")
df["salary"]=le.fit_transform(df["salary"])
df.head()

print("x.head():")
x=df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=df["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
print("Data prediction")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plot_tree(dt,filled=True,feature_names=x.columns,class_names=['salary' , 'left'])
plt.show()

```

## Output:
![decision tree classifier model](sam.png)
![image](https://github.com/user-attachments/assets/281be63f-7241-4ebf-88e1-c7a619709607)
![image](https://github.com/user-attachments/assets/f28a503a-ac04-470d-be95-978924f9e435)
![image](https://github.com/user-attachments/assets/73917578-2ce6-4614-8b03-243c6f11d16a)
![image](https://github.com/user-attachments/assets/7390eb16-1d9c-4902-bf77-be3f9a9a8f7c)
![image](https://github.com/user-attachments/assets/18a8e991-23b4-4746-a5c1-1576c0a68336)
![image](https://github.com/user-attachments/assets/dffb5043-40c3-455f-98c4-ca5dc6754831)
![image](https://github.com/user-attachments/assets/e8d10dc2-d036-46c8-9b7b-efb106b2008b)
![image](https://github.com/user-attachments/assets/48f4994a-5b3d-44f3-a813-57c471a183ab)
![image](https://github.com/user-attachments/assets/bd20effe-85cf-49e1-8bb1-23427f6745a7)
![image](https://github.com/user-attachments/assets/1e2dd20a-a0a2-4a44-9be7-041822ff63e3)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
