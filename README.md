# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Detect File Encoding: Use chardet to determine the dataset's encoding.
2. Load Data: Read the dataset with pandas.read_csv using the detected encoding.
3. Inspect Data: Check dataset structure with .info() and missing values with .isnull().sum().
4. Split Data: Extract text (x) and labels (y) and split into training and test sets using train_test_split.
5. Convert Text to Numerical Data: Use CountVectorizer to transform text into a sparse matrix.
6. Train SVM Model: Fit an SVC model on the training data.
7. Predict Labels: Predict test labels using the trained SVM model.
8. Evaluate Model: Calculate and display accuracy with metrics.accuracy_score. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: VISHAGAN R S
RegisterNumber:  25002817
*/
import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect (rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv('spam.csv', encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train, y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:

<img width="1428" height="40" alt="image" src="https://github.com/user-attachments/assets/b7c0c4eb-58dd-4240-aba7-dc21a47a8b82" />


<img width="680" height="211" alt="image" src="https://github.com/user-attachments/assets/205f4208-04fb-4310-aaa5-c266b9053911" />


<img width="1398" height="222" alt="image" src="https://github.com/user-attachments/assets/28726d9a-a8de-464c-90a1-8d060b3db0de" />


<img width="1417" height="242" alt="image" src="https://github.com/user-attachments/assets/cc6ea51f-ac15-4cf7-a590-b4bc10bdc4d5" />


<img width="1382" height="67" alt="image" src="https://github.com/user-attachments/assets/19dd7e56-b4f2-4463-b8ac-c19064fa5a4e" />


<img width="1442" height="37" alt="image" src="https://github.com/user-attachments/assets/62d06965-8696-404d-92a0-e921ba142275" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
