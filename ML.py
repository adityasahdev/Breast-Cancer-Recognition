import os 
import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder , StandardScaler 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score , classification_report , confusion_matrix

dataset = pd.read_csv('/Users/adityaahdev/Desktop/Dataset Projects/Breast Cancer/Breast_cancer_dataset.csv')

#Diagnosis = M (Malignant): Indicates a cancerous tumor , B (Benign): Indicates a non-cancerous tumor.

df = dataset.copy()

df_input_label = df.iloc[:,2:32]   #choose coloumns , [rows,coloumns]

df_output_label = df.iloc[:,1]


label_en = LabelEncoder() 

df_output = label_en.fit_transform(df_output_label)   #labels categorical data

print(label_en.classes_) #Gets the encoded classes

ss = StandardScaler()

df_input = ss.fit_transform(df_input_label)

x_train, x_test , y_train , y_test = train_test_split(df_input,df_output,
                                         test_size = 0.2, random_state = 42)


model = LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

print('Accuracy Score:',accuracy_score(y_pred , y_test))


print('Classification Report:', classification_report(y_pred , y_test ,
                                                       target_names= label_en.classes_))

print('Confusion Matrix:',confusion_matrix(y_pred , y_test))


