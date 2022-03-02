import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import seaborn as sn
from sklearn.metrics import confusion_matrix



df = pd.read_excel('emp_salary.xlsx')

inputs = df.drop(['salary'],axis=1)
target = df['salary']

le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

inputs['company_n'] =  le_company.fit_transform(inputs['company'])
inputs['job_n'] =  le_job.fit_transform(inputs['job'])
inputs['degree_n'] =  le_degree.fit_transform(inputs['degree'])

inputs.drop(['company','job','degree'],axis=1,inplace=True)


model = tree.DecisionTreeClassifier()

X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.1)

model.fit(X_train,y_train)

y_predict = model.predict(X_test)

print( model.score(X_test,y_test) )

cm = confusion_matrix(y_test,y_predict)
plt.figure(figsize= (7,5))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

plt.show()


