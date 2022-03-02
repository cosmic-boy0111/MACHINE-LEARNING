import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt 

df = pd.read_csv('./spam.csv')

check = LabelEncoder()
df['spam'] = check.fit_transform(df.Category)

print(df.head())
# print(df.groupby('Category').describe())

X_train, X_test, y_train, y_test = train_test_split(df.Message, df.spam, test_size=0.25)


# v = CountVectorizer()
# X_train_count = v.fit_transform(X_train.values)

# print(X_train_count.toarray()[:3])

# model = MultinomialNB()
# model.fit(X_train_count,y_train)

# X_test_count = v.transform(X_test.values)
# print(X_test_count.toarray()[:3])

# print(model.predict(X_test_count))

# print(model.score(X_test_count,y_test))


clf = Pipeline([
    ('vectorizer',CountVectorizer()),
    ('nb',MultinomialNB())
])

clf.fit(X_train,y_train)


print(clf.score(X_test,y_test))

y_predict =  clf.predict(X_test)


cm = confusion_matrix(y_test,y_predict)
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('Predict')
plt.ylabel('Truth')
plt.show()