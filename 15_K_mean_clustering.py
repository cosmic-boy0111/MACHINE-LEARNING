import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


df = pd.read_excel('./income.xlsx')

# plt.scatter(df.Age,df.Income)
# plt.xlabel('Age')
# plt.ylabel(('Income'))
# plt.show()


km = KMeans(n_clusters=3)
y_predict =  km.fit_predict(df[['Age','Income']])
print(y_predict)
df['cluster'] = y_predict

print(df)

print(km.cluster_centers_)

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]

plt.scatter(df1.Age,df1.Income,color='green')
plt.scatter(df2.Age,df2.Income,color='red')
plt.scatter(df3.Age,df3.Income,color='cyan')
plt.xlabel('Age')
plt.ylabel(('Income'))
plt.legend()
plt.show()

scaler = MinMaxScaler()
scaler.fit(df[['Income']])
df.Income = scaler.transform(df[['Income']])
scaler.fit(df[['Age']])
df.Age = scaler.transform(df[['Age']])
print(df.head())

km = KMeans(n_clusters=3)
y_predict =  km.fit_predict(df[['Age','Income']])
print(y_predict)
df['cluster'] = y_predict

print(df)

print(km.cluster_centers_)

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]

plt.scatter(df1.Age,df1.Income,color='green')
plt.scatter(df2.Age,df2.Income,color='red')
plt.scatter(df3.Age,df3.Income,color='cyan')
plt.xlabel('Age')
plt.ylabel(('Income'))
plt.legend()
plt.show()


sse = []

k_rng = range(1,10)

for k in k_rng: 
    km = KMeans(n_clusters=k)
    km.fit(df[['Age']],df.Income)
    sse.append(km.inertia_)

print(sse)

plt.plot(k_rng,sse)
plt.xlabel('K')
plt.ylabel('sse')

plt.show()







