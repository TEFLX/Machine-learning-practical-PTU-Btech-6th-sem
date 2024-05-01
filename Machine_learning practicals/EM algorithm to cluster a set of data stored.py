from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
print("________Ritik kashyap _________")

data = pd.read_csv("sample_dataset.csv")
df1 = pd.DataFrame(data)
print(df1)

f1 = df1['Distance_Feature'].values
f2 = df1['Speeding_Feature'].values

X = np.matrix(list(zip(f1, f2)))

plt.scatter(f1, f2)
plt.xlim([0, 100])
plt.ylim([0, 50])
plt.title('Dataset')
plt.ylabel('speeding_feature')
plt.xlabel('Distance_Feature')
plt.show()

colors = ['b', 'g', 'r']
markers = ['o', 'v', 's']

kmeans_model = KMeans(n_clusters=3).fit(X)

plt.scatter(f1, f2, c=kmeans_model.labels_, cmap='viridis', s=50)
for i, l in enumerate(kmeans_model.labels_):
    plt.plot(f1[i], f2[i], color=colors[l], marker=markers[l], ls='None')
plt.xlim([0, 100])
plt.ylim([0, 50])
plt.show()
