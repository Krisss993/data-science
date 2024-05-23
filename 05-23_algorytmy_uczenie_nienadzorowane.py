
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons
import plotly.express as px
import plotly.offline as pyo
import plotly.figure_factory as ff
import plotly.graph_objects as go


from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from mlxtend.plotting import plot_decision_regions

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn import datasets

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.datasets.fashion_mnist import load_data
from sklearn.cluster import KMeans


from numpy.linalg import norm

from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons

import random

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
from plotly.subplots import make_subplots


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor 

blobs_data = make_blobs(n_samples=1000, cluster_std=0.7, random_state=24, center_box=(-4.0, 4.0))[0]
blobs = pd.DataFrame(blobs_data, columns=['x1', 'x2'])
fig = px.scatter(blobs, 'x1', 'x2', width=950, height=500, title='blobs data', template='plotly_dark')
pyo.plot(fig)

circle_data = make_circles(n_samples=1000, factor=0.5, noise=0.05)[0]
circle = pd.DataFrame(circle_data, columns=['x1', 'x2'])
fig = px.scatter(circle, 'x1', 'x2', width=950, height=500, title='circle data', template='plotly_dark')
pyo.plot(fig)

moons_data = make_moons(n_samples=1000, noise=0.05)[0]
moons = pd.DataFrame(moons_data, columns=['x1', 'x2'])
fig = px.scatter(moons, 'x1', 'x2', width=950, height=500, title='moons data', template='plotly_dark')
pyo.plot(fig)

random_data = np.random.rand(1500, 2)
random = pd.DataFrame(random_data, columns=['x1', 'x2'])
fig = px.scatter(random, 'x1', 'x2', width=950, height=500, title='random data', template='plotly_dark')
pyo.plot(fig)


plt.scatter(blobs.iloc[:,0], blobs['x2'])
plt.show()
plt.scatter(circle.iloc[:,0], circle['x2'])
plt.show()
plt.scatter(moons.iloc[:,0], moons['x2'])
plt.show()
plt.scatter(random.iloc[:,0], random['x2'])
plt.show()






fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.01)

kmeans = KMeans(n_clusters=3)
kmeans.fit(blobs_data)
clusters = kmeans.predict(blobs_data)
blobs['cluster'] = clusters
trace1 = px.scatter(blobs, 'x1', 'x2', 'cluster', width=800, height=500)['data'][0]
fig.add_trace(trace1, row=1, col=1)

agglo = AgglomerativeClustering(n_clusters=3,metric='euclidean')
clusters = agglo.fit_predict(blobs_data)
blobs['cluster'] = clusters
trace2 = px.scatter(blobs, 'x1', 'x2', 'cluster', width=800, height=500)['data'][0]
fig.add_trace(trace2, row=1, col=2)

dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(blobs_data)
clusters = dbscan.labels_
blobs['cluster'] = clusters
trace3 = px.scatter(blobs, 'x1', 'x2', 'cluster', width=800, height=500)['data'][0]
fig.add_trace(trace3, row=1, col=3)

fig.update_layout(title='KMeans vs. Agglomerative Clustering vs. DBSCAN - blobs data', 
                  template='plotly_dark', coloraxis = {'colorscale':'viridis'})
pyo.plot(fig)






fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.01)

kmeans = KMeans(n_clusters=3)
#kmeans.fit_predict()
kmeans.fit(moons_data)
clusters = kmeans.predict(moons_data)
moons['cluster'] = clusters
trace1 = px.scatter(moons, 'x1', 'x2', 'cluster', width=800, height=500)['data'][0]
fig.add_trace(trace1, row=1, col=1)


agglo = AgglomerativeClustering(n_clusters=3, metric='euclidean')

clusters = agglo.fit_predict(moons_data)
moons['cluster'] = clusters
trace2 = px.scatter(moons, 'x1', 'x2', 'cluster', width=800, height=500)['data'][0]
fig.add_trace(trace2, row=1, col=2)



dbscan = DBSCAN(eps=0.1, min_samples=5)
clusters = dbscan.fit_predict(moons_data)
moons['cluster'] = clusters
trace3 = px.scatter(moons, 'x1', 'x2', 'cluster', width=800, height=500)['data'][0]
fig.add_trace(trace3, row=1, col=3)
fig.update_layout(title='KMeans vs. Agglomerative Clustering vs. DBSCAN - blobs data', 
                  template='plotly_dark', coloraxis = {'colorscale':'viridis'})
pyo.plot(fig)






fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.01)

kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(circle_data)
circle['cluster'] = clusters
trace1 = px.scatter(circle, 'x1', 'x2', 'cluster', width=800, height=500)['data'][0]
fig.add_trace(trace1, row=1, col=1)

agglo = AgglomerativeClustering(n_clusters=3,metric='euclidean')
clusters = agglo.fit_predict(circle_data)
circle['cluster'] = clusters
trace2 = px.scatter(circle, 'x1', 'x2', 'cluster', width=800, height=500)['data'][0]
fig.add_trace(trace2, row=1, col=2)

dbscan = DBSCAN(eps=0.2, min_samples=5)
clusters = dbscan.fit_predict(circle_data)
circle['cluster'] = clusters
trace3 = px.scatter(circle, 'x1', 'x2', 'cluster', width=800, height=500)['data'][0]
fig.add_trace(trace3, row=1, col=3)
fig.update_layout(title='KMeans vs. Agglomerative Clustering vs. DBSCAN - blobs data', 
                  template='plotly_dark', coloraxis = {'colorscale':'viridis'})
pyo.plot(fig)






fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.01)

kmeans = KMeans(n_clusters=4)
clusters = kmeans.fit_predict(random_data)
# transf = kmeans.fit_transform(random_data)
random['cluster'] = clusters
trace1 = px.scatter(random, 'x1', 'x2', 'cluster', width=800, height=500)['data'][0]
fig.add_trace(trace1, row=1, col=1)


agglo = AgglomerativeClustering(n_clusters=4, metric='euclidean')
clusters = agglo.fit_predict(random_data)
random['cluster'] = clusters
trace2 = px.scatter(random, 'x1', 'x2', 'cluster', width=800, height=500)['data'][0]
fig.add_trace(trace2, row=1, col=2)


dbscan = DBSCAN(eps=0.1, min_samples=5)
clusters = dbscan.fit_predict(random_data)
random['cluster'] = clusters
trace3 = px.scatter(random, 'x1', 'x2', 'cluster', width=800, height=500)['data'][0]
fig.add_trace(trace3, row=1, col=3)
fig.update_layout(title='KMeans vs. Agglomerative Clustering vs. DBSCAN - blobs data', 
                  template='plotly_dark', coloraxis = {'colorscale':'viridis'})
pyo.plot(fig)







# REDUKCJA WYMIAROWOSCI PCA



df_raw = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df = df_raw.copy()
df.head()

data = df.iloc[:,1:]
target = df.iloc[:,0]


X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)



pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_std)
# X_test_pca = pca.transform(X_test_std)
X_train_pca.shape



results = pd.DataFrame({'explained_variance_ratio':pca.explained_variance_ratio_})
results['cumulative'] = np.cumsum(results['explained_variance_ratio'])
results['component'] = results.index+1
results

fig = go.Figure(data=[go.Bar(x=results['component'], y=results['explained_variance_ratio'], name='explained_variance_ratio'),
                      go.Scatter(x=results['component'], y=results['cumulative'], name='cumulative')],
                layout=go.Layout(title='PCA - 3 components', width=950, template='plotly_dark'))
pyo.plot(fig)

X_train_pca_df = pd.DataFrame(data=np.c_[X_train_pca, y_train], columns=['pca1','pca2','pca3','class'])
X_train_pca_df['class'] = X_train_pca_df['class'].astype(str)
X_train_pca_df

fig = px.scatter_3d(data_frame=X_train_pca_df, x=X_train_pca_df.iloc[:,0], y=X_train_pca_df.iloc[:,1], z=X_train_pca_df.iloc[:,2], color='class', symbol='class')
pyo.plot(fig)


pca = PCA(n_components=2)
X_train_pca_2 = pca.fit_transform(X_train_std)


X_train_pca_df = pd.DataFrame(data=np.c_[X_train_pca_2, y_train], columns=['pca1','pca2','class'])
X_train_pca_df['class'] = X_train_pca_df['class'].astype(str)
X_train_pca_df





# METODA TSNE



tsne = TSNE(n_components=2, verbose=True)
X_train_tsne = tsne.fit_transform(X_train_std)


X_train_tsne_df = pd.DataFrame(data=np.c_[X_train_tsne, y_train], columns=['tsne_1', 'tsne_2', 'class'])
X_train_tsne_df['class'] = X_train_tsne_df['class'].astype(str)
X_train_tsne_df

fig = px.scatter(X_train_tsne_df, x='tsne_1', y='tsne_2', color='class', opacity=0.5, width=950, height=700,
           template='plotly_dark', title='TSNE - 2 components')
pyo.plot(fig)





fig = make_subplots(rows=1, cols=2, subplot_titles=['PCA', 't-SNE'], horizontal_spacing=0.03)

fig1 = px.scatter(X_train_pca_df, x='pca1', y='pca2', color='class', opacity=0.5)
fig2 = px.scatter(X_train_tsne_df, x='tsne_1', y='tsne_2', color='class', opacity=0.5)

for i in range(0, 2):
    fig.add_trace(fig1['data'][i], row=1, col=1)
    fig.add_trace(fig2['data'][i], row=1, col=2)
fig.update_layout(width=950, showlegend=False, template='plotly_dark')
pyo.plot(fig)















# REGULY ASOCJACYJNE ALGORYTM APRIORI





data = {'produkty': ['chleb jajka mleko', 'mleko ser', 'chleb masło ser', 'chleb jajka']}

transactions = pd.DataFrame(data=data, index=[1, 2, 3, 4])
transactions


# rozwinięcie kolumny do obiektu DataFrame
expand = transactions['produkty'].str.split(expand=True)
expand

product_list = []
for i in range(expand.shape[0]):
    for j in range(expand.shape[1]):
        if expand.iloc[i,j] not in product_list and expand.iloc[i,j]:
            product_list.append(expand.iloc[i,j])
product_list


transactions_encoded = np.zeros((len(expand), len(product_list)))
transactions_encoded

for idx, product in enumerate(product_list):
    for i in range(expand.shape[0]):
        for j in range(expand.shape[1]):
            if expand.iloc[i,j] == product:
                transactions_encoded[i,idx] = 1
transactions_encoded
product_list
expand

transactions_encoded_df = pd.DataFrame(transactions_encoded, columns=product_list)
transactions_encoded_df = transactions_encoded_df.astype('int8')

# % WYSTEPOWANIA PRODUKTOW W OGOLE TRANSAKCJI
supports = apriori(transactions_encoded_df, min_support=0.0000001, use_colnames=True)
supports

supports = apriori(transactions_encoded_df, min_support=0.3, use_colnames=True)
supports

rules = association_rules(supports, metric='confidence', min_threshold=0.65)
rules = rules.iloc[:, [0, 1, 4, 5, 6]]
rules






# DETEKCJA ANOMALII - ISOLATION FOREST


data = pd.read_csv('https://storage.googleapis.com/esmartdata-courses-files/ml-course/factory.csv')
data.head()


data.describe()

fig = px.scatter(data, x='item_length', y='item_width', width=950, template='plotly_dark', title='Isolation Forest')
pyo.plot(fig)

# contamination in [0, 0.05]
outlier = IsolationForest(n_estimators=100, contamination=0.05)
outlier.fit(data)



y_pred = outlier.predict(data)
y_pred[:30]


data['outlier_flag'] = y_pred
fig = px.scatter(data, x='item_length', y='item_width', color='outlier_flag', width=950, template='plotly_dark',
           color_continuous_midpoint=-1, title='Isolation Forest')
pyo.plot(fig)



# DETEKCJA ANOMALII - LOCAL OUTLIER FACTOR

data = make_blobs(n_samples=300, cluster_std=2.0, random_state=10)[0]
data[:5]


tmp = pd.DataFrame(data=data, columns=['x1', 'x2'])
fig = px.scatter(tmp, x='x1', y='x2', width=950, title='Local Outlier Factor', template='plotly_dark')
pyo.plot(fig)



lof = LocalOutlierFactor(n_neighbors=20)
y_pred = lof.fit_predict(data)
y_pred[:10]



all_data = np.c_[data, y_pred]
all_data[:5]
     


tmp['y_pred'] = y_pred
fig = px.scatter(tmp, x='x1', y='x2', color='y_pred', width=950, 
           title='Local Outlier Factor', template='plotly_dark')
pyo.plot(fig)



plt.figure(figsize=(12, 7))
plt.scatter(all_data[:, 0], all_data[:, 1], c=all_data[:, 2], cmap='tab10', label='data')
plt.title('Local Outlier Factor')
plt.legend()
plt.show()


LOF_scores = lof.negative_outlier_factor_
radius = (LOF_scores.max() - LOF_scores) / (LOF_scores.max() - LOF_scores.min())
radius[:5]


plt.figure(figsize=(12, 7))
plt.scatter(all_data[:, 0], all_data[:, 1], label='data', cmap='tab10')
plt.scatter(all_data[:, 0], all_data[:, 1], s=2000 * radius, edgecolors='r', facecolors='none', label='outlier scores')
plt.title('Local Outlier Factor')
legend = plt.legend()
legend.legendHandles[1]._sizes = [40]
plt.show()

plt.figure(figsize=(12, 7))
plt.scatter(all_data[:, 0], all_data[:, 1], c=all_data[:, 2], cmap='tab10', label='data')
plt.scatter(all_data[:, 0], all_data[:, 1], s=2000 * radius, edgecolors='r', facecolors='none', label='outlier scores')
plt.title('Local Outlier Factor')
legend = plt.legend()
legend.legendHandles[1]._sizes = [40]
plt.show()











