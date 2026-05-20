import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Purchases - Sheet1.csv')

customer_df = df.groupby('Names').agg(
    total_revenue = ('Selling Price', 'sum'),
    total_orders = ('Selling Price', 'count'),
    avg_order_value = ('Selling Price', 'mean')
).reset_index()
#in agg -^ : its not ['column'].operation(), instead:-
# it is : ('column', 'operation')
print(customer_df.head())
features = customer_df[['total_revenue', 'total_orders', 'avg_order_value']]
scaler = StandardScaler()
#Scaling brings all three columns to the same range so KMeans treats them equally when measuring distance between customers.
scaled_features = scaler.fit_transform(features)
print(scaled_features)

Kmeans = KMeans(n_clusters= 3, random_state= 42, n_init = 10)
Kmeans.fit(scaled_features)
customer_df['Cluster'] = Kmeans.labels_
print(customer_df['Cluster'].value_counts())
print(customer_df.groupby('Cluster')[['total_revenue', 'total_orders', 'avg_order_value']].mean().round(0))
cluster_names = {0: 'Regular', 1: 'Occasional', 2: 'VIP'}
customer_df['Segment'] = customer_df['Cluster'].map(cluster_names)
print(customer_df[['Names', 'total_revenue', 'Segment']].sort_values('total_revenue', ascending=False).head(10))

import matplotlib.pyplot as plt

colors = ['blue', 'green', 'red']
labels = ['Regular', 'Occasional', 'VIP']

for i in range(3):
    cluster_data = customer_df[customer_df['Cluster'] == i]
    plt.scatter(
        cluster_data['total_orders'],
        cluster_data['total_revenue'],
        c=colors[i],
        label=labels[i],
        s=100
    )

plt.xlabel('Total Orders')
plt.ylabel('Total Revenue (₹)')
plt.title('Customer Segmentation - K-Means Clustering')
plt.legend()
plt.tight_layout()
plt.savefig('chart_kmeans_clusters.png', dpi=150)
plt.close()
print("Chart saved!")