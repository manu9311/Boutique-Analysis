
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("Purchases - Sheet1.csv")
customer_df = df.groupby('Names').agg(
    total_revenue = ('Selling Price', 'sum'),
    total_orders = ('Selling Price','count'), #could've used names, count or quantity, count also - basically any column that alr exists
    avg_order_value = ('Selling Price', 'mean')
).reset_index()
print("Customer Summary :- ")
print(customer_df.head(15))
print(f"\nTotal customers: {len(customer_df)}")


features = customer_df[['total_revenue', 'total_orders', 'avg_order_value']]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

print("Scaling done. Shape:", scaled_features.shape)


kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(scaled_features)

customer_df['Cluster'] = kmeans.labels_

print("\nCustomers with cluster labels:")
print(customer_df.head(10))

print("\nHow many customers in each cluster:")
print(customer_df['Cluster'].value_counts())
print("\nAverage stats per cluster:")
print(customer_df.groupby('Cluster')[['total_revenue', 'total_orders', 'avg_order_value']].mean().round(0))
print("\nCustomers in Cluster 0")
print(customer_df[customer_df['Cluster'] == 0][['Names', 'total_revenue', 'total_orders']])
print("\nCustomers in Cluster 1")
print(customer_df[customer_df['Cluster'] == 1][['Names', 'total_revenue', 'total_orders']])
print("\nCustomers in Cluster 2")
print(customer_df[customer_df['Cluster'] == 2][['Names', 'total_revenue', 'total_orders']])

plt.figure(figsize=(10, 6))

colors = ['blue', 'green', 'red']
labels = ['Regular (Cluster 0)', 'Occasional (Cluster 1)', 'VIP (Cluster 2)']

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
print("\nChart saved!")