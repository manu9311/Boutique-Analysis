# ── IMPORTS ──────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("Purchases - Sheet1.csv")
# ── STEP 1: BUILD CUSTOMER SUMMARY ───────────────────────
# For each customer, calculate 3 things:
# - total money spent
# - how many orders they placed
# - their average spend per order
customer_df = df.groupby('Names').agg(
    total_revenue = ('Selling Price', 'sum'),
    total_orders = ('Selling Price','count'), #could've used names, count or quantity, count also - basically any column that alr exists
    avg_order_value = ('Selling Price', 'mean')
).reset_index()
print("Customer Summary :- ")
print(customer_df.head(15))
print(f"\nTotal customers: {len(customer_df)}")
# ── STEP 2: SCALE THE DATA ───────────────────────────────
# K-Means uses distance to group customers.
# If total_revenue is in thousands and total_orders is 1-6,
# revenue will dominate just because its numbers are bigger.
# Scaling brings all 3 columns to the same range so they're equally important.

features = customer_df[['total_revenue', 'total_orders', 'avg_order_value']]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

print("Scaling done. Shape:", scaled_features.shape)
# ── STEP 3: RUN K-MEANS ──────────────────────────────────
# We're asking K-Means to find 3 groups:
# n_clusters=3 → find 3 groups
# random_state=42 → fixes the randomness so you get same result every time you run
# n_init=10 → tries 10 different starting points, picks the best one

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(scaled_features)

# Add the cluster label back to our customer table
# Each customer gets a number: 0, 1, or 2 (their group)
customer_df['Cluster'] = kmeans.labels_

print("\nCustomers with cluster labels:")
print(customer_df.head(10))

print("\nHow many customers in each cluster:")
print(customer_df['Cluster'].value_counts())
# ── STEP 4: NAME THE CLUSTERS ────────────────────────────
print("\nAverage stats per cluster:")
print(customer_df.groupby('Cluster')[['total_revenue', 'total_orders', 'avg_order_value']].mean().round(0))

print("\nCustomers in Cluster 0")
print(customer_df[customer_df['Cluster'] == 0][['Names', 'total_revenue', 'total_orders']])

print("\nCustomers in Cluster 1")
print(customer_df[customer_df['Cluster'] == 1][['Names', 'total_revenue', 'total_orders']])

print("\nCustomers in Cluster 2")
print(customer_df[customer_df['Cluster'] == 2][['Names', 'total_revenue', 'total_orders']])

# ── STEP 5: VISUALIZE THE CLUSTERS ───────────────────────
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