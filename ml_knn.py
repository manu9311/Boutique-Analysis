# KNN
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


df = pd.read_csv('Purchases - Sheet1.csv')
product_df = df.groupby('Category').agg(
    total_units=('Quantity', 'sum'),
    total_revenue=('Selling Price', 'sum'),
    total_orders=('Selling Price', 'count')
).reset_index()

print("Product summary:")
print(product_df)

def label_velocity(units):
    if units >= 50:
        return 'Fast'
    elif units >= 10:
        return 'Medium'
    else:
        return 'Slow'

product_df['Velocity'] = product_df['total_units'].apply(label_velocity)

print("\nCategories with velocity labels:")
print(product_df[['Category', 'total_units', 'Velocity']])

X = product_df[['total_units', 'total_revenue', 'total_orders']]
y = product_df['Velocity']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
print("Actual labels: ", list(y_test))
print("Predicted values: "), list(y_pred)
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))