# ── IMPORTS ──────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# ── LOAD DATA ────────────────────────────────────────────
df = pd.read_csv('Purchases - Sheet1.csv')

# ── STEP 1: BUILD PRODUCT SUMMARY ────────────────────────
# For each category calculate total units sold, revenue, orders
product_df = df.groupby('Category').agg(
    total_units=('Quantity', 'sum'),
    total_revenue=('Selling Price', 'sum'),
    total_orders=('Selling Price', 'count')
).reset_index()

print("Product summary:")
print(product_df)
# ── STEP 2: LABEL THE CATEGORIES ─────────────────────────
# KNN is supervised — it needs labels to learn from
# We define the labels manually based on total units sold

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
# ── STEP 3: PREPARE DATA FOR KNN ─────────────────────────
# X = the input features KNN will learn from
# y = the correct answers (labels) KNN will learn to predict

X = product_df[['total_units', 'total_revenue', 'total_orders']]
y = product_df['Velocity']

# Split into training and testing data
# 80% of data → KNN learns from this
# 20% of data → we test if KNN learned correctly
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ── STEP 4: TRAIN AND TEST KNN ───────────────────────────
# Scale the data first (same reason as K-Means — fair distances)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
# n_neighbors=3 → look at 3 nearest neighbors to make a decision
# example: x_train = features/questions used for teaching; y_train = their correct answers;
# and x_test = features/questions in exam ; y_test = their answers in exam
# hence, x = questions , y = answers

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
print("Actual labels: ", list(y_test))
print("Predicted values: "), list(y_pred)
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))