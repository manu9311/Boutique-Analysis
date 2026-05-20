import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

df = pd.read_csv('Purchases - Sheet1.csv')

product_df = df.groupby('Category').agg(
    total_units = ('Quantity', "sum"),
    total_revenue = ('Selling Price', 'sum'),
    total_orders = ('Selling Price', 'count')
).reset_index()
print(product_df)

def label_velocity(units):
    if units > 50:
        return "Fast"
    elif units >=10:
        return "Medium"
    else: 
        return "Slow"

product_df['Velocity'] = product_df['total_units'].apply(label_velocity)
print(product_df)

X = product_df[['total_units', 'total_revenue', 'total_orders']]
y = product_df['Velocity']

X_train, X_test , y_train , y_test = train_test_split(X,y, random_state=42, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test_scaled)
print(y_pred)
print(classification_report(y_test, y_pred))