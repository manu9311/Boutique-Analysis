import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv('Purchases - Sheet1.csv')

df_encoded = pd.get_dummies(df, columns=['Category'])

print("Original shape:", df.shape)
print("After encoding:", df_encoded.shape)
print("\nNew columns added:")
print([col for col in df_encoded.columns if 'Category' in col])
y = df_encoded['Selling Price']

category_cols = [col for col in df_encoded.columns if 'Category' in col]
X = df_encoded[['Cost Price', 'Quantity'] + category_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Features used: {X.shape[1]}")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: ₹{mae:.0f}")
print(f"R2 Score: {r2:.2f}")
print("\nSample predictions vs actual:")
for i in range(5):
    print(f"Actual: ₹{y_test.iloc[i]:,.0f}  →  Predicted: ₹{y_pred[i]:,.0f}")