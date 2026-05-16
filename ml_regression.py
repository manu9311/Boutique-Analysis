# ── IMPORTS ──────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# ── LOAD DATA ────────────────────────────────────────────
df = pd.read_csv('Purchases - Sheet1.csv')

# ── STEP 1: PREPARE FEATURES ─────────────────────────────
# We want to predict Selling Price using:
# - Cost Price (how much it cost your mom)
# - Quantity (how many units)
# - Category (what type of product)

# Category is text — we need to convert it to numbers
# pd.get_dummies converts each category into its own column of 0s and 1s
df_encoded = pd.get_dummies(df, columns=['Category'])

print("Original shape:", df.shape)
print("After encoding:", df_encoded.shape)
print("\nNew columns added:")
print([col for col in df_encoded.columns if 'Category' in col])
# ── STEP 2: DEFINE X AND y ────────────────────────────────
# y = what we want to predict
y = df_encoded['Selling Price']

# X = everything we use to predict it
# We use Cost Price, Quantity, and all the Category columns
category_cols = [col for col in df_encoded.columns if 'Category' in col]
X = df_encoded[['Cost Price', 'Quantity'] + category_cols]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Features used: {X.shape[1]}")
# ── STEP 3: TRAIN AND TEST REGRESSION ────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Test the model
y_pred = model.predict(X_test_scaled)

# ── STEP 4: CHECK HOW GOOD IT IS ─────────────────────────
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: ₹{mae:.0f}")
print(f"R2 Score: {r2:.2f}")
print("\nSample predictions vs actual:")
for i in range(5):
    print(f"Actual: ₹{y_test.iloc[i]:,.0f}  →  Predicted: ₹{y_pred[i]:,.0f}")