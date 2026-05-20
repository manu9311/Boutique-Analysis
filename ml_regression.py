import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

df = pd.read_csv("Purchases - Sheet1.csv")
df_encoded = pd.get_dummies(df, columns= ['Category']) # all columns plus category columns separated for suits, sarees etc.
print(df.shape)
print(df_encoded.shape)

# y = what i want to predict
# X = everything used to get Y

y = df_encoded['Selling Price']

category_cols = []
for i in df_encoded.columns:
 if 'Category' in i :
  category_cols.append(i)

X = df_encoded[['Cost Price', 'Quantity']+ category_cols]
print(X.shape)
print(y.shape)

X_train , X_test ,  y_train , y_test = train_test_split(X,y , random_state=42, test_size=0.2)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
print(y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(mae)
print(r2)