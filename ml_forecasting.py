import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("Purchases - Sheet1.csv")
df['Date'] = pd.to_datetime(df['Date'], dayfirst = True)
df['YearMonth'] = df['Date'].dt.to_period('M')

monthly_df = df.groupby('YearMonth')['Selling Price'].sum().reset_index()
print(monthly_df)

monthly_df['MonthNumber'] = range(1, len(monthly_df) + 1)
X = monthly_df[['MonthNumber']]
y = monthly_df['Selling Price']
print(X)
print(y)

model = LinearRegression()
model.fit(X, y)

future_pred = pd.DataFrame({'MonthNumber': [16,17,18]})
predictions = model.predict(future_pred)
print(f"Jun 2026: ₹{predictions[0]:,.0f}")
print(f"Jul 2026: ₹{predictions[1]:,.0f}")
print(f"Aug 2026: ₹{predictions[2]:,.0f}")

# For own understanding:-
# n KNN and Regression we split because we wanted to test accuracy — hide some data, see if the model predicts it correctly.
# For forecasting, we're not testing accuracy on past data. We're predicting future months that don't exist yet in our data — June, July, August 2026. There's no "correct answer" to compare against.
# So there's nothing to hide. We give the model all 15 months so it learns the best possible trend, then ask it to predict beyond month 15.
# More data it learns from → better the forecast.

plt.figure(figsize=(12, 6))

plt.plot(monthly_df['MonthNumber'], monthly_df['Selling Price'], 
         marker='o', color='blue', label='Actual Revenue', linewidth=2)

plt.plot(monthly_df['MonthNumber'], model.predict(X),
         color='green', linestyle='--', label='Trend Line', linewidth=2)

plt.plot([15, 16, 17, 18], 
         [monthly_df['Selling Price'].iloc[-1]] + list(predictions),
         marker='o', color='red', linestyle='--', 
         label='Forecast', linewidth=2)

all_months = list(monthly_df['YearMonth'].astype(str)) + ['Jun 2026', 'Jul 2026', 'Aug 2026']
plt.xticks(range(1, 19), all_months, rotation=45, ha='right')

plt.title('Monthly Revenue — Actual vs Forecast')
plt.xlabel('Month')
plt.ylabel('Revenue (₹)')
plt.legend()
plt.tight_layout()
plt.savefig('chart_forecast.png', dpi=150)
plt.close()
print("Chart saved!")