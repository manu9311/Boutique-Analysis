import pandas as pd
import numpy as np
from datetime import datetime

df = pd.read_csv('/Users/manurana/Documents/boutique_analysis/Purchases - Sheet1.csv')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

df['Month'] = df['Date'].dt.to_period('M')
monthly = df.groupby('Month')['Selling Price'].sum().reset_index()
monthly.columns = ['Month', 'Revenue']
monthly['Month_Num'] = range(len(monthly))

from numpy.polynomial.polynomial import polyfit

x = monthly['Month_Num'].values
y = monthly['Revenue'].values
coefs = polyfit(x, y, 1)

next_3 = []
last_month_num = monthly['Month_Num'].max()
for i in range(1, 4):
    predicted = coefs[0] + coefs[1] * (last_month_num + i)
    next_3.append(predicted)

category_revenue = df.groupby('Category')['Selling Price'].sum()
total_revenue = category_revenue.sum()
category_pct = (category_revenue / total_revenue * 100).round(1)

top_categories = category_pct.sort_values(ascending=False).head(5)

df['MonthName'] = df['Date'].dt.strftime('%B')
seasonality = df.groupby('MonthName')['Selling Price'].sum().sort_values(ascending=False)

avg_margin = ((df['Selling Price'] - df['Cost Price']) / df['Selling Price']).mean()
avg_predicted = np.mean(next_3)
recommended_stock_budget = avg_predicted * (1 - avg_margin)

print("=" * 60)
print("PROBLEM 3: STOCK PLANNING REPORT")
print("=" * 60)

print("\n📅 REVENUE BY MONTH (actual):")
print(monthly[['Month', 'Revenue']].to_string(index=False))

print("\n📈 NEXT 3 MONTHS FORECAST:")
months_ahead = ['Month +1', 'Month +2', 'Month +3']
for label, val in zip(months_ahead, next_3):
    print(f"  {label}:  ₹{val:,.0f}")

print(f"\n📊 WHERE YOUR REVENUE ACTUALLY COMES FROM:")
for cat, pct in top_categories.items():
    print(f"  {cat:<20} {pct}% of total revenue")

print(f"\n🗓️  BEST AND WORST MONTHS (historical):")
for month, rev in seasonality.items():
    bar = '█' * int(rev / 5000)
    print(f"  {month:<12} ₹{rev:>8,.0f}  {bar}")

print(f"\n💰 STOCK BUDGET RECOMMENDATION:")
print(f"  Average predicted monthly revenue: ₹{avg_predicted:,.0f}")
print(f"  Your average margin:               {avg_margin*100:.1f}%")
print(f"  Recommended stock budget/month:    ₹{recommended_stock_budget:,.0f}")

print(f"\n🎯 WHERE TO PUT THAT BUDGET (based on actual sales):")
for cat, pct in top_categories.items():
    alloc = recommended_stock_budget * (pct / 100)
    print(f"  {cat:<20} ₹{alloc:,.0f}  ({pct}%)")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
best_month = seasonality.index[0]
worst_month = seasonality.index[-1]
print(f"  Best month historically:   {best_month} — stock up before this")
print(f"  Worst month historically:  {worst_month} — reduce orders, run discounts")
print(f"  #1 category to always stock: {top_categories.index[0]} ({top_categories.iloc[0]}% of revenue)")