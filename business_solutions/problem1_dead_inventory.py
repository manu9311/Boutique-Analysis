import pandas as pd
from datetime import datetime

df = pd.read_csv('/Users/manurana/Documents/boutique_analysis/Purchases - Sheet1.csv')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

#last sale date for each category
last_sold = df.groupby('Category')['Date'].max().reset_index()
last_sold.columns = ['Category', 'Last Sale Date']

today = pd.Timestamp.today()
last_sold['Days Since Last Sale'] = (today - last_sold['Last Sale Date']).dt.days

#Total cost price invested
cost_invested = df.groupby('Category')['Cost Price'].sum().reset_index()
cost_invested.columns = ['Category', 'Total Cost Invested']

# Total revenue
revenue = df.groupby('Category')['Selling Price'].sum().reset_index()
revenue.columns = ['Category', 'Total Revenue']


summary = last_sold.merge(cost_invested, on='Category').merge(revenue, on='Category')

summary['Status'] = summary['Days Since Last Sale'].apply(
    lambda x: '🔴 DEAD' if x > 60 else ('🟡 SLOW' if x > 30 else '🟢 ACTIVE')
)


dead = summary[summary['Status'] == '🔴 DEAD']
slow = summary[summary['Status'] == '🟡 SLOW']

dead_cash = dead['Total Cost Invested'].sum()
slow_cash = slow['Total Cost Invested'].sum()

print("=" * 60)
print("PROBLEM 1: DEAD INVENTORY REPORT")
print("=" * 60)
print(summary[['Category', 'Days Since Last Sale', 'Total Cost Invested', 'Status']].sort_values('Days Since Last Sale', ascending=False).to_string(index=False))

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"🔴 Dead categories (60+ days):   ₹{dead_cash:,.0f} locked in stock")
print(f"🟡 Slow categories (30-60 days): ₹{slow_cash:,.0f} at risk")
print(f"\nIf you run a 15% discount sale on dead stock:")
print(f"  You recover:    ₹{dead_cash * 0.85:,.0f}")
print(f"  You lose:       ₹{dead_cash * 0.15:,.0f} (discount cost)")
print(f"  vs doing nothing: ₹0 recovered, cash stays locked")