import pandas as pd
import numpy as np
from datetime import datetime

df = pd.read_csv('/Users/manurana/Documents/boutique_analysis/Purchases - Sheet1.csv')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

today = pd.Timestamp.today()

customer_stats = df.groupby('Names').agg(
    total_orders=('Date', 'count'),
    total_spent=('Selling Price', 'sum'),
    avg_order_value=('Selling Price', 'mean'),
    last_purchase=('Date', 'max'),
    first_purchase=('Date', 'min')
).reset_index()

customer_stats['Days Since Last Purchase'] = (today - customer_stats['last_purchase']).dt.days

top_category = df.groupby('Names').apply(
    lambda x: x.groupby('Category')['Selling Price'].sum().idxmax()
).reset_index()
top_category.columns = ['Names', 'Favourite Category']

customer_stats = customer_stats.merge(top_category, on='Names')

def avg_gap(dates):
    sorted_dates = sorted(dates)
    if len(sorted_dates) < 2:
        return None
    gaps = [(sorted_dates[i+1] - sorted_dates[i]).days
            for i in range(len(sorted_dates)-1)]
    return round(sum(gaps) / len(gaps))

gaps = df.groupby('Names')['Date'].apply(avg_gap).reset_index()
gaps.columns = ['Names', 'Avg Days Between Purchases']
customer_stats = customer_stats.merge(gaps, on='Names')
def followup_status(row):
    if row['total_orders'] == 1:
        if row['Days Since Last Purchase'] > 30:
            return 'MESSAGE NOW — single buyer, re-engage'
        return 'Too recent — wait'
    avg = row['Avg Days Between Purchases']
    if avg is None:
        return 'Unknown pattern'
    days_since = row['Days Since Last Purchase']
    overdue = days_since - avg
    if overdue > 30:
        return f'🔴 OVERDUE by {overdue} days — message now'
    elif overdue > 0:
        return f'🟠 DUE — message this week'
    else:
        days_left = abs(overdue)
        return f'🟢 Not yet — message in {days_left} days'

customer_stats['Follow Up Status'] = customer_stats.apply(followup_status, axis=1)
customer_stats['Potential Revenue'] = customer_stats['avg_order_value']
priority = customer_stats[
    customer_stats['Follow Up Status'].str.contains('OVERDUE|DUE|MESSAGE NOW', na=False)
].sort_values('total_spent', ascending=False)

print("=" * 70)
print("PROBLEM 4: FOLLOW-UP TIMING REPORT")
print("=" * 70)

print("\n🎯 CUSTOMERS TO MESSAGE RIGHT NOW (sorted by value):")
print(f"{'Name':<25} {'Last Buy':<10} {'Fav Category':<15} {'Avg Order':<12} {'Status'}")
print("-" * 70)
for _, row in priority.head(20).iterrows():
    print(f"{row['Names']:<25} {row['Days Since Last Purchase']:<10} {row['Favourite Category']:<15} ₹{row['avg_order_value']:<10,.0f} {row['Follow Up Status']}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
total_potential = priority.head(20)['avg_order_value'].sum()
print(f"  Customers to contact today:     {len(priority)}")
print(f"  Potential revenue if 50% respond: ₹{total_potential * 0.5:,.0f}")
print(f"  Cost of outreach:               ₹0 (WhatsApp)")
print(f"\n  What to say to each customer:")
print(f"  → Suits buyer:    'New suits arrived in stock'")
print(f"  → Saree buyer:    'Festive collection arrived'")
print(f"  → Anarkali buyer: 'New Designs arrived'")