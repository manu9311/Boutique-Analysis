import pandas as pd
from datetime import datetime

df = pd.read_csv('/Users/manurana/Documents/boutique_analysis/Purchases - Sheet1.csv')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
today = pd.Timestamp.today()

customer_stats = df.groupby('Names').agg(
    last_purchase=('Date', 'max'),
    total_orders=('Date', 'count'),
    total_spent=('Selling Price', 'sum'),
    avg_order_value=('Selling Price', 'mean')
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
    gaps = [(sorted_dates[i+1] - sorted_dates[i]).days for i in range(len(sorted_dates)-1)]
    return sum(gaps) / len(gaps)

gaps = df.groupby('Names')['Date'].apply(avg_gap).reset_index()
gaps.columns = ['Names', 'Avg Days Between Purchases']

customer_stats = customer_stats.merge(gaps, on='Names')


def churn_flag(row):
    days = row['Days Since Last Purchase']
    if days > 365:
        return '🔴 LOST'
    elif days > 180:
        return '🟠 AT RISK'
    elif days > 90:
        return '🟡 INACTIVE'
    else:
        return '🟢 ACTIVE'

customer_stats['Status'] = customer_stats.apply(churn_flag, axis=1)

at_risk = customer_stats[customer_stats['Status'].isin(['🟠 AT RISK', '🔴 LOST'])]
revenue_at_risk = at_risk['avg_order_value'].sum()

print("=" * 65)
print("PROBLEM 2: CUSTOMER CHURN RISK REPORT")
print("=" * 65)

for status in ['🔴 LOST', '🟠 AT RISK', '🟡 ONE-TIME', '🟢 ACTIVE']:
    group = customer_stats[customer_stats['Status'] == status]
    if len(group) > 0:
        print(f"\n{status} — {len(group)} customers")
        print(group[['Names', 'Favourite Category', 'Days Since Last Purchase', 'total_spent', 'avg_order_value']].sort_values('total_spent', ascending=False).to_string(index=False))

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
lost = customer_stats[customer_stats['Status'] == '🔴 LOST']
at_risk_only = customer_stats[customer_stats['Status'] == '🟠 AT RISK']

print(f"🔴 Lost customers:      {len(lost)} people — ₹{lost['total_spent'].sum():,.0f} in past revenue")
print(f"🟠 At risk customers:   {len(at_risk_only)} people — ₹{at_risk_only['total_spent'].sum():,.0f} in past revenue")
print(f"\nIf you recover just 30% of at-risk customers:")
print(f"  Potential revenue:  ₹{at_risk_only['total_spent'].sum() * 0.3:,.0f}")
print(f"  Cost to try:        ₹0  (just a WhatsApp message)")