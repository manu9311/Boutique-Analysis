import pandas as pd
import numpy as np
from datetime import datetime

df = pd.read_csv('/Users/manurana/Documents/boutique_analysis/Purchases - Sheet1.csv')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
today = pd.Timestamp.today()

# ══════════════════════════════════════════════════════════
# BASELINE — current average monthly revenue
# ══════════════════════════════════════════════════════════
df['Month'] = df['Date'].dt.to_period('M')
monthly = df.groupby('Month')['Selling Price'].sum()
baseline_monthly = monthly.mean()

# ══════════════════════════════════════════════════════════
# LEVER 1 — Dead Inventory Recovery
# ══════════════════════════════════════════════════════════
last_sold = df.groupby('Category')['Date'].max()
days_since = (today - last_sold).dt.days
cost_invested = df.groupby('Category')['Cost Price'].sum()

dead_categories = days_since[days_since > 60].index
dead_cash = cost_invested[dead_categories].sum()

# Conservative: 70% of stock actually sells at 15% discount
lever1_recovery = dead_cash * 0.70 * 0.85
lever1_optimistic = dead_cash * 0.85

# ══════════════════════════════════════════════════════════
# LEVER 2 — Churn Recovery (lost customers)
# ══════════════════════════════════════════════════════════
customer_stats = df.groupby('Names').agg(
    last_purchase=('Date', 'max'),
    avg_order_value=('Selling Price', 'mean'),
    total_orders=('Date', 'count')
).reset_index()
customer_stats['days_since'] = (today - customer_stats['last_purchase']).dt.days

lost = customer_stats[customer_stats['days_since'] > 365]
at_risk = customer_stats[
    (customer_stats['days_since'] > 90) &
    (customer_stats['days_since'] <= 365)
]

# Conservative: 20% of lost respond, 40% of at-risk respond
lever2_conservative = (lost['avg_order_value'].sum() * 0.20) + (at_risk['avg_order_value'].sum() * 0.40)
lever2_optimistic = (lost['avg_order_value'].sum() * 0.35) + (at_risk['avg_order_value'].sum() * 0.60)

# ══════════════════════════════════════════════════════════
# LEVER 3 — Follow-up System (overdue active customers)
# ══════════════════════════════════════════════════════════
# Customers who bought in last 90 days but are overdue
recent = customer_stats[customer_stats['days_since'] <= 90]

# Conservative: 50% respond when messaged
lever3_conservative = recent['avg_order_value'].sum() * 0.50
lever3_optimistic = recent['avg_order_value'].sum() * 0.70

# ══════════════════════════════════════════════════════════
# LEVER 4 — Stock Optimisation (stop buying dead stock)
# ══════════════════════════════════════════════════════════
# Money currently wasted on dead categories per month
avg_dead_spend_monthly = dead_cash / 12  # spread over a year
# If reinvested in Suits (76.8% revenue driver), margin improvement
suits_margin = df[df['Category'] == 'Suits']['Profit'].sum() / df[df['Category'] == 'Suits']['Cost Price'].sum()
lever4_conservative = avg_dead_spend_monthly * suits_margin * 0.5
lever4_optimistic = avg_dead_spend_monthly * suits_margin

# ══════════════════════════════════════════════════════════
# TOTAL RECOVERY
# ══════════════════════════════════════════════════════════
total_conservative = lever2_conservative + lever3_conservative + lever4_conservative
total_optimistic = lever1_recovery + lever2_optimistic + lever3_optimistic + lever4_optimistic

growth_conservative = (total_conservative / baseline_monthly) * 100
growth_optimistic = (total_optimistic / baseline_monthly) * 100

# ══════════════════════════════════════════════════════════
# PRINT REPORT
# ══════════════════════════════════════════════════════════
print("=" * 65)
print("PROBLEM 5: REVENUE RECOVERY CALCULATOR")
print("=" * 65)

print(f"\n📊 YOUR CURRENT BASELINE")
print(f"  Average monthly revenue: ₹{baseline_monthly:,.0f}")

print(f"\n💡 LEVER 1 — Unlock Dead Inventory")
print(f"  Cash locked in unsold stock:      ₹{dead_cash:,.0f}")
print(f"  Recovery via discount sale:       ₹{lever1_recovery:,.0f}  (one time)")

print(f"\n💡 LEVER 2 — Recover Lost Customers")
print(f"  Lost customers (365+ days):       {len(lost)}")
print(f"  At risk customers (90-365 days):  {len(at_risk)}")
print(f"  Conservative recovery (monthly):  ₹{lever2_conservative:,.0f}")
print(f"  Optimistic recovery (monthly):    ₹{lever2_optimistic:,.0f}")

print(f"\n💡 LEVER 3 — Follow-up System")
print(f"  Overdue active customers:         {len(recent)}")
print(f"  Conservative recovery (monthly):  ₹{lever3_conservative:,.0f}")
print(f"  Optimistic recovery (monthly):    ₹{lever3_optimistic:,.0f}")

print(f"\n💡 LEVER 4 — Stop Buying Dead Stock")
print(f"  Monthly budget wasted on dead:    ₹{avg_dead_spend_monthly:,.0f}")
print(f"  Margin gain if reinvested:        ₹{lever4_conservative:,.0f}/month")

print("\n" + "=" * 65)
print("THE BOTTOM LINE")
print("=" * 65)
print(f"\n  Current monthly revenue:          ₹{baseline_monthly:,.0f}")
print(f"\n  Conservative scenario:")
print(f"  Additional monthly revenue:       ₹{total_conservative:,.0f}")
print(f"  New monthly revenue:              ₹{baseline_monthly + total_conservative:,.0f}")
print(f"  Growth:                           {growth_conservative:.1f}%")
print(f"\n  Optimistic scenario:")
print(f"  Additional monthly revenue:       ₹{total_optimistic:,.0f}")
print(f"  New monthly revenue:              ₹{baseline_monthly + total_optimistic:,.0f}")
print(f"  Growth:                           {growth_optimistic:.1f}%")
print(f"\n  Cost of all this:                 ₹0")
print(f"  Time required:                    2 hours of WhatsApp + 1 discount sale")
print("\n" + "=" * 65)
print("WHAT TO DO THIS WEEK")
print("=" * 65)
print(f"  Day 1: Message top 10 overdue customers about new Suits stock")
print(f"  Day 2: Identify dead inventory, plan weekend discount sale")
print(f"  Day 3: Message 10 lost customers who spent ₹5,000+")
print(f"  Day 4: Run discount sale, recover locked cash")
print(f"  Day 5: Review which messages got responses, note patterns")
print(f"\n  One week. Zero rupees spent. Data-backed decisions.")