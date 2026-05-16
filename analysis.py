import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('Purchases - Sheet1.csv')

# Fix date column
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Month'] = df['Date'].dt.strftime('%b %Y')
df['Month_num'] = df['Date'].dt.to_period('M')

# ── 1. SUMMARY STATS ──────────────────────────────────────
print("=== SUMMARY STATS ===")
print(f"Total Revenue:  ₹{df['Selling Price'].sum():,}")
print(f"Total Profit:   ₹{df['Profit'].sum():,}")
print(f"Total Orders:   {len(df)}")
print(f"Unique Customers: {df['Names'].nunique()}")
print(f"Avg Order Value: ₹{df['Selling Price'].mean():,.0f}")
print(f"Avg Profit/Order: ₹{df['Profit'].mean():,.0f}")

print("\n=== SALES BY CATEGORY ===")
print(df.groupby('Category')[['Selling Price','Profit']].sum().sort_values('Profit', ascending=False))

print("\n=== TOP 10 CUSTOMERS ===")
print(df.groupby('Names')['Selling Price'].sum().sort_values(ascending=False).head(10))

# ── 2. CHART 1: Monthly Revenue ───────────────────────────
monthly = df.groupby('Month_num')['Selling Price'].sum().reset_index()
monthly['Month_num'] = monthly['Month_num'].astype(str)

plt.figure(figsize=(12, 5))
plt.plot(monthly['Month_num'], monthly['Selling Price'], marker='o', color='teal', linewidth=2)
plt.title("Monthly Revenue", fontsize=16)
plt.xlabel("Month")
plt.ylabel("Revenue (₹)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('chart_monthly_revenue.png', dpi=150)
plt.close()
print("\n✅ Saved: chart_monthly_revenue.png")

# ── 3. CHART 2: Revenue by Category ──────────────────────
cat = df.groupby('Category')['Selling Price'].sum().sort_values()

plt.figure(figsize=(10, 5))
cat.plot(kind='barh', color='steelblue')
plt.title("Revenue by Category", fontsize=16)
plt.xlabel("Revenue (₹)")
plt.tight_layout()
plt.savefig('chart_category_revenue.png', dpi=150)
plt.close()
print("✅ Saved: chart_category_revenue.png")

# ── 4. CHART 3: Profit by Category ───────────────────────
cat_profit = df.groupby('Category')['Profit'].sum().sort_values()

plt.figure(figsize=(10, 5))
cat_profit.plot(kind='barh', color='mediumseagreen')
plt.title("Profit by Category", fontsize=16)
plt.xlabel("Profit (₹)")
plt.tight_layout()
plt.savefig('chart_category_profit.png', dpi=150)
plt.close()
print("✅ Saved: chart_category_profit.png")

# ── 5. CHART 4: Top 10 Customers ─────────────────────────
top_customers = df.groupby('Names')['Selling Price'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 5))
top_customers.plot(kind='bar', color='coral')
plt.title("Top 10 Customers by Revenue", fontsize=16)
plt.ylabel("Revenue (₹)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('chart_top_customers.png', dpi=150)
plt.close()
print("✅ Saved: chart_top_customers.png")

print("\n🎉 All done! Check your folder for the 4 chart images.")