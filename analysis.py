import pandas as pd
df = pd.read_csv('Purchases - Sheet1.csv')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Month_num'] = df['Date'].dt.to_period('M')
print(df.head())
#Total Revenue:-
Total_Revenue = df['Selling Price'].sum()
print(f"Total Revenue is : {Total_Revenue: ,}")
#Total Profit:-
Total_Profit = df['Profit'].sum()
print(f"Total_Profit is: {Total_Profit: ,}")
#Total Orders :-
Total_Orders = len(df)
print(f"Total_Orders is: {Total_Orders}")
#Total Customers:-
Total_Customers = df['Names'].nunique()
print(f"Total_Customers is: {Total_Customers}")

#Sales by category:-
Sales_by_Category = df.groupby('Category')[['Selling Price', 'Profit']].sum().sort_values(by='Profit', ascending=False)
print(Sales_by_Category)

#Top10 customers by revenue :-
Top_10 = df.groupby('Names')[['Selling Price']].sum().sort_values(by='Selling Price', ascending=False)
print(Top_10.head(10))

import matplotlib.pyplot as plt

# Chart 1: Monthly Revenue
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

# Chart 2: Revenue by Category
cat = df.groupby('Category')['Selling Price'].sum().sort_values()
plt.figure(figsize=(10, 5))
cat.plot(kind='barh', color='steelblue')
plt.title("Revenue by Category", fontsize=16)
plt.xlabel("Revenue (₹)")
plt.tight_layout()
plt.savefig('chart_category_revenue.png', dpi=150)
plt.close()

# Chart 3: Profit by Category
cat_profit = df.groupby('Category')['Profit'].sum().sort_values()
plt.figure(figsize=(10, 5))
cat_profit.plot(kind='barh', color='mediumseagreen')
plt.title("Profit by Category", fontsize=16)
plt.xlabel("Profit (₹)")
plt.tight_layout()
plt.savefig('chart_category_profit.png', dpi=150)
plt.close()

# Chart 4: Top 10 Customers
top_customers = df.groupby('Names')['Selling Price'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 5))
top_customers.plot(kind='bar', color='coral')
plt.title("Top 10 Customers by Revenue", fontsize=16)
plt.ylabel("Revenue (₹)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('chart_top_customers.png', dpi=150)
plt.close()