# customer_sales_analysis.py

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------
# Load Dataset
# ---------------------------------------

file_path = "/Users/suriya/Documents/ai for everyone/q1/sales_data.csv"

df = pd.read_csv(file_path, encoding="latin1")

print("=" * 60)
print("ORIGINAL DATASET")
print("=" * 60)
print(df.head())

# ---------------------------------------
# Check Missing Values
# ---------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

# ---------------------------------------
# Data Cleaning
# ---------------------------------------

# Remove duplicate rows
df = df.drop_duplicates()

# Fill missing values (if any)
df["Customer Name"] = df["Customer Name"].fillna("Unknown")
df["Category"] = df["Category"].fillna("Others")
df["Sales"] = df["Sales"].fillna(df["Sales"].median())
df["Quantity"] = df["Quantity"].fillna(1)

# Convert Order Date to datetime
df["Order Date"] = pd.to_datetime(df["Order Date"])

# Save cleaned dataset
df.to_csv("cleaned_sales_data.csv", index=False)

print("\nCleaned dataset saved as 'cleaned_sales_data.csv'")

# ---------------------------------------
# Summary Statistics
# ---------------------------------------

print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

print(df.describe())

# ---------------------------------------
# Revenue by Product Category
# ---------------------------------------

category_revenue = (
    df.groupby("Category")["Sales"]
      .sum()
      .sort_values(ascending=False)
)

print("\n" + "=" * 60)
print("TOTAL REVENUE BY CATEGORY")
print("=" * 60)

print(category_revenue)

# ---------------------------------------
# Top 10 Customers
# ---------------------------------------

top_customers = (
    df.groupby("Customer Name")["Sales"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
)

print("\n" + "=" * 60)
print("TOP 10 CUSTOMERS")
print("=" * 60)

print(top_customers)

# ---------------------------------------
# Monthly Sales
# ---------------------------------------

df["Month"] = df["Order Date"].dt.to_period("M").astype(str)

monthly_sales = (
    df.groupby("Month")["Sales"]
      .sum()
)

# ---------------------------------------
# Plot 1 : Monthly Sales
# ---------------------------------------

plt.figure(figsize=(12,6))

plt.plot(monthly_sales.index,
         monthly_sales.values,
         marker='o')

plt.title("Monthly Sales")

plt.xlabel("Month")

plt.ylabel("Revenue")

plt.xticks(rotation=45)

plt.grid(True)

plt.tight_layout()

plt.savefig("monthly_sales.png", dpi=300)

plt.show()

# ---------------------------------------
# Plot 2 : Revenue by Product Category
# ---------------------------------------

plt.figure(figsize=(8,6))

category_revenue.plot(kind="bar")

plt.title("Revenue by Product Category")

plt.xlabel("Category")

plt.ylabel("Revenue")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig("category_revenue.png", dpi=300)

plt.show()

# ---------------------------------------
# Plot 3 : Top 10 Customers
# ---------------------------------------

plt.figure(figsize=(12,6))

top_customers.sort_values().plot(kind="barh")

plt.title("Top 10 Customers by Sales")

plt.xlabel("Revenue")

plt.ylabel("Customer")

plt.tight_layout()

plt.savefig("top10_customers.png", dpi=300)

plt.show()

print("\nAnalysis completed successfully!")

print("\nGenerated Files:")
print("1. cleaned_sales_data.csv")
print("2. monthly_sales.png")
print("3. category_revenue.png")
print("4. top10_customers.png")
