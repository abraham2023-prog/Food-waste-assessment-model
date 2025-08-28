# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Food Waste Dashboard", layout="wide")

# ========= LOAD DATA =========
@st.cache_data
def load_data():
    df = pd.read_csv("Only food.csv")
    category_analysis = pd.read_csv("Only food.csv", index_col=0)
    return df, category_analysis

df, category_analysis = load_data()

st.title("📊 Food Waste Analysis Dashboard")

# ========= SUMMARY METRICS =========
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Production (tons)", f"{df['Production'].sum():,.0f}")
col2.metric("Estimated Waste (tons)", f"{df['PotentialWaste'].sum():,.0f}")
col3.metric("Inventory Shrinkage (tons)", f"{df['InventoryShrinkage'].sum():,.0f}")
col4.metric("Products Analyzed", f"{df['Product'].nunique()}")

st.markdown("---")

# ========= CATEGORY ANALYSIS =========
st.subheader("Waste by Category")
fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(data=category_analysis.reset_index(),
            x="Category", y="PotentialWaste", palette="viridis", ax=ax)
ax.set_title("Potential Waste by Category")
ax.set_ylabel("Waste (tons)")
st.pyplot(fig)

# ========= WASTE % OF PRODUCTION =========
st.subheader("Waste as % of Production")
fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(data=category_analysis.reset_index(),
            x="Category", y="WastePercentage", palette="magma", ax=ax)
ax.set_title("Waste Percentage by Category")
ax.set_ylabel("Waste %")
st.pyplot(fig)

# ========= TOP PRODUCTS =========
st.subheader("Top 10 High-Risk Products")
high_waste_products = df.groupby(['Product']).agg({
    'PotentialWaste': 'sum',
    'Production': 'sum',
    'MonthsOfInventory': 'mean'
}).nlargest(10, 'PotentialWaste')

high_waste_products['WastePercentage'] = (high_waste_products['PotentialWaste'] /
                                         high_waste_products['Production'] * 100).round(2)

st.dataframe(high_waste_products)

# ========= INTERACTIVE FILTER =========
st.subheader("Interactive Product Analysis")
product = st.selectbox("Select a Product:", df['Product'].unique())

prod_data = df[df['Product'] == product]
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(prod_data['Year'].astype(str) + "-" + prod_data['Month'].astype(str),
        prod_data['PotentialWaste'], marker="o", label="Potential Waste")
ax.plot(prod_data['Year'].astype(str) + "-" + prod_data['Month'].astype(str),
        prod_data['Production'], marker="x", label="Production")
ax.legend()
ax.set_title(f"Trend for {product}")
ax.set_xlabel("Time")
ax.set_ylabel("Tons")
plt.xticks(rotation=45)
st.pyplot(fig)


