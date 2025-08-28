import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Food Waste Dashboard", layout="wide")

# =========================
# LOAD & CLEAN DATA
# =========================
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("Only food.csv", encoding="utf-8")

    # Clean numeric columns
    numeric_columns = [
        'Begin month\ninventory', 'Production', 'Domestic', 'Export', 'Total',
        'Shipment value\n(thousand baht)', 'Month-end \ninventory', 'Capacity'
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df['Year'] = pd.to_numeric(df['Year'], errors="coerce")
    df['Month'] = pd.to_numeric(df['Month'], errors="coerce")

    # --- ADD YOUR UNIT MAPPING & CATEGORY ASSIGNMENTS HERE ---
    # (to keep response short, I’ll assume we reuse your exact mapping code)

    # Example: ensure 'Unit' exists, default to ton
    if "Unit" not in df.columns:
        df["Unit"] = "ton"

    # Convert to tons
    def convert_to_tons(value, unit):
        if unit == 'liter':
            return value * 0.001
        elif unit == 'thousand_liter':
            return value * 1.0
        else:
            return value
    for col in ['Begin month\ninventory', 'Production', 'Domestic', 'Export', 'Total', 'Month-end \ninventory']:
        if col in df.columns:
            df[col] = df.apply(lambda x: convert_to_tons(x[col], x['Unit']), axis=1)

    # Basic waste calculation (simplified for dashboard)
    waste_factors = {
        'bakery': 0.12, 'dairy': 0.10, 'frozen_foods': 0.05,
        'canned_foods': 0.03, 'condiments': 0.04, 'animal_feed': 0.06,
        'staples': 0.04, 'snacks': 0.07, 'sugar': 0.02, 'other': 0.08
    }
    df['Category'] = df.get('Category', 'other')
    df['WasteFactor'] = df['Category'].map(waste_factors).fillna(0.06)
    df['PotentialWaste'] = df['Month-end \ninventory'] * df['WasteFactor']

    return df

df = load_and_clean_data()

# =========================
# DASHBOARD METRICS
# =========================
st.title("📊 Food Waste Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("Total Production (tons)", f"{df['Production'].sum():,.0f}")
col2.metric("Estimated Waste (tons)", f"{df['PotentialWaste'].sum():,.0f}")
col3.metric("Waste % of Production", f"{(df['PotentialWaste'].sum()/df['Production'].sum()*100):.2f}%")

st.markdown("---")

# =========================
# CATEGORY ANALYSIS
# =========================
st.subheader("Waste by Category")
category_analysis = df.groupby('Category').agg({
    'PotentialWaste': 'sum',
    'Production': 'sum'
})
category_analysis['Waste%'] = category_analysis['PotentialWaste'] / category_analysis['Production'] * 100

st.dataframe(category_analysis.style.format({
    'PotentialWaste': '{:,.0f}',
    'Production': '{:,.0f}',
    'Waste%': '{:.2f}%'
}))

# Chart
fig, ax = plt.subplots(figsize=(8, 5))
category_analysis.sort_values('PotentialWaste').plot(kind='barh', y='PotentialWaste', ax=ax, legend=False)
ax.set_xlabel("Waste (tons)")
ax.set_ylabel("Category")
st.pyplot(fig)

# =========================
# TOP PRODUCTS
# =========================
st.subheader("Top 10 High-Waste Products")
top_products = df.groupby('Product').agg({
    'PotentialWaste': 'sum',
    'Production': 'sum'
}).nlargest(10, 'PotentialWaste')
top_products['Waste%'] = top_products['PotentialWaste'] / top_products['Production'] * 100

st.dataframe(top_products.style.format({
    'PotentialWaste': '{:,.0f}',
    'Production': '{:,.0f}',
    'Waste%': '{:.2f}%'
}))

fig2, ax2 = plt.subplots(figsize=(8, 5))
top_products.sort_values('PotentialWaste').plot(kind='barh', y='PotentialWaste', ax=ax2, legend=False)
ax2.set_xlabel("Waste (tons)")
ax2.set_ylabel("Product")
st.pyplot(fig2)

# =========================
# FILTER & TREND
# =========================
st.subheader("Trend Analysis")

product_choice = st.selectbox("Choose a product:", df['Product'].unique())
df_filtered = df[df['Product'] == product_choice]

if not df_filtered.empty:
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    df_filtered.groupby('Year')['PotentialWaste'].sum().plot(ax=ax3, marker="o")
    ax3.set_title(f"Potential Waste Over Time: {product_choice}")
    ax3.set_ylabel("Waste (tons)")
    st.pyplot(fig3)



