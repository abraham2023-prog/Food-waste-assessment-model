import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Food Waste Dashboard", layout="wide")

st.title("Food Waste Analysis Dashboard")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    # Load and clean data
    df = pd.read_csv(uploaded_file, encoding='utf-8')

    # --- Numeric columns cleaning ---
    numeric_columns = ['Begin month\ninventory', 'Production', 'Domestic', 'Export', 'Total',
                       'Shipment value\n(thousand baht)', 'Month-end \ninventory', 'Capacity']

    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Month'] = pd.to_numeric(df['Month'], errors='coerce')

    # --- Unit mapping ---
    unit_mapping = {
        # Products in liters
        'Table condiments': 'liter',
        'Soy sauce, fermented soybean paste, dark soy sauce': 'liter',
        'Soy sauce, fermented soybean paste, light soy sauce': 'liter',
        'small condiments or seasoning dispensers': 'liter',
        # Products in thousand liters
        'Soy milk': 'thousand_liter',
        # Products in tons (everything else)
        'Cake': 'ton',
        'Cookie': 'ton',
        'Dried fruits and vegetables': 'ton',
        'Frozen and chilled chicken meat': 'ton',
        'Frozen and chilled pork': 'ton',
        'Frozen prepared food': 'ton',
        'Instant noodles': 'ton',
        'Molasses': 'ton',
        'Monosodium glutamate': 'ton',
        'Other baked goods (pizza, donuts, sandwich bread)': 'ton',
        'Other crispy snacks (Corn chips, prawn crackers, etc)': 'ton',
        'Pet feed': 'ton',
        'Premix': 'ton',
        'Ready made chicken feed': 'ton',
        'Ready made duck feed': 'ton',
        'Ready made fish feed': 'ton',
        'Ready made pet feed': 'ton',
        'Ready made pig feed': 'ton',
        'Ready made shrimp feed': 'ton',
        'Ready to cook meals (others)': 'ton',
        'Ready-made pig feed': 'ton',
        'Toasted bread/Cracker/Biscuit': 'ton',
        'Wafer biscuit': 'ton',
        'Yogurt': 'ton',
        'bacon': 'ton',
        'biscuits/crackers': 'ton',
        'cake': 'ton',
        'canned pickles': 'ton',
        'canned pineapple': 'ton',
        'canned sardines': 'ton',
        'canned sweet corn': 'ton',
        'canned tuna': 'ton',
        'coconut milk': 'ton',
        'dried fruits & vegetables': 'ton',
        'frozen fish': 'ton',
        'frozen fruits and vegetables': 'ton',
        'frozen shrimp': 'ton',
        'frozen squid': 'ton',
        'ham': 'ton',
        'ice cream': 'ton',
        'instant noodles': 'ton',
        'minced fish meat': 'ton',
        'other baked goods': 'ton',
        'other canned fruits': 'ton',
        'other crispy baked snacks': 'ton',
        'pure white sugar': 'ton',
        'raw sugar': 'ton',
        'ready made feed for other livestock': 'ton',
        'sausage': 'ton',
        'seasoned chicken meat': 'ton',
        'sweep/suck': 'ton',
        'white sugar': 'ton',
        'Tapioca flour': 'ton',
        'Soy sauce, fermented soybean paste, light soy sauce ': 'ton'
    }

    df['Unit'] = df['Product'].map(unit_mapping)
    df['Unit'] = df['Unit'].fillna('ton')

    # --- Convert all to tons ---
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

    # --- Category mapping ---
    # (Your existing category mapping logic remains unchanged)
    # Add waste calculation and inventory metrics here (copy from your code)

    # --- Dashboard metrics ---
    st.subheader("Summary Metrics")
    st.write(f"Total Records: {len(df)}")
    st.write(f"Total Products Analyzed: {len(df['Product'].unique())}")
    st.write(f"Year Range: {df['Year'].min()} - {df['Year'].max()}")

    # Category-wise analysis
    st.subheader("Category-wise Analysis")
    category_analysis = df.groupby('Category').agg({
        'PotentialWaste': 'sum',
        'InventoryShrinkage': 'sum',
        'Production': 'sum',
        'Total': 'sum',
        'MonthsOfInventory': 'mean'
    }).round(2)
    category_analysis['WastePercentage'] = (category_analysis['PotentialWaste'] /
                                           category_analysis['Production'] * 100).round(2)
    st.dataframe(category_analysis)

    # Top 10 high-waste products
    st.subheader("Top 10 High-Risk Products")
    high_waste_products = df.groupby(['Product', 'Unit']).agg({
        'PotentialWaste': 'sum',
        'Production': 'sum',
        'MonthsOfInventory': 'mean'
    }).nlargest(10, 'PotentialWaste')
    high_waste_products['WastePercentage'] = (high_waste_products['PotentialWaste'] /
                                             high_waste_products['Production'] * 100).round(2)
    st.dataframe(high_waste_products)

    # --- Visualizations ---
    st.subheader("Visualizations")

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.barplot(x=category_analysis.index, y='PotentialWaste', data=category_analysis, ax=ax1)
    ax1.set_ylabel("Potential Waste (tons)")
    ax1.set_xlabel("Category")
    ax1.set_title("Potential Waste by Category")
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.barplot(x=high_waste_products.index.get_level_values(0), y='PotentialWaste', data=high_waste_products, ax=ax2)
    ax2.set_ylabel("Potential Waste (tons)")
    ax2.set_xlabel("Product")
    ax2.set_title("Top 10 High-Risk Products")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    st.success("Dashboard generated successfully!")



