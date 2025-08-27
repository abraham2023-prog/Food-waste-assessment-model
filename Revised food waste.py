import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Continue building the dashboard from where you left off

# Analysis parameters in sidebar
with st.sidebar:
    st.subheader("Analysis Parameters")
    
    # Product selection
    if 'product' in df.columns:
        product_options = df['product'].unique()
    else:
        product_options = ["Unknown Product"]
        df['product'] = "Unknown Product"
    
    selected_products = st.multiselect(
        "Select Products to Analyze",
        options=product_options,
        default=product_options[:3] if len(product_options) > 3 else product_options
    )
    
    # Year selection
    if 'Year' in df.columns:
        year_options = sorted(df['Year'].unique())
        selected_years = st.multiselect(
            "Select Years",
            options=year_options,
            default=year_options
        )
    else:
        selected_years = []
    
    # Category selection
    if 'Category' in df.columns:
        category_options = sorted(df['Category'].unique())
        selected_categories = st.multiselect(
            "Select Categories",
            options=category_options,
            default=category_options
        )
    else:
        selected_categories = []

# Apply filters
filtered_df = df.copy()
if selected_products:
    filtered_df = filtered_df[filtered_df['product'].isin(selected_products)]
if selected_years:
    filtered_df = filtered_df[filtered_df['Year'].isin(selected_years)]
if selected_categories:
    filtered_df = filtered_df[filtered_df['Category'].isin(selected_categories)]

# ================= KPI Metrics =================
st.markdown('<p class="section-header">📌 Key Metrics</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Production (tons)", f"{filtered_df['production'].sum():,.0f}")
with col2:
    st.metric("Total Waste (tons)", f"{filtered_df['waste'].sum():,.0f}")
with col3:
    st.metric("Waste Rate (%)", f"{filtered_df['waste_rate'].mean()*100:.2f}%")
with col4:
    st.metric("Total Waste Value (Baht)", f"{filtered_df['waste_value'].sum():,.0f}")

# ================= Waste by Category =================
st.markdown('<p class="section-header">🍽 Waste by Category</p>', unsafe_allow_html=True)

if 'Category' in filtered_df.columns:
    waste_by_category = filtered_df.groupby('Category')['waste'].sum().reset_index()
    fig1 = px.bar(
        waste_by_category,
        x='Category',
        y='waste',
        title="Total Waste by Category",
        text_auto=True,
        color='Category'
    )
    st.plotly_chart(fig1, use_container_width=True)

# ================= Waste Trend =================
st.markdown('<p class="section-header">📈 Waste Trend Over Time</p>', unsafe_allow_html=True)

if 'Year' in filtered_df.columns and 'Month' in filtered_df.columns:
    filtered_df['Date'] = pd.to_datetime(filtered_df['Year'].astype(str) + '-' + filtered_df['Month'].astype(str) + '-01')
    waste_trend = filtered_df.groupby('Date')['waste'].sum().reset_index()
    fig2 = px.line(waste_trend, x='Date', y='waste', title="Waste Trend Over Time")
    st.plotly_chart(fig2, use_container_width=True)

# ================= Top 10 High-Risk Products =================
st.markdown('<p class="section-header">⚠️ Top 10 High-Risk Products</p>', unsafe_allow_html=True)

if 'product' in filtered_df.columns:
    high_risk_products = (
        filtered_df.groupby('product')['waste']
        .sum()
        .reset_index()
        .sort_values(by='waste', ascending=False)
        .head(10)
    )
    fig3 = px.bar(
        high_risk_products,
        x='product',
        y='waste',
        title="Top 10 Products by Waste",
        text_auto=True,
        color='waste'
    )
    st.plotly_chart(fig3, use_container_width=True)

# ================= Inventory Analysis =================
st.markdown('<p class="section-header">📦 Inventory Analysis</p>', unsafe_allow_html=True)

if 'InventoryTurnover' in filtered_df.columns and 'Category' in filtered_df.columns:
    fig4 = px.box(
        filtered_df,
        x='Category',
        y='InventoryTurnover',
        title="Inventory Turnover Distribution by Category",
        color='Category'
    )
    st.plotly_chart(fig4, use_container_width=True)

# ================= Download Processed Data =================
st.markdown('<p class="section-header">⬇️ Download Processed Data</p>', unsafe_allow_html=True)

csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='processed_food_waste.csv',
    mime='text/csv',
)

