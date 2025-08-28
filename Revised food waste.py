import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import io
import zipfile
from PIL import Image
import hashlib


# Function to convert matplotlib figure to bytes
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', transparent=True)
    buf.seek(0)
    return buf

# Function to generate unique keys based on content
def generate_key(prefix, data=None):
    if data is not None:
        # Create a hash based on the data to ensure uniqueness
        data_str = str(data).encode()
        hash_str = hashlib.md5(data_str).hexdigest()[:8]
        return f"{prefix}_{hash_str}"
    return f"{prefix}_{hashlib.md5(str(np.random.rand()).encode()).hexdigest()[:8]}"
    

# Set page configuration
st.set_page_config(
    page_title="Food Waste Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# File upload function
def load_and_process_data(uploaded_file):
    # Read the uploaded file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    else:
        # Handle other file types if needed
        st.error("Please upload a CSV file")
        return None
    
    # Your existing data processing code here
    # Clean numeric columns
    numeric_columns = ['Begin month\ninventory', 'Production', 'Domestic', 'Export', 'Total',
                       'Shipment value\n(thousand baht)', 'Month-end \ninventory', 'Capacity']

    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert other columns
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Month'] = pd.to_numeric(df['Month'], errors='coerce')

    # PRECISE UNIT MAPPING BASED ON YOUR PRODUCT LIST
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

    # Apply unit mapping
    df['Unit'] = df['Product'].map(unit_mapping)

    # Check for any products without unit mapping
    missing_units = df[df['Unit'].isna()]
    if not missing_units.empty:
        print("Products missing unit mapping:")
        print(missing_units['Product'].unique())
        # Fill missing units with 'ton' as default
        df['Unit'] = df['Unit'].fillna('ton')

    # CONVERT ALL VALUES TO TONS FOR CONSISTENT ANALYSIS
    # Using industry-standard conversion factors
    def convert_to_tons(value, unit):
        if unit == 'liter':
            # For most food products: 1 liter ≈ 1 kg = 0.001 tons
            # (Adjust if you have specific density information)
            return value * 0.001
        elif unit == 'thousand_liter':
            # 1000 liters ≈ 1 ton (for most liquids)
            return value * 1.0
        else:  # ton - no conversion needed
            return value

    # Convert all numeric columns to tons
    for col in ['Begin month\ninventory', 'Production', 'Domestic', 'Export', 'Total', 'Month-end \ninventory']:
        if col in df.columns:
            df[col] = df.apply(lambda x: convert_to_tons(x[col], x['Unit']), axis=1)

    # DATA VALIDATION CHECKS
    # Filter out invalid records
    df = df[df['Production'] > 0]
    df = df[df['Month-end \ninventory'] >= 0]
    df = df[df['Month-end \ninventory'] <= 50000]

    # Analyze by product category - REVISED TO MATCH YOUR PRODUCT LIST
    product_categories = {
        'sugar': ['pure white sugar', 'raw sugar', 'white sugar', 'Molasses', 'sweep/suck'],
        'bakery': ['Cake', 'Cookie', 'cake', 'cookie', 'Toasted bread/Cracker/Biscuit',
                   'Wafer biscuit', 'biscuits/crackers', 'Other baked goods'],
        'canned_foods': ['canned tuna', 'canned sardines', 'canned pineapple',
                        'canned sweet corn', 'canned pickles', 'other canned fruits'],
        'frozen_foods': ['Frozen and chilled chicken meat', 'Frozen and chilled pork',
                        'frozen fish', 'frozen shrimp', 'frozen squid',
                        'frozen fruits and vegetables', 'Frozen prepared food',
                        'bacon', 'ham', 'sausage', 'seasoned chicken meat', 'Ready to cook meals'],
        'condiments': ['Table condiments', 'Monosodium glutamate',
                      'Soy sauce, fermented soybean paste, dark soy sauce',
                      'Soy sauce, fermented soybean paste, light soy sauce', 'coconut milk'],
        'animal_feed': ['Ready made pig feed', 'Ready made chicken feed',
                       'Ready made fish feed', 'Ready made shrimp feed',
                       'Ready made duck feed', 'Ready made pet feed', 'Pet feed',
                       'ready made feed for other livestock', 'Ready-made pig feed'],
        'dairy': ['Yogurt', 'ice cream'],
        'staples': ['Tapioca flour', 'Instant noodles', 'instant noodles',
                   'Premix', 'Soy milk', 'Dried fruits and vegetables',
                   'dried fruits & vegetables', 'minced fish meat'],
        'snacks': ['Other crispy snacks', 'Other crispy snacks (Corn chips, prawn crackers, etc)',
                  'other crispy baked snacks'],
        'other': ['sweep/suck']
    }

    # Categorize products
    df['Category'] = 'other'

    for category, keywords in product_categories.items():
        for keyword in keywords:
            # Use exact matching or contains based on the keyword type
            if '*' in keyword or 'etc' in keyword:
                # Use contains for partial matches
                mask = df['Product'].str.contains(keyword.replace('*', '').split('(')[0].strip(), case=False, na=False)
            else:
                # Use exact match or contains for specific products
                if keyword in ['Cake', 'Cookie', 'cake', 'cookie']:
                    # For these, we need to be careful about case sensitivity
                    mask = df['Product'].str.lower() == keyword.lower()
                else:
                    mask = df['Product'].str.contains(keyword, case=False, na=False)
            df.loc[mask, 'Category'] = category

    # Manual categorization for specific products that need exact matching
    manual_categories = {
        'Other baked goods (pizza, donuts, sandwich bread)': 'bakery',
        'Other crispy snacks (Corn chips, prawn crackers, etc)': 'snacks',
        'Ready to cook meals (others)': 'frozen_foods',
        'Ready-made pig feed': 'animal_feed',
        'Soy sauce, fermented soybean paste, dark soy sauce': 'condiments',
        'Soy sauce, fermented soybean paste, light soy sauce ': 'condiments',
        'dried fruits & vegetables': 'staples',
        'sweep/suck': 'other',
        'Dried fruits and vegetables': 'staples',
        'Frozen and chilled chicken meat': 'frozen_foods',
        'Frozen and chilled pork': 'frozen_foods',
        'Frozen prepared food': 'frozen_foods',
        'Instant noodles': 'staples',
        'Molasses': 'sugar',
        'Monosodium glutamate': 'condiments',
        'Pet feed': 'animal_feed',
        'Premix': 'staples',
        'Ready made chicken feed': 'animal_feed',
        'Ready made duck feed': 'animal_feed',
        'Ready made fish feed': 'animal_feed',
        'Ready made pet feed': 'animal_feed',
        'Ready made pig feed': 'animal_feed',
        'Ready made shrimp feed': 'animal_feed',
        'Soy milk': 'staples',
        'Table condiments': 'condiments',
        'Tapioca flour': 'staples',
        'Toasted bread/Cracker/Biscuit': 'bakery',
        'Wafer biscuit': 'bakery',
        'Yogurt': 'dairy',
        'bacon': 'frozen_foods',
        'biscuits/crackers': 'bakery',
        'canned pickles': 'canned_foods',
        'canned pineapple': 'canned_foods',
        'canned sardines': 'canned_foods',
        'canned sweet corn': 'canned_foods',
        'canned tuna': 'canned_foods',
        'coconut milk': 'condiments',
        'frozen fish': 'frozen_foods',
        'frozen fruits and vegetables': 'frozen_foods',
        'frozen shrimp': 'frozen_foods',
        'frozen squid': 'frozen_foods',
        'ham': 'frozen_foods',
        'ice cream': 'dairy',
        'minced fish meat': 'staples',
        'other baked goods': 'bakery',
        'other canned fruits': 'canned_foods',
        'other crispy baked snacks': 'snacks',
        'pure white sugar': 'sugar',
        'raw sugar': 'sugar',
        'ready made feed for other livestock': 'animal_feed',
        'sausage': 'frozen_foods',
        'seasoned chicken meat': 'frozen_foods',
        'white sugar': 'sugar'
    }

    for product, category in manual_categories.items():
        df.loc[df['Product'] == product, 'Category'] = category

    # Calculate basic metrics
    df = df.sort_values(['Product', 'Year', 'Month'])
    df['MonthsOfInventory'] = df['Month-end \ninventory'] / (df['Total'] / 12).replace(0, np.nan)
    df['AvgInventory'] = (df['Begin month\ninventory'] + df['Month-end \ninventory']) / 2
    df['InventoryTurnover'] = df['Total'] / df['AvgInventory'].replace(0, np.nan)

    # Filter out unrealistic values
    df = df[df['MonthsOfInventory'] < 36]  # No more than 3 years inventory
    df = df[df['InventoryTurnover'] < 24]  # Reasonable turnover

    # ADD THIS AFTER UNIT CONVERSION BUT BEFORE WASTE CALCULATION

    # 1. Remove physically impossible inventory durations
    # Most food products cannot have >12 months inventory
    df = df[df['MonthsOfInventory'] <= 18]  # Reduced from 36 to 18 months

    # 2. Cap extreme production and inventory values
    # Use 99th percentile to identify and cap outliers
    production_cap = df['Production'].quantile(0.99)
    inventory_cap = df['Month-end \ninventory'].quantile(0.99)

    df['Production'] = np.minimum(df['Production'], production_cap)
    df['Month-end \ninventory'] = np.minimum(df['Month-end \ninventory'], inventory_cap)
    df['Begin month\ninventory'] = np.minimum(df['Begin month\ninventory'], inventory_cap)

    # 3. Recalculate metrics after capping
    df['MonthsOfInventory'] = df['Month-end \ninventory'] / (df['Total'] / 12).replace(0, np.nan)
    df['AvgInventory'] = (df['Begin month\ninventory'] + df['Month-end \ninventory']) / 2
    df['InventoryTurnover'] = df['Total'] / df['AvgInventory'].replace(0, np.nan)

    # 4. Add realistic constraints based on product type
    # Different maximum inventory periods for different product types
    max_inventory_months = {
        'dairy': 3,      # Highly perishable
        'bakery': 6,     # Perishable
        'frozen_foods': 12,  # Frozen lasts longer
        'canned_foods': 24,  # Canned goods can last 2 years
        'condiments': 18,    # Preserved foods
        'animal_feed': 12,
        'staples': 15,
        'snacks': 9,
        'sugar': 24,     # Sugar has long shelf life
        'other': 6
    }

    df['MaxInventoryMonths'] = df['Category'].map(max_inventory_months)
    df = df[df['MonthsOfInventory'] <= df['MaxInventoryMonths']]

    # 5. Use industry-standard waste percentages as maximums
    max_waste_percentage = {
        'dairy': 0.08,      # 8% max waste
        'bakery': 0.10,     # 10% max waste
        'frozen_foods': 0.05,
        'canned_foods': 0.04,
        'condiments': 0.03,
        'animal_feed': 0.05,
        'staples': 0.06,
        'snacks': 0.07,
        'sugar': 0.04,
        'other': 0.10
    }

    # IMPROVED WASTE ESTIMATION
    # Use product-specific waste factors based on perishability
    waste_factors = {
        'bakery': 0.12,  # Highly perishable (cakes, cookies, bread)
        'dairy': 0.10,   # Yogurt, ice cream
        'frozen_foods': 0.05,  # Frozen items last longer
        'canned_foods': 0.03,  # Canned goods have long shelf life
        'condiments': 0.04,    # Sauces and condiments
        'animal_feed': 0.06,   # Animal feed
        'staples': 0.04,       # Noodles, flour, etc.
        'snacks': 0.07,        # Snack foods
        'sugar': 0.02,         # Sugar products
        'other': 0.08          # Miscellaneous
    }

    # Apply waste factors
    df['WasteFactor'] = df['Category'].map(waste_factors).fillna(0.06)
    df['MaxWastePct'] = df['Category'].map(max_waste_percentage).fillna(0.08)

    # Calculate waste based on inventory age with realistic constraints
    df['InventoryRisk'] = np.where(
        df['MonthsOfInventory'] > 6,
        np.minimum(df['MonthsOfInventory'] / 12 * df['WasteFactor'] * 1.2, df['MaxWastePct']),
        np.minimum(df['WasteFactor'] * 0.6, df['MaxWastePct'] * 0.7)
    )

    df['PotentialWaste'] = df['Month-end \ninventory'] * df['InventoryRisk']

    # Ensure waste doesn't exceed maximum percentage of production
    max_waste_absolute = df['Production'] * df['MaxWastePct']
    df['PotentialWaste'] = np.minimum(df['PotentialWaste'], max_waste_absolute)

    # 6. Calculate realistic inventory shrinkage (max 2% of production)
    df['ExpectedInventory'] = df['Begin month\ninventory'] + df['Production'] - df['Total']
    df['InventoryShrinkage'] = np.maximum(0, df['ExpectedInventory'] - df['Month-end \ninventory'])
    df['InventoryShrinkage'] = np.minimum(df['InventoryShrinkage'], df['Production'] * 0.02)

    # Create a date column for time series analysis
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')
    
    return df

# Main dashboard
st.title("📊 Food Waste Analysis Dashboard")
st.markdown("Analyzing food waste patterns across different product categories")

# File upload section
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload your food data CSV file", type=["csv"])

if uploaded_file is not None:
    # Process the uploaded file
    df = load_and_process_data(uploaded_file)
    
    if df is not None:
        # Create summary data
        category_analysis = df.groupby('Category').agg({
            'PotentialWaste': 'sum',
            'InventoryShrinkage': 'sum',
            'Production': 'sum',
            'Total': 'sum',
            'MonthsOfInventory': 'mean'
        }).round(2)

        category_analysis['WastePercentage'] = (category_analysis['PotentialWaste'] /
                                               category_analysis['Production'] * 100).round(2)

        # Top waste products
        high_waste_products = df.groupby(['Product', 'Unit']).agg({
            'PotentialWaste': 'sum',
            'Production': 'sum',
            'MonthsOfInventory': 'mean'
        }).nlargest(10, 'PotentialWaste')

        high_waste_products['WastePercentage'] = (high_waste_products['PotentialWaste'] /
                                                 high_waste_products['Production'] * 100).round(2)

        # Sidebar filters
        st.sidebar.header("Filters")
        selected_categories = st.sidebar.multiselect(
            "Select Categories",
            options=df['Category'].unique(),
            default=df['Category'].unique()
        )

        year_range = st.sidebar.slider(
            "Select Year Range",
            min_value=int(df['Year'].min()),
            max_value=int(df['Year'].max()),
            value=(int(df['Year'].min()), int(df['Year'].max()))
        )

        # Filter data based on selections
        filtered_df = df[(df['Category'].isin(selected_categories)) & 
                         (df['Year'] >= year_range[0]) & 
                         (df['Year'] <= year_range[1])]

        # Key metrics
        st.subheader("Key Metrics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_waste = filtered_df['PotentialWaste'].sum()
            st.metric("Total Potential Waste", f"{total_waste:,.0f} tons")

        with col2:
            total_production = filtered_df['Production'].sum()
            waste_percentage = (total_waste / total_production * 100) if total_production > 0 else 0
            st.metric("Waste Percentage", f"{waste_percentage:.2f}%")

        with col3:
            avg_inventory_months = filtered_df['MonthsOfInventory'].mean()
            st.metric("Avg Inventory Months", f"{avg_inventory_months:.1f}")

        with col4:
            total_shrinkage = filtered_df['InventoryShrinkage'].sum()
            st.metric("Inventory Shrinkage", f"{total_shrinkage:,.0f} tons")

        # Charts
        st.subheader("Waste Analysis by Category")

        col1, col2 = st.columns(2)

        with col1:
            # Waste by category (bar chart)
            fig = px.bar(
                category_analysis.reset_index(), 
                x='Category', 
                y='PotentialWaste',
                title="Total Waste by Category",
                labels={'PotentialWaste': 'Waste (tons)', 'Category': 'Product Category'}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Waste percentage by category
            fig = px.bar(
                category_analysis.reset_index(), 
                x='Category', 
                y='WastePercentage',
                title="Waste Percentage by Category",
                labels={'WastePercentage': 'Waste (%)', 'Category': 'Product Category'}
            )
            st.plotly_chart(fig, use_container_width=True)

        # Time series analysis - Matplotlib version
        st.subheader("Food Waste Trends by Category (2000-2025)")
        
        # Create the matplotlib plot
        yearly_waste = df.groupby(['Year', 'Category'])['PotentialWaste'].sum().unstack().fillna(0)

        fig, ax = plt.subplots(figsize=(14, 8))
        yearly_waste.plot(kind='line', marker='o', ax=ax)
        ax.set_title('Food Waste Trends by Category (2000-2025)')
        ax.set_ylabel('Tons of Potential Waste')
        ax.set_xlabel('Year')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Display the plot in Streamlit
        st.pyplot(fig)

        # Inventory turnover analysis
        st.subheader("Inventory Turnover Analysis")
        
        # Create the matplotlib plot
        turnover_by_category = df.groupby('Category')['InventoryTurnover'].mean().sort_values()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        turnover_by_category.plot(kind='barh', ax=ax)
        ax.set_title('Average Inventory Turnover by Product Category')
        ax.set_xlabel('Inventory Turnover Ratio')
        ax.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='Ideal Minimum (1.0)')
        ax.legend()
        plt.tight_layout()
        
        # Display the plot in Streamlit
        st.pyplot(fig)

        # 1. Waste by Category (Pie chart)
        st.subheader("Potential Waste Distribution by Category")
        waste_by_category = category_analysis['PotentialWaste']
        
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        ax1.pie(waste_by_category, labels=waste_by_category.index, autopct='%1.1f%%')
        ax1.set_title('Potential Waste Distribution by Category')
        ax1.axis('equal')
        ax1.grid(False)
        
        st.pyplot(fig1)
        
        # Generate unique key for this button
        key1 = generate_key("pie_chart", waste_by_category.sum())
        buf1 = fig_to_bytes(fig1)
        st.download_button(
            label="Download Pie Chart",
            data=buf1,
            file_name="waste_distribution_pie_chart.png",
            mime="image/png",
            key=key1  # Unique key
        )
        plt.close(fig1)
        
        # 2. Waste Percentage by Category (Bar chart)
        st.subheader("Waste as Percentage of Production")
        sorted_categories = category_analysis.sort_values('WastePercentage', ascending=True)
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.barh(sorted_categories.index, sorted_categories['WastePercentage'])
        ax2.set_title('Waste as Percentage of Production')
        ax2.set_xlabel('Waste Percentage (%)')
        ax2.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        st.pyplot(fig2)
        
        # Generate unique key for this button
        key2 = generate_key("waste_pct", sorted_categories['WastePercentage'].sum())
        buf2 = fig_to_bytes(fig2)
        st.download_button(
            label="Download Waste Percentage Chart",
            data=buf2,
            file_name="waste_percentage_chart.png",
            mime="image/png",
            key=key2  # Unique key
        )
        plt.close(fig2)
        
        # 3. Inventory Months by Category
        st.subheader("Average Months of Inventory by Category")
        inventory_months = category_analysis.sort_values('MonthsOfInventory', ascending=True)
        
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.barh(inventory_months.index, inventory_months['MonthsOfInventory'])
        ax3.set_title('Average Months of Inventory by Category')
        ax3.set_xlabel('Months of Inventory')
        ax3.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        st.pyplot(fig3)
        
        # Generate unique key for this button
        key3 = generate_key("inventory_months", inventory_months['MonthsOfInventory'].sum())
        buf3 = fig_to_bytes(fig3)
        st.download_button(
            label="Download Inventory Months Chart",
            data=buf3,
            file_name="inventory_months_chart.png",
            mime="image/png",
            key=key3  # Unique key
        )
        plt.close(fig3)

        
        # 4. Top 10 Waste Products - NEW
        st.subheader("Top 10 Products by Potential Waste")
        top_products = high_waste_products.head(10)
        
        fig4, ax4 = plt.subplots(figsize=(10, 8))
        ax4.barh(range(len(top_products)), top_products['PotentialWaste'])
        ax4.set_yticks(range(len(top_products)))
        ax4.set_yticklabels([p[:20] + '...' if len(p) > 20 else p for p in top_products.index])
        ax4.set_title('Top 10 Products by Potential Waste')
        ax4.set_xlabel('Tons of Potential Waste')
        ax4.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        st.pyplot(fig4)

        # Generate unique key for this button
        key4 = generate_key("top_products", inventory_months['PotentialWaste'].sum())
        buf4 = fig_to_bytes(fig4)
        st.download_button(
            label="Download top products by waste",
            data=buf4,
            file_name="top_products_by_waste_chart.png",
            mime="image/png",
            key=key4  
        )
        plt.close(fig4)

        # 5. Only plot categories with significant waste - NEW
        st.subheader("Food Waste Trends for High-Waste Categories")
        significant_categories = yearly_waste.columns[yearly_waste.sum() > 10000]
        
        fig5, ax5 = plt.subplots(figsize=(14, 8))
        yearly_waste[significant_categories].plot(kind='line', marker='o', linewidth=2, ax=ax5)
        ax5.set_title('Food Waste Trends for High-Waste Categories (2000-2025)')
        ax5.set_ylabel('Tons of Potential Waste')
        ax5.set_xlabel('Year')
        ax5.grid(True, alpha=0.3)
        ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        st.pyplot(fig5)
        
        # Add download button with unique key
        buf5 = fig_to_bytes(fig5)
        st.download_button(
            label="Download High-Waste Trends Chart",
            data=buf5,
            file_name="high_waste_trends_chart.png",
            mime="image/png",
            key="high_waste_trends_download_005"  # Unique key for this button
        )
        plt.close(fig5)
        
        # # 7. Annual waste summary - NEW
        # st.subheader("Total Annual Potential Food Waste")
        # annual_waste = df.groupby('Year')['PotentialWaste'].sum()
        
        # fig, ax = plt.subplots(figsize=(12, 6))
        # ax.plot(annual_waste.index, annual_waste.values, marker='o', linewidth=2)
        # ax.set_title('Total Annual Potential Food Waste')
        # ax.set_ylabel('Tons of Waste')
        # ax.set_xlabel('Year')
        # ax.grid(True, alpha=0.3)
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        
        # st.pyplot(fig)

        
        # 6. Annual waste summary
        st.subheader("Total Annual Potential Food Waste")
        annual_waste = df.groupby('Year')['PotentialWaste'].sum()
        
        fig6, ax6 = plt.subplots(figsize=(12, 6))
        ax6.plot(annual_waste.index, annual_waste.values, marker='o', linewidth=2)
        ax6.set_title('Total Annual Potential Food Waste')
        ax6.set_ylabel('Tons of Waste')
        ax6.set_xlabel('Year')
        ax6.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig6)
        buf6 = fig_to_bytes(fig6)
        st.download_button(
            label="Download Annual Waste Chart",
            data=buf6,
            file_name="annual_waste_chart.png",
            mime="image/png",
            key="annual_waste_download_006"
        )
        plt.close(fig6)
        
        # 7. Time series analysis (if not already included)
        st.subheader("Food Waste Trends by Category")
        yearly_waste = df.groupby(['Year', 'Category'])['PotentialWaste'].sum().unstack().fillna(0)
        
        fig7, ax7 = plt.subplots(figsize=(14, 8))
        yearly_waste.plot(kind='line', marker='o', ax=ax7)
        ax7.set_title('Food Waste Trends by Category (2000-2025)')
        ax7.set_ylabel('Tons of Potential Waste')
        ax7.set_xlabel('Year')
        ax7.grid(True, alpha=0.3)
        ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        st.pyplot(fig7)
        buf7 = fig_to_bytes(fig7)
        st.download_button(
            label="Download All Categories Trends Chart",
            data=buf7,
            file_name="all_categories_trends_chart.png",
            mime="image/png",
            key="all_categories_trends_download_007"
        )
        plt.close(fig7

        
        # # Inventory analysis section
        # st.subheader("Inventory Analysis")
        
        # col1, col2 = st.columns(2)
        
        # with col1:
        #     # Inventory turnover by category (using Plotly)
        #     turnover_by_category = filtered_df.groupby('Category')['InventoryTurnover'].mean().sort_values()
            
        #     fig = go.Figure()
        #     fig.add_trace(go.Bar(
        #         y=turnover_by_category.index,
        #         x=turnover_by_category.values,
        #         orientation='h',
        #         marker_color='lightblue'
        #     ))
        #     fig.add_vline(x=1, line_dash="dash", line_color="red", 
        #                   annotation_text="Ideal Minimum (1.0)", 
        #                   annotation_position="top right")
        #     fig.update_layout(
        #         title='Average Inventory Turnover by Category',
        #         xaxis_title='Turnover Ratio',
        #         yaxis_title='Category',
        #         height=400
        #     )
        #     st.plotly_chart(fig, use_container_width=True)
        
        # with col2:
        #     # Months of inventory by category (existing code)
        #     months_data = filtered_df.groupby('Category')['MonthsOfInventory'].mean().reset_index()
        #     fig = px.bar(
        #         months_data, 
        #         x='Category', 
        #         y='MonthsOfInventory',
        #         title="Average Months of Inventory by Category",
        #         labels={'MonthsOfInventory': 'Months', 'Category': 'Product Category'}
        #     )
        #     st.plotly_chart(fig, use_container_width=True)
        
        # Time series analysis - Plotly version
        st.subheader("Trend Analysis Over Time")

        # Aggregate data by date and category
        time_series_data = filtered_df.groupby(['Date', 'Category']).agg({
            'PotentialWaste': 'sum',
            'Production': 'sum',
            'Month-end \ninventory': 'mean'
        }).reset_index()

        fig = px.line(
            time_series_data, 
            x='Date', 
            y='PotentialWaste', 
            color='Category',
            title="Waste Trends Over Time",
            labels={'PotentialWaste': 'Waste (tons)', 'Date': 'Time'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Top products with highest waste
        st.subheader("Top 10 Products with Highest Waste")

        fig = px.bar(
            high_waste_products.reset_index(), 
            x='Product', 
            y='PotentialWaste',
            title="Top 10 High Waste Products",
            labels={'PotentialWaste': 'Waste (tons)', 'Product': 'Product Name'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Inventory analysis
        st.subheader("Inventory Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Inventory turnover by category
            turnover_data = filtered_df.groupby('Category')['InventoryTurnover'].mean().reset_index()
            fig = px.bar(
                turnover_data, 
                x='Category', 
                y='InventoryTurnover',
                title="Average Inventory Turnover by Category",
                labels={'InventoryTurnover': 'Turnover Ratio', 'Category': 'Product Category'}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Months of inventory by category
            months_data = filtered_df.groupby('Category')['MonthsOfInventory'].mean().reset_index()
            fig = px.bar(
                months_data, 
                x='Category', 
                y='MonthsOfInventory',
                title="Average Months of Inventory by Category",
                labels={'MonthsOfInventory': 'Months', 'Category': 'Product Category'}
            )
            st.plotly_chart(fig, use_container_width=True)

        # Detailed data view
        st.subheader("Detailed Data")
        if st.checkbox("Show raw data"):
            st.dataframe(filtered_df)

        # Download button
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(filtered_df)

        st.download_button(
            label="Download filtered data as CSV",
            data=csv,
            file_name="filtered_food_waste_data.csv",
            mime="text/csv",
        )

    else:
        st.error("Error processing the uploaded file. Please check the file format.")
else:
    st.info("Please upload a CSV file to begin analysis.")

# Footer
st.markdown("---")
st.markdown("**Note:** All values have been converted to tons for consistent analysis.")



