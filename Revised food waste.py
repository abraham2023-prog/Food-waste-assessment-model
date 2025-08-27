import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Food Waste Analysis Dashboard",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #2e8b57; font-weight: 700;}
    .section-header {font-size: 1.8rem; color: #2e8b57; border-bottom: 2px solid #2e8b57; padding-bottom: 0.3rem;}
    .metric-label {font-weight: 600; color: #2e8b57;}
    .positive-metric {color: #228B22;}
    .negative-metric {color: #DC143C;}
    .info-text {background-color: #f0f8f0; padding: 15px; border-radius: 5px; border-left: 4px solid #2e8b57;}
</style>
""", unsafe_allow_html=True)

# Check if statsmodels is available for trendlines
try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# App title
st.markdown('<p class="main-header">🍎 Food Waste Analysis Dashboard</p>', unsafe_allow_html=True)
st.markdown("Analyze food waste patterns from production and inventory data")

# Function to normalize column names
def normalize_column_names(df):
    """Normalize column names to handle different naming conventions"""
    # Create a mapping of possible column name variations
    column_mapping = {
        'tsic': 'TSIC',
        'code': 'Code',
        'product': 'Product',
        'unit': 'Unit',
        'year': 'Year',
        'month': 'Month',
        'begin month inventory': 'Begin month inventory',
        'begin_month_inventory': 'Begin month inventory',
        'beginmonthinventory': 'Begin month inventory',
        'production': 'Production',
        'domestic': 'Domestic',
        'export': 'Export',
        'total': 'Total',
        'shipment value (thousand baht)': 'Shipment value (thousand baht)',
        'shipment_value_thousand_baht': 'Shipment value (thousand baht)',
        'month-end inventory': 'Month-end inventory',
        'month_end_inventory': 'Month-end inventory',
        'monthendinventory': 'Month-end inventory',
        'capacity': 'Capacity'
    }
    
    # Normalize the column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
    
    # Map to standardized names
    new_columns = []
    for col in df.columns:
        normalized = col.lower().replace(' ', '_').replace('-', '_')
        new_columns.append(column_mapping.get(normalized, col))
    
    df.columns = new_columns
    return df

# Generate sample data if not uploaded
def generate_sample_data():
    # Create sample data with realistic values
    np.random.seed(42)
    n_records = 1000
    
    # Raw product list from your dataset
    raw_products = [
        'Ready made pig feed', 'Ready made chicken feed', 'Ready made fish feed',
        'Ready made shrimp feed', 'Premix', 'Dried fruits and vegetables',
        'Ice cream', 'Yogurt', 'Soy milk', 'Tapioca flour', 'Cake',
        'Other baked goods (pizza donuts sandwich bread)', 'Wafer biscuit',
        'Cookie', 'Toasted bread/Cracker/Biscuit',
        'Other crispy snacks (Corn chips prawn crackers etc)', 'Molasses',
        'Instant noodles', 'Table condiments',
        'Soy sauce fermented soybean paste light soy sauce ',
        'Monosodium glutamate', 'Ready to cook meals (others)', 'Pet feed',
        'Frozen and chilled pork', 'Frozen and chilled chicken meat',
        'ham', 'bacon', 'sausage', 'seasoned chicken meat', 'frozen fish',
        'minced fish meat', 'frozen shrimp', 'frozen squid', 'canned tuna',
        'canned sardines', 'frozen fruits and vegetables', 'canned pineapple',
        'other canned fruits', 'canned sweet corn', 'canned pickles',
        'dried fruits & vegetables', 'coconut milk', 'ice cream',
        'tapioca starch', 'cake', 'other baked goods', 'cookie',
        'biscuits/crackers', 'other crispy baked snacks', 'raw sugar',
        'white sugar', 'pure white sugar', 'molasses', 'sweep/suck',
        'instant noodles', 'Frozen prepared food',
        'small condiments or seasoning dispensers',
        'Soy sauce fermented soybean paste dark soy sauce',
        'Ready made pet feed', 'Ready-made pig feed', 'Ready made duck feed',
        'ready made feed for other livestock'
    ]
    
    # Create sample data
    data = {
        'Product': np.random.choice(raw_products, n_records),
        'Year': np.random.randint(2020, 2024, n_records),
        'Month': np.random.randint(1, 13, n_records),
        'Begin month inventory': np.random.uniform(100, 10000, n_records),
        'Production': np.random.uniform(500, 50000, n_records),
        'Domestic': np.random.uniform(400, 40000, n_records),
        'Export': np.random.uniform(50, 5000, n_records),
        'Month-end inventory': np.random.uniform(80, 8000, n_records),
        'Capacity': np.random.uniform(1000, 100000, n_records),
        'Shipment value (thousand baht)': np.random.uniform(1000, 100000, n_records)
    }
    
    df = pd.DataFrame(data)
    df['Total'] = df['Domestic'] + df['Export']
    
    return df

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

# Product categories for analysis
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

# Waste factors by category
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

# Maximum waste percentages by category
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

# Maximum inventory months by category
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

# Function to convert units to tons
def convert_to_tons(value, unit):
    if unit == 'liter':
        # For most food products: 1 liter ≈ 1 kg = 0.001 tons
        return value * 0.001
    elif unit == 'thousand_liter':
        # 1000 liters ≈ 1 ton (for most liquids)
        return value * 1.0
    else:  # ton - no conversion needed
        return value

# Load and clean data
with st.sidebar:
    st.header("Data Configuration")
    st.subheader("Upload Data")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Initialize session state
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'column_mapping' not in st.session_state:
    st.session_state.column_mapping = {}
if 'mapping_complete' not in st.session_state:
    st.session_state.mapping_complete = False

# Process uploaded file
if uploaded_file is not None:
    if st.session_state.df_processed is None:
        df = pd.read_csv(uploaded_file)
        
        # Store original column names for reference
        original_columns = df.columns.tolist()
        st.session_state.original_columns = original_columns
        
        # Simple normalization (convert to lowercase and replace spaces with underscores)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
        st.session_state.df_raw = df
        
        # Check for required columns
        required_cols = ['begin_month_inventory', 'production', 'domestic', 'export', 'month_end_inventory']
        available_cols = df.columns.tolist()
        
        missing_cols = [c for c in required_cols if c not in available_cols]
        
        if missing_cols:
            st.session_state.missing_cols = missing_cols
            st.session_state.mapping_needed = True
        else:
            st.session_state.mapping_needed = False
            st.session_state.df_processed = df
            st.session_state.mapping_complete = True

# Show column mapping interface if needed
if uploaded_file is not None and st.session_state.get('mapping_needed', False):
    st.warning("Some required columns are missing from your dataset.")
    st.info("Please help us map your dataset columns to the required columns:")
    
    st.write("**Your dataset columns:**")
    for col in st.session_state.original_columns:
        st.write(f"- '{col}'")
    
    st.write("---")
    
    # Create mapping interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**Required Column**")
        st.write("Begin month inventory")
        st.write("Month-end inventory")
        st.write("Production")
        st.write("Domestic")
        st.write("Export")
    
    with col2:
        st.write("**Map to Your Column**")
        
        # Get available columns for mapping
        available_options = [""] + st.session_state.df_raw.columns.tolist()
        
        # Create select boxes for each required column
        begin_map = st.selectbox(
            "Select column for Begin month inventory",
            options=available_options,
            key="begin_map",
            label_visibility="collapsed"
        )
        
        end_map = st.selectbox(
            "Select column for Month-end inventory",
            options=available_options,
            key="end_map",
            label_visibility="collapsed"
        )
        
        production_map = st.selectbox(
            "Select column for Production",
            options=available_options,
            key="production_map",
            label_visibility="collapsed"
        )
        
        domestic_map = st.selectbox(
            "Select column for Domestic",
            options=available_options,
            key="domestic_map",
            label_visibility="collapsed"
        )
        
        export_map = st.selectbox(
            "Select column for Export",
            options=available_options,
            key="export_map",
            label_visibility="collapsed"
        )
    
    if st.button("Apply Mapping"):
        # Store the mapping
        mapping = {
            'begin_month_inventory': begin_map,
            'month_end_inventory': end_map,
            'production': production_map,
            'domestic': domestic_map,
            'export': export_map
        }
        
        # Apply the mapping
        df_processed = st.session_state.df_raw.copy()
        
        for required_col, dataset_col in mapping.items():
            if dataset_col and dataset_col in df_processed.columns:
                df_processed[required_col] = df_processed[dataset_col]
        
        # Check if we have all required columns now
        missing_after_mapping = [c for c in ['begin_month_inventory', 'month_end_inventory', 'production', 'domestic', 'export'] 
                               if c not in df_processed.columns]
        
        if missing_after_mapping:
            st.error(f"Still missing columns after mapping: {missing_after_mapping}")
        else:
            st.session_state.df_processed = df_processed
            st.session_state.mapping_complete = True
            st.session_state.column_mapping = mapping
            st.success("Column mapping applied successfully!")
            st.rerun()

# Use sample data if no file uploaded or mapping not complete
if uploaded_file is None or not st.session_state.get('mapping_complete', False):
    if uploaded_file is None:
        st.info("Using sample data for demonstration")
    else:
        st.info("Please complete the column mapping above to proceed with your data")
    
    df = generate_sample_data()
else:
    df = st.session_state.df_processed
    st.success("Using your uploaded data with applied column mapping")

# Now process the numeric columns
numeric_cols = ['begin_month_inventory', 'production', 'domestic', 'export', 
                'month_end_inventory', 'capacity', 'shipment_value_thousand_baht', 'total']

for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)

# Apply unit mapping and convert to tons
if 'product' in df.columns:
    df['Unit'] = df['Product'].map(unit_mapping)
    df['Unit'] = df['Unit'].fillna('ton')  # Default to tons if no mapping found
    
    # Convert all numeric columns to tons
    for col in ['begin_month_inventory', 'production', 'domestic', 'export', 'total', 'month_end_inventory']:
        if col in df.columns:
            df[col] = df.apply(lambda x: convert_to_tons(x[col], x['Unit']), axis=1)

# Categorize products
if 'product' in df.columns:
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

if 'product' in df.columns and 'Category' in df.columns:
    for product, category in manual_categories.items():
        df.loc[df['Product'] == product, 'Category'] = category

# Calculate realistic waste metrics
if all(col in df.columns for col in ['begin_month_inventory', 'production', 'domestic', 'export', 'month_end_inventory']):
    # Calculate months of inventory
    df['MonthsOfInventory'] = df['month_end_inventory'] / (df['total'] / 12).replace(0, np.nan)
    df['AvgInventory'] = (df['begin_month_inventory'] + df['month_end_inventory']) / 2
    df['InventoryTurnover'] = df['total'] / df['AvgInventory'].replace(0, np.nan)
    
    # Filter out unrealistic values
    df = df[df['MonthsOfInventory'] < 36]  # No more than 3 years inventory
    df = df[df['InventoryTurnover'] < 24]  # Reasonable turnover
    
    # Cap extreme production and inventory values
    if 'production' in df.columns:
        production_cap = df['production'].quantile(0.99)
        df['production'] = np.minimum(df['production'], production_cap)
    
    if 'month_end_inventory' in df.columns:
        inventory_cap = df['month_end_inventory'].quantile(0.99)
        df['month_end_inventory'] = np.minimum(df['month_end_inventory'], inventory_cap)
        df['begin_month_inventory'] = np.minimum(df['begin_month_inventory'], inventory_cap)
    
    # Recalculate metrics after capping
    df['MonthsOfInventory'] = df['month_end_inventory'] / (df['total'] / 12).replace(0, np.nan)
    df['AvgInventory'] = (df['begin_month_inventory'] + df['month_end_inventory']) / 2
    df['InventoryTurnover'] = df['total'] / df['AvgInventory'].replace(0, np.nan)
    
    # Apply category-specific constraints
    if 'Category' in df.columns:
        df['MaxInventoryMonths'] = df['Category'].map(max_inventory_months)
        df = df[df['MonthsOfInventory'] <= df['MaxInventoryMonths']]
        
        # Calculate realistic waste based on inventory age and category
        df['WasteFactor'] = df['Category'].map(waste_factors).fillna(0.06)
        df['MaxWastePct'] = df['Category'].map(max_waste_percentage).fillna(0.08)
        
        # Calculate waste based on inventory age with realistic constraints
        df['InventoryRisk'] = np.where(
            df['MonthsOfInventory'] > 6,
            np.minimum(df['MonthsOfInventory'] / 12 * df['WasteFactor'] * 1.2, df['MaxWastePct']),
            np.minimum(df['WasteFactor'] * 0.6, df['MaxWastePct'] * 0.7)
        )
        
        df['waste'] = df['month_end_inventory'] * df['InventoryRisk']
        
        # Ensure waste doesn't exceed maximum percentage of production
        max_waste_absolute = df['production'] * df['MaxWastePct']
        df['waste'] = np.minimum(df['waste'], max_waste_absolute)
        
        # Calculate waste rate
        df['waste_rate'] = df['waste'] / df['production'].replace(0, 0.001)
        
        # Calculate inventory shrinkage (max 2% of production)
        df['ExpectedInventory'] = df['begin_month_inventory'] + df['production'] - df['total']
        df['InventoryShrinkage'] = np.maximum(0, df['ExpectedInventory'] - df['month_end_inventory'])
        df['InventoryShrinkage'] = np.minimum(df['InventoryShrinkage'], df['production'] * 0.02)
        
        # Calculate waste value if shipment value is available
        if 'shipment_value_thousand_baht' in df.columns:
            df['value_per_unit'] = df['shipment_value_thousand_baht'] / df['total'].replace(0, 0.001)
            df['waste_value'] = df['waste'] * df['value_per_unit']
        else:
            df['value_per_unit'] = 0
            df['waste_value'] = 0
    else:
        # Fallback calculation if category is not available
        df['waste'] = (df['begin_month_inventory'] + df['production']) - (df['domestic'] + df['export'] + df['month_end_inventory'])
        df['waste'] = np.maximum(0, df['waste'])  # Ensure waste is not negative
        df['waste_rate'] = df['waste'] / df['production'].replace(0, 0.001)
        df['waste_value'] = 0
else:
    st.warning("Cannot calculate waste metrics - missing required inventory columns")
    df['waste'] = 0
    df['waste_rate'] = 0
    df['AvgInventory'] = 0
    df['InventoryTurnover'] = 0
    df['waste_value'] = 0

# Calculate other metrics
if 'capacity' in df.columns:
    df['capacity_utilization'] = df['production'] / df['capacity'].replace(0, 0.001)
else:
    df['capacity_utilization'] = 0

# Analysis parameters in sidebar
with st.sidebar:
    st.subheader("Analysis Parameters")
    
    # Get unique products, handling potential missing 'Product' column
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

    # Get categories if available
    if 'Category' in df.columns:
        category_options = df['Category'].unique()
        selected_categories = st.multiselect(
            "Select Categories to Analyze",
            options=category_options,
            default=category_options
        )
    else:
        selected_categories = []

    # Get year range, handling potential missing 'year' column
    if 'year' in df.columns:
        min_year = int(df['year'].min())
        max_year = int(df['year'].max())
    else:
        min_year = 2020
        max_year = 2023
        df['year'] = 2022  # Default year
    
    year_range = st.slider(
        "Select Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )

    st.subheader("Waste Calculation")
    st.info("Realistic waste estimation based on inventory age and product category")

# Filter data based on selections
df_filtered = df[
    (df['product'].isin(selected_products)) &
    (df['year'] >= year_range[0]) &
    (df['year'] <= year_range[1])
].copy()

if selected_categories and 'Category' in df.columns:
    df_filtered = df_filtered[df_filtered['Category'].isin(selected_categories)]

# Create tabs for different analyses
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Waste Analysis",
    "Inventory Analysis",
    "Production vs Demand",
    "Economic Impact",
    "Category Analysis"
])

with tab1:
    st.markdown('<p class="section-header">📊 Overview Metrics</p>', unsafe_allow_html=True)

    # Calculate overall metrics
    total_waste = df_filtered['waste'].sum()
    total_production = df_filtered['production'].sum()
    overall_waste_rate = total_waste / total_production if total_production > 0 else 0
    total_waste_value = df_filtered['waste_value'].sum()
    avg_turnover = df_filtered['InventoryTurnover'].mean() if 'InventoryTurnover' in df_filtered.columns else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Waste", f"{total_waste:,.0f} tons")
    with col2:
        st.metric("Overall Waste Rate", f"{overall_waste_rate:.2%}")
    with col3:
        st.metric("Value of Waste", f"฿{total_waste_value:,.0f}")
    with col4:
        st.metric("Avg Inventory Turnover", f"{avg_turnover:.2f}")

    # Time series of waste
    st.markdown("#### Waste Trends Over Time")
    waste_by_time = df_filtered.groupby(['year', 'month']).agg({'waste': 'sum'}).reset_index()
    waste_by_time['date'] = pd.to_datetime(waste_by_time['year'].astype(str) + '-' + waste_by_time['month'].astype(str))

    fig = px.line(waste_by_time, x='date', y='waste',
                  title="Total Waste Over Time")
    st.plotly_chart(fig, use_container_width=True)

    # Waste by product
    st.markdown("#### Waste by Product")
    waste_by_product = df_filtered.groupby('product').agg({
        'waste': 'sum',
        'production': 'sum',
        'waste_value': 'sum'
    }).reset_index()
    waste_by_product['waste_rate'] = waste_by_product['waste'] / waste_by_product['production']

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(waste_by_product, x='product', y='waste',
                     title="Total Waste by Product")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(waste_by_product, x='product', y='waste_rate',
                     title="Waste Rate by Product")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown('<p class="section-header">📈 Waste Analysis</p>', unsafe_allow_html=True)

    # Seasonal waste patterns
    st.markdown("#### Seasonal Waste Patterns")
    waste_by_month = df_filtered.groupby(['month', 'product']).agg({'waste': 'mean'}).reset_index()

    fig = px.line(waste_by_month, x='month', y='waste', color='product',
                  title="Average Waste by Month")
    st.plotly_chart(fig, use_container_width=True)

    # Waste distribution
    st.markdown("#### Waste Distribution by Product")
    fig = px.box(df_filtered, x='product', y='waste',
                 title="Distribution of Waste Amounts by Product")
    st.plotly_chart(fig, use_container_width=True)

    # Yearly comparison
    st.markdown("#### Yearly Waste Comparison")
    waste_by_year = df_filtered.groupby(['year', 'product']).agg({'waste': 'sum'}).reset_index()

    fig = px.bar(waste_by_year, x='year', y='waste', color='product',
                 barmode='group', title="Total Waste by Year and Product")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown('<p class="section-header">📦 Inventory Analysis</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Inventory Turnover by Product")
        turnover_by_product = df_filtered.groupby('product').agg({
            'InventoryTurnover': 'mean'
        }).reset_index().sort_values('InventoryTurnover', ascending=False)

        fig = px.bar(turnover_by_product, x='product', y='InventoryTurnover',
                     title="Average Inventory Turnover Ratio")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Months of Inventory Supply")
        months_supply_by_product = df_filtered.groupby('product').agg({
            'MonthsOfInventory': 'mean'
        }).reset_index().sort_values('MonthsOfInventory', ascending=False)

        fig = px.bar(months_supply_by_product, x='product', y='MonthsOfInventory',
                     title="Average Months of Inventory Supply")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("##### Inventory vs Waste Relationship")
        # Only add trendline if statsmodels is available
        if HAS_STATSMODELS:
            fig = px.scatter(df_filtered, x='AvgInventory', y='waste', color='product',
                             trend

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

