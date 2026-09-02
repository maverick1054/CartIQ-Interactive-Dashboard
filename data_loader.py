import os
import pandas as pd
import numpy as np

DATASET_DIR = os.path.join(os.path.dirname(__file__), 'Dataset', 'CartIQ_Dataset')

def get_raw_data():
    """Load raw CSV files from dataset directory."""
    customers = pd.read_csv(os.path.join(DATASET_DIR, 'customers.csv'))
    orders = pd.read_csv(os.path.join(DATASET_DIR, 'orders.csv'))
    products = pd.read_csv(os.path.join(DATASET_DIR, 'products.csv'))
    sales = pd.read_csv(os.path.join(DATASET_DIR, 'sales.csv'))
    return customers, orders, products, sales

def get_cleaned_merged_data():
    """Clean and merge all datasets into a single master transaction DataFrame."""
    customers, orders, products, sales = get_raw_data()
    
    # Clean product column names
    products = products.rename(columns={'product_ID': 'product_id'})
    
    # Process dates
    orders['order_date'] = pd.to_datetime(orders['order_date'], errors='coerce')
    orders['delivery_date'] = pd.to_datetime(orders['delivery_date'], errors='coerce')
    
    # Calculate delivery duration in days
    orders['delivery_days'] = (orders['delivery_date'] - orders['order_date']).dt.days
    
    # Remove negative or invalid delivery days if any
    orders['delivery_days'] = orders['delivery_days'].where(orders['delivery_days'] >= 0, np.nan)
    med_days = orders['delivery_days'].median()
    if pd.isna(med_days):
        med_days = 14.0
    orders['delivery_days'] = orders['delivery_days'].fillna(med_days)
    
    # Create binary target: is_delayed (if delivery takes more than median delivery days)
    orders['is_delayed'] = (orders['delivery_days'] > med_days).astype(int)
    
    # Extract order time features
    orders['order_month'] = orders['order_date'].dt.month
    orders['order_year'] = orders['order_date'].dt.year
    orders['order_dayofweek'] = orders['order_date'].dt.dayofweek
    
    # Merge sales + orders
    sales_orders = pd.merge(sales, orders, on='order_id', how='left')
    
    # Merge + customers
    sales_orders_cust = pd.merge(sales_orders, customers, on='customer_id', how='left')
    
    # Merge + products
    master_df = pd.merge(sales_orders_cust, products, on='product_id', how='left', suffixes=('_sales', '_product'))
    
    # Clean up column names and handle missing values
    if 'price_per_unit' in master_df.columns:
        master_df['price_per_unit'] = master_df['price_per_unit'].fillna(master_df['price'])
    
    # Age groups
    bins = [0, 25, 40, 60, 100]
    labels = ['<25', '25-40', '41-60', '60+']
    master_df['age_group'] = pd.cut(master_df['age'], bins=bins, labels=labels)
    
    return master_df

def get_customer_summary():
    """Aggregate dataset to Customer level for Spend & Behavior analysis."""
    master_df = get_cleaned_merged_data()
    
    cust_agg = master_df.groupby('customer_id').agg(
        total_spend=('total_price', 'sum'),
        total_orders=('order_id', 'nunique'),
        total_items=('quantity_sales', 'sum'),
        avg_delivery_days=('delivery_days', 'mean'),
        age=('age', 'first'),
        gender=('gender', 'first'),
        state=('state', 'first'),
        city=('city', 'first'),
        zip_code=('zip_code', 'first')
    ).reset_index()
    
    cust_agg['avg_order_value'] = np.where(cust_agg['total_orders'] > 0, cust_agg['total_spend'] / cust_agg['total_orders'], 0)
    
    # Create binary high_value customer feature (spend > 75th percentile)
    spend_75 = cust_agg['total_spend'].quantile(0.75)
    cust_agg['is_high_value'] = (cust_agg['total_spend'] >= spend_75).astype(int)
    
    bins = [0, 25, 40, 60, 100]
    labels = ['<25', '25-40', '41-60', '60+']
    cust_agg['age_group'] = pd.cut(cust_agg['age'], bins=bins, labels=labels)
    
    return cust_agg

def get_order_summary():
    """Aggregate dataset to Order level for Delivery Delay & Order value analysis."""
    master_df = get_cleaned_merged_data()
    
    order_agg = master_df.groupby('order_id').agg(
        customer_id=('customer_id', 'first'),
        order_date=('order_date', 'first'),
        delivery_date=('delivery_date', 'first'),
        delivery_days=('delivery_days', 'first'),
        is_delayed=('is_delayed', 'first'),
        order_month=('order_month', 'first'),
        order_dayofweek=('order_dayofweek', 'first'),
        payment=('payment', 'first'),
        total_order_price=('total_price', 'sum'),
        item_count=('quantity_sales', 'sum'),
        age=('age', 'first'),
        gender=('gender', 'first'),
        state=('state', 'first')
    ).reset_index()
    
    return order_agg
