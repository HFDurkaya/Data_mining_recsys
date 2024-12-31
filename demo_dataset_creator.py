import os
import pandas as pd
from glob import glob
import shutil
from pathlib import Path

# Create demo_dataset directory if it doesn't exist
os.makedirs('demo_dataset', exist_ok=True)

# Load the original datasets
print("Loading datasets...")
customers = pd.read_csv("data/customers.csv", dtype={'customer_id': str})
articles = pd.read_csv("data/articles.csv", dtype={'article_id': str})
transactions = pd.read_csv("data/transactions_train.csv", dtype={'article_id': str})

# Process customers data
print("Processing customer data...")
# Count purchases per customer
purchase_counts = transactions['customer_id'].value_counts()
# Filter customers with 20-40 purchases
valid_customers = purchase_counts[(purchase_counts >= 20) & (purchase_counts <= 40)].index

# Take only the first 100 valid customers
valid_customers = valid_customers[:100]

# Filter customers DataFrame
filtered_customers = customers[customers['customer_id'].isin(valid_customers)].copy()

# Fill missing values and create age intervals
filtered_customers["age"] = filtered_customers["age"].fillna(filtered_customers["age"].median())

def create_age_interval(x):
    if x <= 25:
        return 0
    elif x <= 35:
        return 1
    elif x <= 45:
        return 2
    elif x <= 55:
        return 3
    elif x <= 65:
        return 4
    else:
        return 5

filtered_customers["age_interval"] = filtered_customers["age"].apply(create_age_interval)

# Filter transactions
print("Filtering transactions...")
filtered_transactions = transactions[transactions['customer_id'].isin(valid_customers)].copy()

# Get relevant article IDs
valid_articles = filtered_transactions['article_id'].unique()

# Filter articles
print("Filtering articles...")
filtered_articles = articles[articles['article_id'].isin(valid_articles)].copy()

# Filter and copy images
print("Copying relevant images...")
demo_images_dir = Path('demo_dataset/images')
demo_images_dir.mkdir(parents=True, exist_ok=True)

# Create a set of valid article IDs for faster lookup
valid_articles_set = set(valid_articles)

# Get all image files using pathlib for better path handling
source_images_dir = Path('data/images')
copied_count = 0

# Walk through the directory structure
for img_path in source_images_dir.rglob('*.jpg'):
    img_id = img_path.stem  # Get filename without extension
    if img_id in valid_articles_set:
        # Create corresponding subdirectory in demo_dataset
        relative_path = img_path.relative_to(source_images_dir)
        target_path = demo_images_dir / relative_path.parent
        target_path.mkdir(parents=True, exist_ok=True)
        
        # Copy the image
        shutil.copy2(img_path, target_path / img_path.name)
        copied_count += 1
        
        if copied_count % 100 == 0:
            print(f"Copied {copied_count} images...")

# Optimize data types
print("Optimizing data types...")
# Customers
filtered_customers['customer_id'] = filtered_customers['customer_id'].astype('string')
filtered_customers['age'] = filtered_customers['age'].astype('int8')
filtered_customers['age_interval'] = filtered_customers['age_interval'].astype('int8')

# Articles
filtered_articles['article_id'] = filtered_articles['article_id'].astype('string')
filtered_articles['product_type_no'] = filtered_articles['product_type_no'].astype('int16')
filtered_articles['product_type_name'] = filtered_articles['product_type_name'].astype('category')
filtered_articles['product_group_name'] = filtered_articles['product_group_name'].astype('category')
filtered_articles['garment_group_no'] = filtered_articles['garment_group_no'].astype('int16')
filtered_articles['garment_group_name'] = filtered_articles['garment_group_name'].astype('category')
filtered_articles['colour_group_code'] = filtered_articles['colour_group_code'].astype('int16')
filtered_articles['colour_group_name'] = filtered_articles['colour_group_name'].astype('category')
filtered_articles['section_no'] = filtered_articles['section_no'].astype('int16')
filtered_articles['section_name'] = filtered_articles['section_name'].astype('category')
filtered_articles['perceived_colour_value_id'] = filtered_articles['perceived_colour_value_id'].astype('int16')
filtered_articles['perceived_colour_value_name'] = filtered_articles['perceived_colour_value_name'].astype('category')
filtered_articles['perceived_colour_master_id'] = filtered_articles['perceived_colour_master_id'].astype('int16')
filtered_articles['perceived_colour_master_name'] = filtered_articles['perceived_colour_master_name'].astype('category')
filtered_articles['detail_desc'] = filtered_articles['detail_desc'].fillna("").astype('string')

# Transactions
filtered_transactions['t_dat'] = pd.to_datetime(filtered_transactions['t_dat'])
filtered_transactions['customer_id'] = filtered_transactions['customer_id'].astype('string')
filtered_transactions['article_id'] = filtered_transactions['article_id'].astype('string')
filtered_transactions['sales_channel_id'] = filtered_transactions['sales_channel_id'].astype('int8')
filtered_transactions['price'] = filtered_transactions['price'].astype('float32')

# Save filtered datasets as pickles
print("Saving filtered datasets...")
filtered_customers.to_pickle("demo_dataset/customers.pkl")
filtered_articles.to_pickle("demo_dataset/articles.pkl")
filtered_transactions.to_pickle("demo_dataset/transactions.pkl")

# Print summary
print("\nDataset Summary:")
print(f"Number of customers: {len(filtered_customers)}")
print(f"Number of articles: {len(filtered_articles)}")
print(f"Number of transactions: {len(filtered_transactions)}")
print(f"Number of images copied: {copied_count}")

# Print first few customer IDs to verify we got the first ones
print("\nFirst few customer IDs in the dataset:")
print(filtered_customers['customer_id'].head().tolist())

# Additional verification statistics
print("\nPurchase count verification:")
final_purchase_counts = filtered_transactions['customer_id'].value_counts()
print(f"Min purchases per customer: {final_purchase_counts.min()}")
print(f"Max purchases per customer: {final_purchase_counts.max()}")
print(f"Average purchases per customer: {final_purchase_counts.mean():.2f}")