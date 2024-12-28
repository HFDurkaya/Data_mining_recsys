import streamlit as st
import os
import pickle
import pandas as pd
import math
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
import random

# Page config
st.set_page_config(page_title="H&M Fashion Recommender System Demo", layout="wide")

# Title
st.title("H&M Fashion Recommender System Demo")

@st.cache_data
def load_random_users(n_users=25):
    """Load random 25 users who have between 20 and 40 purchases"""
    transactions = pd.read_pickle("data/transactions.pkl")
    user_purchase_counts = transactions['customer_id'].value_counts()
    filtered_users = user_purchase_counts[
        (user_purchase_counts > 20) & 
        (user_purchase_counts < 40)
    ].index.tolist()
    # Randomly select 25 users
    random_users = random.sample(filtered_users, min(n_users, len(filtered_users)))
    return random_users

@st.cache_data
def load_data_for_user(user_id):
    """Load only necessary data for a specific user"""
    articles = pd.read_pickle("data/articles.pkl")
    transactions = pd.read_pickle("data/transactions.pkl")
    
    user_transactions = transactions[transactions['customer_id'] == user_id].copy()
    user_transactions = user_transactions.merge(
        articles[['article_id', 'product_type_name']], 
        on='article_id', 
        how='left'
    )
    user_transactions = user_transactions.sort_values('t_dat', ascending=False)
    
    return articles, user_transactions

@st.cache_resource
def load_resnet_model():
    """Load pre-trained ResNet model"""
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model

def process_image(image_path):
    """Process image for ResNet"""
    try:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        return image.unsqueeze(0)
    except:
        return None

def get_image_embedding(model, image_path):
    """Get image embedding using ResNet"""
    image = process_image(image_path)
    if image is not None:
        with torch.no_grad():
            embedding = model(image)
            return embedding.squeeze().numpy()
    return None

def cluster_recommendations(candidates_df, n_clusters=3):
    """Cluster recommendations based on visual similarity"""
    model = load_resnet_model()
    embeddings = []
    valid_indices = []
    
    for idx, row in candidates_df.iterrows():
        article_id = str(row['article_id']).zfill(10)
        folder_prefix = article_id[:3]
        image_path = f"data/images/{folder_prefix}/{article_id}.jpg"
        
        if os.path.exists(image_path):
            embedding = get_image_embedding(model, image_path)
            if embedding is not None:
                embeddings.append(embedding.flatten())
                valid_indices.append(idx)
    
    if not embeddings:
        return None
        
    embeddings = np.array(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    candidates_df.loc[valid_indices, 'cluster'] = clusters
    
    return candidates_df

def load_model(model_type):
    """Load pre-trained model from weights"""
    model_path = f"Weights/{model_type.lower().replace(' ', '_')}_recommender.pkl"
    
    if not os.path.exists(model_path):
        st.error(f"Model weights not found at {model_path}.")
        st.stop()
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def display_items_grid(items_df, num_cols=8, include_score=False, width=100):
    """Display items in a grid with images"""
    num_items = len(items_df)
    num_rows = math.ceil(num_items / num_cols)
    
    for row in range(num_rows):
        cols = st.columns(num_cols)
        for col in range(num_cols):
            idx = row * num_cols + col
            if idx < num_items:
                with cols[col]:
                    item = items_df.iloc[idx]
                    article_id = str(item['article_id']).zfill(10)
                    folder_prefix = article_id[:3]
                    image_path = f"data/images/{folder_prefix}/{article_id}.jpg"
                    
                    if os.path.exists(image_path):
                        caption = f"""
                            ID: {item['article_id']}
                            {item['product_type_name']}
                            {item['t_dat'].date() if 't_dat' in item else ''}
                            {f"Score: {item['score']:.3f}" if include_score else ''}
                            {f"Cluster: {int(item['cluster'])+1}" if 'cluster' in item else ''}
                        """
                        st.image(image_path, caption=caption, width=width)
                    else:
                        st.write(f"No image: {item['article_id']}")

# Sidebar
st.sidebar.header('Settings')
model_type = st.sidebar.selectbox(
    'Select Model Type',
    ['Numerical CBF', 'Hybrid', 'ALS', 'SVD']
)

# Load random users
try:
    random_users = load_random_users()
    if not random_users:
        st.error("No users found.")
        st.stop()
except FileNotFoundError:
    st.error("Transaction data file not found.")
    st.stop()

# Main content
st.header('Fashion Recommendations')

# User selection dropdown with random users
st.write(f"Showing 25 random users with less than 40 purchases")
user_id = st.selectbox('Choose User ID:', random_users)

# Add a button to start processing
if st.button('Load User Data and Generate Recommendations'):
    # Load data for selected user
    with st.spinner('Loading user data...'):
        articles, user_transactions = load_data_for_user(user_id)
    
    # Display purchase history
    st.subheader(f"Purchase History for User {user_id}")
    st.write(f"Showing all purchases (Total: {len(user_transactions)})")
    display_items_grid(user_transactions)
    
    # Generate and display recommendations
    with st.spinner('Loading model and generating recommendations...'):
        model = load_model(model_type)
        candidates = model.recommend_items(user_id, n_items=20, filter_already_purchased=True)
        candidates_df = pd.DataFrame(candidates, columns=['article_id', 'score'])
        candidates_df = candidates_df.merge(
            articles[['article_id', 'product_type_name']], 
            on='article_id', 
            how='left'
        )
    
    # Perform visual similarity clustering
    with st.spinner('Analyzing visual similarities...'):
        candidates_df = cluster_recommendations(candidates_df)
    
    # Display recommendations by cluster
    st.subheader(f"Top 20 Recommendations for User {user_id}")
    if candidates_df is not None and 'cluster' in candidates_df.columns:
        for cluster in sorted(candidates_df['cluster'].unique()):
            st.write(f"\nCluster {cluster + 1} - Visually Similar Items:")
            cluster_items = candidates_df[candidates_df['cluster'] == cluster]
            display_items_grid(cluster_items, include_score=True)
            
        # Add similarity analysis
        st.subheader("Visual Similarity Analysis")
        st.write("Items in each cluster share similar visual characteristics such as:")
        st.write("- Color patterns and schemes")
        st.write("- Style elements and design features")
        st.write("- Product type and category similarities")
    else:
        display_items_grid(candidates_df, include_score=True)

# Footer
st.markdown("---")
st.markdown("H&M Fashion Recommender System Demo")