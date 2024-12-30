#Necessary Libraries
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
from sklearn.metrics.pairwise import cosine_similarity
import random

# Page config
st.set_page_config(page_title="H&M Fashion Recommender System Demo", layout="wide")

# Title
st.title("H&M Fashion Recommender System Demo")

# Load data
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

# Load data for a specific user
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

# Load pre-trained ResNet model
@st.cache_resource
def load_resnet_model():
    """Load pre-trained ResNet model"""
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model

# Functions for clustering and recommendations
def assign_recommendations_to_clusters(purchase_embeddings, rec_embeddings, purchase_clusters):
    """
    Assign each recommended item to the most similar purchase cluster
    based on visual embeddings
    """
    # Calculate cosine similarity between each recommendation and purchase cluster centers
    cluster_centers = {}
    for cluster in np.unique(purchase_clusters):
        cluster_mask = purchase_clusters == cluster
        cluster_centers[cluster] = np.mean(purchase_embeddings[cluster_mask], axis=0)
    
    rec_cluster_assignments = []
    rec_cluster_similarities = []
    
    for rec_embedding in rec_embeddings:
        similarities = []
        for cluster in cluster_centers:
            similarity = cosine_similarity(
                rec_embedding.reshape(1, -1),
                cluster_centers[cluster].reshape(1, -1)
            )[0][0]
            similarities.append((cluster, similarity))
        
        # Assign to most similar cluster
        best_cluster, best_similarity = max(similarities, key=lambda x: x[1])
        rec_cluster_assignments.append(best_cluster)
        rec_cluster_similarities.append(best_similarity)
    
    return np.array(rec_cluster_assignments), np.array(rec_cluster_similarities)


def determine_optimal_clusters(n_purchases):
    """
    Determine the optimal number of clusters based on purchase history size
    """
    # Minimum 3 clusters, maximum 7 clusters
    # Scale clusters with purchase count, but keep it manageable
    if n_purchases < 15:
        return 3
    elif n_purchases < 25:
        return 4
    elif n_purchases < 35:
        return 5
    else:
        return 6

# Main functions
# Cluster purchase history and assign recommendations to clusters
def cluster_and_recommend(user_transactions, candidates_df):
    """
    Cluster purchase history and associate recommendations with purchase clusters
    """
    model = load_resnet_model()
    
    # Process purchase history
    purchase_embeddings = []
    purchase_valid_indices = []
    
    for idx, row in user_transactions.iterrows():
        article_id = str(row['article_id']).zfill(10)
        folder_prefix = article_id[:3]
        image_path = f"data/images/{folder_prefix}/{article_id}.jpg"
        
        if os.path.exists(image_path):
            embedding = get_image_embedding(model, image_path)
            if embedding is not None:
                purchase_embeddings.append(embedding.flatten())
                purchase_valid_indices.append(idx)
    
    if not purchase_embeddings:
        return None, None, None
    
    purchase_embeddings = np.array(purchase_embeddings)
    
    # Determine optimal number of clusters
    n_clusters = determine_optimal_clusters(len(purchase_embeddings))
    
    # Cluster purchase history
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    purchase_clusters = kmeans.fit_predict(purchase_embeddings)
    user_transactions.loc[purchase_valid_indices, 'cluster'] = purchase_clusters
    
    # Process recommendations
    rec_embeddings = []
    rec_valid_indices = []
    
    for idx, row in candidates_df.iterrows():
        article_id = str(row['article_id']).zfill(10)
        folder_prefix = article_id[:3]
        image_path = f"data/images/{folder_prefix}/{article_id}.jpg"
        
        if os.path.exists(image_path):
            embedding = get_image_embedding(model, image_path)
            if embedding is not None:
                rec_embeddings.append(embedding.flatten())
                rec_valid_indices.append(idx)
    
    if not rec_embeddings:
        return None, None, None
    
    rec_embeddings = np.array(rec_embeddings)
    
    # Assign recommendations to purchase clusters
    rec_clusters, rec_similarities = assign_recommendations_to_clusters(
        purchase_embeddings,
        rec_embeddings,
        purchase_clusters
    )
    
    candidates_df.loc[rec_valid_indices, 'cluster'] = rec_clusters
    candidates_df.loc[rec_valid_indices, 'cluster_similarity'] = rec_similarities
    
    return user_transactions, candidates_df, purchase_embeddings

# Image processing functions
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

# Get image embedding using ResNet
def get_image_embedding(model, image_path):
    """Get image embedding using ResNet"""
    image = process_image(image_path)
    if image is not None:
        with torch.no_grad():
            embedding = model(image)
            return embedding.squeeze().numpy()
    return None

# Cluster items based on visual similarity
def cluster_items(items_df, n_clusters=5):
    """Cluster items based on visual similarity"""
    model = load_resnet_model()
    embeddings = []
    valid_indices = []
    
    for idx, row in items_df.iterrows():
        article_id = str(row['article_id']).zfill(10)
        folder_prefix = article_id[:3]
        image_path = f"data/images/{folder_prefix}/{article_id}.jpg"
        
        if os.path.exists(image_path):
            embedding = get_image_embedding(model, image_path)
            if embedding is not None:
                embeddings.append(embedding.flatten())
                valid_indices.append(idx)
    
    if not embeddings:
        return None, None
        
    embeddings = np.array(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    items_df.loc[valid_indices, 'cluster'] = clusters
    
    return items_df, embeddings

# Find similar clusters between purchases and recommendations
def find_similar_clusters(purchase_embeddings, rec_embeddings, rec_clusters):
    """Find similar clusters between purchases and recommendations"""
    similarities = {}
    for i, p_emb in enumerate(purchase_embeddings):
        cluster_similarities = []
        for j, r_emb in enumerate(rec_embeddings):
            similarity = np.dot(p_emb, r_emb) / (np.linalg.norm(p_emb) * np.linalg.norm(r_emb))
            cluster_similarities.append((rec_clusters[j], similarity))
        
        # Get most similar recommendation cluster for this purchase cluster
        avg_similarities = {}
        for cluster, sim in cluster_similarities:
            if cluster not in avg_similarities:
                avg_similarities[cluster] = []
            avg_similarities[cluster].append(sim)
        
        similarities[i] = max(
            [(c, sum(s)/len(s)) for c, s in avg_similarities.items()],
            key=lambda x: x[1]
        )[0]
    
    return similarities

# Load pre-trained model from weights
def load_model(model_type):
    """Load pre-trained model from weights"""
    model_path = f"Weights/{model_type.lower().replace(' ', '_')}_recommender.pkl"
    
    if not os.path.exists(model_path):
        st.error(f"Model weights not found at {model_path}.")
        st.stop()
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Display items in a grid with images
def display_items_grid(items_df, num_cols=10, include_score=False, width=100):
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

# Modified main section of your Streamlit app
if st.button('Load User Data and Generate Recommendations'):
    # Load data for selected user
    with st.spinner('Loading user data...'):
        articles, user_transactions = load_data_for_user(user_id)
    
    # Generate recommendations
    with st.spinner('Loading model and generating recommendations...'):
        model = load_model(model_type)
        candidates = model.recommend_items(user_id, n_items=100, filter_already_purchased=True)
        candidates_df = pd.DataFrame(candidates, columns=['article_id', 'score'])
        candidates_df = candidates_df.merge(
            articles[['article_id', 'product_type_name']], 
            on='article_id', 
            how='left'
        )
    
    # Cluster purchases and assign recommendations
    clustered_purchases, clustered_recommendations, purchase_embeddings = cluster_and_recommend(
        user_transactions,
        candidates_df
    )
    
    if clustered_purchases is not None and clustered_recommendations is not None:
        # Display purchase clusters
        st.subheader(f"Purchase History Clusters for User {user_id}")
        for cluster in sorted(clustered_purchases['cluster'].unique()):
            st.write(f"\nPurchase Cluster {cluster + 1}:")
            cluster_items = clustered_purchases[clustered_purchases['cluster'] == cluster]
            display_items_grid(cluster_items)
        
        # Display recommendations by cluster
        st.subheader("Recommendations by Similar Purchase Cluster")
        for cluster in sorted(clustered_purchases['cluster'].unique()):
            st.write(f"\nRecommendations similar to Purchase Cluster {cluster + 1}:")
            cluster_recs = clustered_recommendations[
                clustered_recommendations['cluster'] == cluster
            ].sort_values('cluster_similarity', ascending=False)
            display_items_grid(cluster_recs, include_score=True)
            
            # Show cluster statistics
            avg_similarity = cluster_recs['cluster_similarity'].mean()
            num_items = len(cluster_recs)
            st.write(f"""
                Cluster Statistics:
                - Average similarity to purchase cluster: {avg_similarity:.3f}
                - Number of recommended items: {num_items}
            """)
    else:
        st.error("Could not generate clusters for purchases and recommendations.")
# Footer
st.markdown("---")
st.markdown("H&M Fashion Recommender System Demo")