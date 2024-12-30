import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



class HMSmartRecommender:
    """
        Define the class HMSmartRecommender
        This class is a smart recommendation system for H&M products that combines multiple recommendation strategies.
        It has the following attributes:
        - articles_df (pd.DataFrame): Product catalog with article details
        - transactions_df (pd.DataFrame): Customer purchase history
        - customers_df (pd.DataFrame): Customer information
        - tfidf (TfidfVectorizer): TF-IDF vectorizer for text feature extraction
        - feature_matrix: Computed TF-IDF feature matrix for articles
        It has the following methods:
        - __init__: Initialize the recommender system with necessary data
        - prepare_content_features: Prepare article features for recommendation including popularity scores and text-based features using TF-IDF
        - get_recommendations: Generate personalized recommendations for a customer
        - _get_customer_preferences: Calculate customer category preferences based on purchase history
        - _get_cold_start_recommendations: Generate recommendations for new customers without purchase history
    """
    
    def __init__(self, articles_df, transactions_df, customers_df):

        self.articles_df = articles_df.copy()
        self.transactions_df = transactions_df.copy()
        self.customers_df = customers_df.copy()
        self.tfidf = TfidfVectorizer(stop_words='english')
     
    # Prepare content features for articles    
    def prepare_content_features(self):

        # Calculate and normalize popularity scores
        popularity = self.transactions_df['article_id'].value_counts()
        self.articles_df['popularity_score'] = (
            self.articles_df['article_id']
            .map(popularity)
            .fillna(0)
        )
        self.articles_df['popularity_normalized'] = (
            self.articles_df['popularity_score'] / 
            self.articles_df['popularity_score'].max()
        )

        # Define text columns for feature extraction
        text_columns = [
            'product_type_name',
            'product_group_name',
            'garment_group_name',
            'colour_group_name',
            'section_name',
            'perceived_colour_value_name',
            'perceived_colour_master_name',
            'detail_desc'
        ]

        # Process text columns
        for col in text_columns:
            if col in self.articles_df.columns:
                if pd.api.types.is_categorical_dtype(self.articles_df[col]):
                    self.articles_df[col] = self.articles_df[col].astype(str)
                self.articles_df[col] = self.articles_df[col].fillna('')
        
        # Combine text features
        df_text = self.articles_df[text_columns]
        self.articles_df['combined_features'] = (
            df_text.apply(' '.join, axis=1)
            .str.lower()
        )

        # Create TF-IDF matrix
        self.feature_matrix = self.tfidf.fit_transform(
            self.articles_df['combined_features']
        )

    # Generate personalized recommendations for a customer
    def get_recommendations(
        self,
        customer_id,
        n_recommendations=5,
        similarity_weight=0.3,
        customer_preference_weight=0.4,
        popularity_weight=0.3
    ):
        
        # Get customer purchase history
        customer_history = self.transactions_df[
            self.transactions_df['customer_id'] == customer_id
        ]
        
        # Handle cold start case
        if customer_history.empty:
            return self._get_cold_start_recommendations(n_recommendations)

        # Calculate component scores
        category_preferences = self._get_customer_preferences(customer_history)
        recent_items = (
            customer_history
            .sort_values('t_dat', ascending=False)['article_id']
            .head(3)
            .tolist()
        )

        # Calculate similarity scores
        similarity_scores = np.zeros(len(self.articles_df))
        for article_id in recent_items:
            article_idx = self.articles_df[
                self.articles_df['article_id'] == article_id
            ].index[0]
            similarity_scores += cosine_similarity(
                self.feature_matrix[article_idx:article_idx + 1],
                self.feature_matrix
            ).flatten()
        similarity_scores /= max(1, len(recent_items))

        # Calculate preference and popularity scores
        preference_scores = (
            self.articles_df['product_group_name']
            .map(category_preferences)
            .fillna(0)
            .values
        )
        popularity_scores = self.articles_df['popularity_normalized'].values

        # Combine all scores with weights
        final_scores = (
            similarity_weight * similarity_scores +
            customer_preference_weight * preference_scores +
            popularity_weight * popularity_scores
        )

        # Exclude previously purchased items
        purchased_indices = self.articles_df[
            self.articles_df['article_id'].isin(customer_history['article_id'])
        ].index
        final_scores[purchased_indices] = -1

        # Get top recommendations
        top_indices = final_scores.argsort()[::-1][:n_recommendations]
        top_articles = self.articles_df.iloc[top_indices]['article_id'].values
        top_scores = final_scores[top_indices]

        return list(zip(top_articles, top_scores))

    # Calculate customer category preferences based on purchase history
    def _get_customer_preferences(self, customer_history):

        purchases_with_info = customer_history.merge(
            self.articles_df[['article_id', 'product_group_name']],
            on='article_id'
        )
        category_counts = purchases_with_info['product_group_name'].value_counts()
        return (category_counts / len(purchases_with_info)).to_dict()

    # Generate recommendations for new customers
    def _get_cold_start_recommendations(self, n_recommendations):

        popular_items = self.articles_df.sort_values(
            'popularity_normalized',
            ascending=False
        )
        diverse_recommendations = []
        seen_groups = set()

        for _, item in popular_items.iterrows():
            if item['product_group_name'] not in seen_groups:
                diverse_recommendations.append(
                    (item['article_id'], item['popularity_normalized'])
                )
                seen_groups.add(item['product_group_name'])
            if len(diverse_recommendations) >= n_recommendations:
                break

        return diverse_recommendations