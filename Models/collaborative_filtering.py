import pickle
import implicit
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import svds            

class ALSRecommender:
    
    """
        Define the class ALSRecommender
        This class is a collaborative filtering recommender based on Alternating Least Squares (ALS).
        It has the following attributes:
        - model (implicit.als.AlternatingLeastSquares): ALS model
        - sparse_matrix (csr_matrix): Sparse user-item matrix
        - customer_index (dict): Mapping of customer IDs to indices
        - article_index (dict): Mapping of article IDs to indices
        - reverse_customer_index (dict): Mapping of indices to customer IDs
        - reverse_article_index (dict): Mapping of indices to article IDs
        It has the following methods:
        - __init__: Initialize the ALS recommender with hyperparameters
        - _create_sparse_matrix_coo: Create a sparse user-item matrix in COO format
        - fit: Fit the ALS model on transaction, customer, and article data
        - recommend_items: Generate personalized recommendations for a customer
        - save: Save the ALS model to disk
        - load: Load the ALS model from disk
    """
    
    def __init__(
        self,
        factors=100,
        regularization=0.01,
        alpha=40,
        iterations=15,
        num_threads=4,
        use_gpu=False
    ):

        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            alpha=alpha,
            iterations=iterations,
            num_threads=num_threads,
            use_gpu=use_gpu
        )
        
        # Initialize storage
        self.sparse_matrix = None
        self.customer_index = None
        self.article_index = None
        self.reverse_customer_index = None
        self.reverse_article_index = None
        
    def _create_sparse_matrix_coo(self, transactions_df, customers_df, articles_df):

        # Create index mappings
        customer_index = {
            id_: i for i, id_ in enumerate(customers_df['customer_id'])
        }
        article_index = {
            id_: i for i, id_ in enumerate(articles_df['article_id'])
        }
        
        # Vectorized operations for index lookup
        customer_indices = np.array([
            customer_index[cid] for cid in transactions_df['customer_id']
        ])
        article_indices = np.array([
            article_index[aid] for aid in transactions_df['article_id']
        ])
        
        # Create sparse matrix
        sparse_matrix = coo_matrix(
            (
                np.ones(len(transactions_df)),
                (customer_indices, article_indices)
            ),
            shape=(len(customers_df), len(articles_df))
        ).tocsr()
        
        return sparse_matrix, customer_index, article_index
    
    def fit(self, transactions_df, customers_df, articles_df):

        # Create sparse matrix and indices
        (
            self.sparse_matrix,
            self.customer_index,
            self.article_index
        ) = self._create_sparse_matrix_coo(
            transactions_df,
            customers_df,
            articles_df
        )
        
        # Create reverse indices
        self.reverse_customer_index = {
            v: k for k, v in self.customer_index.items()
        }
        self.reverse_article_index = {
            v: k for k, v in self.article_index.items()
        }
        
        # Fit the model
        self.model.fit(self.sparse_matrix)

    def recommend_items(self, customer_id, n_items=10, filter_already_purchased=True):

        if customer_id not in self.customer_index:
            raise ValueError(f"Customer ID {customer_id} not found in training data")
            
        # Get recommendations from the model
        user_idx = self.customer_index[customer_id]
        item_scores = self.model.recommend(
            user_idx,
            self.sparse_matrix[user_idx],
            N=n_items,
            filter_already_liked_items=filter_already_purchased
        )
        
        # Convert indices back to article IDs and include scores
        return [
            (self.reverse_article_index[idx], float(score))
            for idx, score in zip(item_scores[0], item_scores[1])
        ]
    
    def save(self, path):

        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    @classmethod
    def load(cls, path):

        with open(path, 'rb') as f:
            return pickle.load(f)
        
        
        
        

class SVDRecommender:
    def __init__(self, factors=100):

        self.factors = factors
        
        # Initialize storage for learned matrices
        self.U = None  # User factors
        self.V = None  # Item factors
        self.sigma = None  # Singular values
        
        # Initialize indices
        self.customer_index = None
        self.article_index = None
        self.reverse_customer_index = None
        self.reverse_article_index = None
        self.sparse_matrix = None
        
    def _create_sparse_matrix_coo(self, transactions_df, customers_df, articles_df):

        # Create index mappings
        customer_index = {
            id_: i for i, id_ in enumerate(customers_df['customer_id'])
        }
        article_index = {
            id_: i for i, id_ in enumerate(articles_df['article_id'])
        }
        
        # Vectorized operations for index lookup
        customer_indices = np.array([
            customer_index[cid] for cid in transactions_df['customer_id']
        ])
        article_indices = np.array([
            article_index[aid] for aid in transactions_df['article_id']
        ])
        
        # Create sparse matrix
        sparse_matrix = coo_matrix(
            (
                np.ones(len(transactions_df)),
                (customer_indices, article_indices)
            ),
            shape=(len(customers_df), len(articles_df))
        ).tocsr()
        
        return sparse_matrix, customer_index, article_index
        
    def fit(self, transactions_df, customers_df, articles_df):

        # Create sparse matrix and indices
        (
            self.sparse_matrix,
            self.customer_index,
            self.article_index
        ) = self._create_sparse_matrix_coo(
            transactions_df,
            customers_df,
            articles_df
        )
        
        # Create reverse indices
        self.reverse_customer_index = {
            v: k for k, v in self.customer_index.items()
        }
        self.reverse_article_index = {
            v: k for k, v in self.article_index.items()
        }
        
        # Perform SVD
        U, sigma, Vt = svds(
            self.sparse_matrix,
            k=min(self.factors, min(self.sparse_matrix.shape) - 1)
        )
        
        # Store decomposed matrices
        self.U = U
        self.sigma = sigma
        self.V = Vt.T
        
    def recommend_items(self, customer_id, n_items=10, filter_already_purchased=True):

        if customer_id not in self.customer_index:
            raise ValueError(f"Customer ID {customer_id} not found in training data")
            
        # Get user vector and calculate scores
        user_idx = self.customer_index[customer_id]
        user_vector = self.U[user_idx]
        scores = user_vector @ np.diag(self.sigma) @ self.V.T
        
        # Filter out purchased items if requested
        if filter_already_purchased:
            user_items = self.sparse_matrix[user_idx].toarray().flatten()
            scores[user_items > 0] = -np.inf
            
        # Get top items
        top_items = np.argsort(scores)[::-1][:n_items]
        
        # Convert indices back to article IDs and include scores
        return [
            (self.reverse_article_index[idx], float(scores[idx]))
            for idx in top_items
        ]

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)