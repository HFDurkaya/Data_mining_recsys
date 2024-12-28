import pandas as pd
import matplotlib.pyplot as plt
import os

def get_image_path(image_id: str) -> str:
    """
    Construct image path from image ID using forward slashes.
    """
    first_three = str(image_id)[:3]
    return f"data/images/{first_three}/{str(image_id)}"

def visualize_bought(
    user_id: str, 
    df: pd.DataFrame, 
    image_ids: pd.DataFrame
) -> None:
    '''
    Visualize bought items for a user.
    '''
    user_history = df[df['customer_id'] == user_id]
    num_transactions = len(user_history)
    col = 10
    rows = (num_transactions // col) + 1
    fig, axs = plt.subplots(rows, col, figsize=(col * 2, rows * 2))
    
    article_ids = user_history['article_id'].values
    
    for i, ax in enumerate(axs.flatten()):
        if i < len(article_ids):
            try:
                image_path = get_image_path(article_ids[i])
                image_path = f"{image_path}.jpg"
                image = plt.imread(image_path)
                ax.imshow(image)
            except Exception as e:
                print(f"Error loading image {article_ids[i]}: {e}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_candidates(
    candidates: pd.DataFrame, 
    image_ids: pd.DataFrame
) -> None:
    '''
    Visualize candidate items for the user.
    '''
    num_candidates = len(candidates)
    col = 10
    rows = (num_candidates // col) + 1
    fig, axs = plt.subplots(rows, col, figsize=(col * 2, rows * 2))
    
    for i, ax in enumerate(axs.flatten()):
        if i < len(candidates):
            try:
                image_id = candidates.iloc[i]['article_id']
                image_path = get_image_path(image_id)
                image_path = f"{image_path}.jpg"
                image = plt.imread(image_path)
                ax.imshow(image)
            except Exception as e:
                print(f"Error loading image {image_id}: {e}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()