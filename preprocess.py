import numpy as np
import pandas as pd

def create_negative_samples(ratings_df, n_negatives=5, random_seed=42):
    """
    Create negative samples for each user by sampling from items they haven't interacted with.
    
    Args:
        ratings_df: DataFrame containing user-item interactions
        n_negatives: Number of negative samples per positive sample
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with positive and negative samples, including a label column
    """
    np.random.seed(random_seed)
    all_items = ratings_df['item_id'].unique()
    user_groups = ratings_df.groupby('user_id')
    user_list = []
    item_list = []
    label_list = []
    
    for user_id, group in user_groups:
        positive_items = set(group['item_id'].values)
        possible_negatives = list(set(all_items) - positive_items)
        
        # For very active users, we might not have enough negative samples
        if len(possible_negatives) < n_negatives:
            sampled_negatives = possible_negatives
        else:
            sampled_negatives = np.random.choice(possible_negatives, size=n_negatives, replace=False)
        
        # Add positive samples
        for p_item in positive_items:
            user_list.append(user_id)
            item_list.append(p_item)
            label_list.append(1)
        
        # Add negative samples
        for n_item in sampled_negatives:
            user_list.append(user_id)
            item_list.append(n_item)
            label_list.append(0)
    
    out_df = pd.DataFrame({
        'user_id': user_list,
        'item_id': item_list,
        'label': label_list
    })
    
    return out_df

if __name__ == "__main__":
    # Import the new featurize function for the 1M dataset
    from featurize_1m import featurize_movielens_1m
    
    # Load the data using the 1M function
    train_df, test_df, users_df, items_df = featurize_movielens_1m(data_path='data/ml-1m')
    
    print("Original train shape:", train_df.shape, "Original test shape:", test_df.shape)
    
    # For the 1M dataset, consider reducing n_negatives to avoid memory issues
    n_neg = 3  # Reduced from 5 to 3 to manage memory better with the larger dataset
    
    # Process in batches for the larger dataset
    print("Creating negative samples for train set...")
    train_preprocessed = create_negative_samples(train_df, n_negatives=n_neg)
    
    print("Creating negative samples for test set...")
    test_preprocessed = create_negative_samples(test_df, n_negatives=n_neg)
    
    print("Train preprocessed shape:", train_preprocessed.shape)
    print("Test preprocessed shape:", test_preprocessed.shape)
    
    # Optionally save to disk to avoid reprocessing
    print("Saving preprocessed data...")
    train_preprocessed.to_csv('data/ml-1m/train_preprocessed.csv', index=False)
    test_preprocessed.to_csv('data/ml-1m/test_preprocessed.csv', index=False)
    print("Preprocessing complete.")