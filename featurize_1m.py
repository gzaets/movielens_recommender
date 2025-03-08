import pandas as pd
from sklearn.model_selection import train_test_split
import os

def featurize_movielens_1m(data_path='data/ml-1m'):
    """
    Load and featurize the MovieLens 1M dataset.
    
    Args:
        data_path: Path to the MovieLens 1M dataset
        
    Returns:
        train_df: Training data
        test_df: Test data
        users_df: User features
        items_df: Item features
    """
    # Load ratings - the file format is different from 100K
    ratings = pd.read_csv(
        os.path.join(data_path, 'ratings.dat'),
        sep='::',
        names=['user_id', 'item_id', 'rating', 'timestamp'],
        engine='python'  # Use python engine to handle :: separator
    )
    
    # Load users - the file format and columns are different
    users = pd.read_csv(
        os.path.join(data_path, 'users.dat'),
        sep='::',
        names=['user_id', 'gender', 'age', 'occupation', 'zip_code'],
        engine='python'
    )
    
    # Load movies - the file format and columns are different
    # In 1M dataset, genres are pipe-separated strings rather than individual columns
    movies = pd.read_csv(
        os.path.join(data_path, 'movies.dat'),
        sep='::',
        names=['item_id', 'title', 'genres'],
        engine='python',
        encoding='latin-1'
    )
    
    # Create separate columns for genres (similar to the 100K dataset approach)
    # Split the genres string and create binary indicators
    genre_list = set()
    for genres in movies['genres'].str.split('|'):
        if genres is not None:
            genre_list.update(genres)
            
    for genre in genre_list:
        movies[f'genre_{genre}'] = movies['genres'].apply(
            lambda x: 1 if genre in str(x).split('|') else 0
        )
    
    print("Ratings shape:", ratings.shape)
    print("Users shape:", users.shape)
    print("Movies shape:", movies.shape)
    
    # Split into train and test sets
    train, test = train_test_split(ratings, test_size=0.2, random_state=42)
    print("Train size:", train.shape[0], "Test size:", test.shape[0])
    
    # Check for cold-start users
    train_user_ids = set(train['user_id'].unique())
    test_user_ids = set(test['user_id'].unique())
    cold_users = test_user_ids - train_user_ids
    warm_users = test_user_ids & train_user_ids
    print(f"Number of cold users in test: {len(cold_users)}")
    print(f"Number of warm users in test: {len(warm_users)}")
    
    return train, test, users, movies

if __name__ == "__main__":
    train_df, test_df, users_df, items_df = featurize_movielens_1m()
    print("Featurization complete.")