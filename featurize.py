import pandas as pd
from sklearn.model_selection import train_test_split
import os

def featurize_movielens_100k(data_path='data/ml-100k'):
    ratings = pd.read_csv(
        os.path.join(data_path, 'u.data'),
        sep='\t',
        names=['user_id', 'item_id', 'rating', 'timestamp']
    )

    users = pd.read_csv(
        os.path.join(data_path, 'u.user'),
        sep='|',
        names=['user_id', 'age', 'gender', 'occupation', 'zip_code']
    )

    items = pd.read_csv(
        os.path.join(data_path, 'u.item'),
        sep='|',
        names=['item_id','title','release_date','video_release_date','imdb_url']
               + [f'genre_{i}' for i in range(19)],
        encoding='latin-1'
    )

    print("Ratings shape:", ratings.shape)
    print("Users shape:", users.shape)
    print("Items shape:", items.shape)

    train, test = train_test_split(ratings, test_size=0.2, random_state=42)
    print("Train size:", train.shape[0], "Test size:", test.shape[0])

    train_user_ids = set(train['user_id'].unique())
    test_user_ids = set(test['user_id'].unique())
    cold_users = test_user_ids - train_user_ids
    warm_users = test_user_ids & train_user_ids

    print(f"Number of cold users in test: {len(cold_users)}")
    print(f"Number of warm users in test: {len(warm_users)}")

    return train, test, users, items

if __name__ == "__main__":
    train_df, test_df, users_df, items_df = featurize_movielens_100k()
    print("Featurization complete.")
