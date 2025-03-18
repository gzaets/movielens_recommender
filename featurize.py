# featurize_100k.py
import pandas as pd
import os

def featurize_movielens_100k(data_path='data/ml-100k'):
    # Load ratings, users, and items
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

    # Sort by timestamp
    ratings = ratings.sort_values(['user_id', 'timestamp'])

    train_rows = []
    test_rows = []

    # Group by user, do leave-one-out
    for uid, group in ratings.groupby('user_id'):
        if len(group) < 2:
            continue
        test_item = group.iloc[-1]
        train_items = group.iloc[:-1]
        test_rows.append(test_item)
        train_rows.append(train_items)

    train_df = pd.concat(train_rows, axis=0)
    test_df = pd.DataFrame(test_rows)

    print("After leave-one-out split:")
    print("Train size:", len(train_df), "Test size:", len(test_df))
    print("Users shape:", users.shape)
    print("Items shape:", items.shape)

    return train_df, test_df, users, items

if __name__ == "__main__":
    tr, te, u, i = featurize_movielens_100k()
    print("Featurization complete.")
