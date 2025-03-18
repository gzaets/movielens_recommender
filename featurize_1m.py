import pandas as pd
import os

def featurize_movielens_1m(data_path='data/ml-1m'):
    """
    Loads MovieLens-1M and applies the following constraints:
      1) Keep only ratings == 5 (perfect five-star).
      2) For each user, keep only those who have >= 10 such ratings (10-core).
      3) Sort by timestamp and do 'leave-one-out': last item to test, rest to train.
      4) Return train_df, test_df, users_df, movies_df.

    The resulting train_df/test_df set ensures each user has at least
    9 items in train + 1 item in test = 10 total 5-star items.

    If you'd like to filter items with fewer than X ratings, you can add
    an "item-core" step as well (commented out below).
    """

    # -------------------------------------------------------------------------
    # 1. Load raw .dat files for ratings/users/movies
    # -------------------------------------------------------------------------
    ratings = pd.read_csv(
        os.path.join(data_path, 'ratings.dat'),
        sep='::',
        names=['user_id', 'item_id', 'rating', 'timestamp'],
        engine='python'
    )
    users = pd.read_csv(
        os.path.join(data_path, 'users.dat'),
        sep='::',
        names=['user_id', 'gender', 'age', 'occupation', 'zip_code'],
        engine='python'
    )
    movies = pd.read_csv(
        os.path.join(data_path, 'movies.dat'),
        sep='::',
        names=['item_id', 'title', 'genres'],
        engine='python',
        encoding='latin-1'
    )

    # -------------------------------------------------------------------------
    # 2. Keep only rating==5
    # -------------------------------------------------------------------------
    ratings = ratings[ratings["rating"] == 5].copy()

    # -------------------------------------------------------------------------
    # 3. 10-core filtering: keep only users with >= 10 total 5-star ratings
    # -------------------------------------------------------------------------
    user_counts = ratings.groupby("user_id").size()
    valid_users = user_counts[user_counts >= 10].index
    ratings = ratings[ratings["user_id"].isin(valid_users)]

    # -------------------------------------------------------------------------
    # (Optional) item-core filtering, if needed
    # item_counts = ratings.groupby("item_id").size()
    # valid_items = item_counts[item_counts >= 5].index  # e.g. 5-core for items
    # ratings = ratings[ratings["item_id"].isin(valid_items)]

    # -------------------------------------------------------------------------
    # 4. Sort by timestamp and do "leave-one-out"
    # -------------------------------------------------------------------------
    ratings = ratings.sort_values(["user_id", "timestamp"])

    train_rows = []
    test_rows = []
    for uid, group in ratings.groupby("user_id"):
        # last row => test, rest => train
        test_item = group.iloc[-1]
        train_items = group.iloc[:-1]
        test_rows.append(test_item)
        train_rows.append(train_items)

    train_df = pd.concat(train_rows, axis=0)
    test_df = pd.DataFrame(test_rows)

    print("\nAfter rating==5 & 10-core + LOO:")
    print(f"Train size: {len(train_df)}   Test size: {len(test_df)}")
    print(f"Unique users in train: {train_df['user_id'].nunique()}, in test: {test_df['user_id'].nunique()}")
    print("\nUsers shape:", users.shape)
    print("Movies shape:", movies.shape)

    return train_df, test_df, users, movies

if __name__ == "__main__":
    tr, te, u, m = featurize_movielens_1m()
    print("\nFeaturization complete.")
