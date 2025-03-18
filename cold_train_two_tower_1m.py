import tensorflow as tf
import tensorflow_recommenders as tfrs
from tensorflow.keras import layers
import pandas as pd
import numpy as np

# Change this import to your 1M featurize function
from featurize_1m import featurize_movielens_1m

def df_to_tf_dataset(df, shuffle=True, batch_size=1024):
    """Convert a Pandas DataFrame into a tf.data.Dataset for training."""
    ds = tf.data.Dataset.from_tensor_slices(dict(df))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    return ds

class UserModel(tf.keras.Model):
    """User tower of a Two-Tower Model."""
    def __init__(self, num_users, embedding_dim=16):
        super().__init__()
        self.user_embedding = layers.Embedding(
            input_dim=num_users + 1,
            output_dim=embedding_dim
        )

    def call(self, inputs):
        return self.user_embedding(inputs["user_id"])

class ItemModel(tf.keras.Model):
    """Item tower of a Two-Tower Model."""
    def __init__(self, num_items, embedding_dim=16):
        super().__init__()
        self.item_embedding = layers.Embedding(
            input_dim=num_items + 1,
            output_dim=embedding_dim
        )

    def call(self, inputs):
        return self.item_embedding(inputs["item_id"])

class TwoTowerModel(tfrs.models.Model):
    """Two-Tower retrieval model with TFRS."""
    def __init__(self, user_model, item_model, loss):
        super().__init__()
        self.user_model = user_model
        self.item_model = item_model
        self.task = tfrs.tasks.Retrieval(loss=loss)

    def compute_loss(self, features, training=False):
        user_embeddings = self.user_model(features)
        item_embeddings = self.item_model(features)
        return self.task(user_embeddings, item_embeddings)

def evaluate_with_negative_sampling(model, test_df, train_df, num_items, K_values=[1, 3, 5, 10], num_neg=99, seed=42):
    """Evaluates Hit Rate and NDCG using 'Leave-One-Out + Negative Sampling'."""
    rng = np.random.default_rng(seed)

    users_in_test = list(set(test_df["user_id_int"]))

    # Gather all items seen in training or test (per user)
    user_seen_items = {}
    for df in [train_df, test_df]:
        for row in df.itertuples():
            user_seen_items.setdefault(row.user_id_int, set()).add(row.item_id_int)

    # Prepare accumulators
    hr_acc = {k: 0.0 for k in K_values}
    ndcg_acc = {k: 0.0 for k in K_values}

    def dcg_at_k(ranked_items, rel_item, k):
        """Compute DCG if rel_item is found within top k."""
        for i, it in enumerate(ranked_items[:k]):
            if it == rel_item:
                return 1.0 / np.log2(i + 2)
        return 0.0

    def idcg_at_k(k):
        """Ideal DCG is 1.0 since there's only one positive item per user."""
        return 1.0

    for user_id in users_in_test:
        # We assume there's only 1 item per user in test (common in "leave-one-out")
        test_items = test_df[test_df["user_id_int"] == user_id]["item_id_int"].unique()
        if len(test_items) == 0:
            continue
        positive_item = test_items[0]

        seen = user_seen_items.get(user_id, set())
        all_candidates = np.arange(1, num_items + 1)
        valid_negatives = np.setdiff1d(all_candidates, list(seen), assume_unique=True)

        # If we can't even find `num_neg` unique negatives, skip
        if len(valid_negatives) < num_neg:
            continue

        negatives = rng.choice(valid_negatives, size=num_neg, replace=False)
        candidates = np.concatenate(([positive_item], negatives))

        # Get model scores
        user_emb = model.user_model({"user_id": tf.constant([user_id], dtype=tf.int32)})
        item_embs = model.item_model({"item_id": tf.constant(candidates, dtype=tf.int32)})
        scores = tf.reduce_sum(user_emb * item_embs, axis=1).numpy()

        # Sort in descending order
        ranked_items = candidates[np.argsort(-scores)]

        for k in K_values:
            # Hit Rate
            if positive_item in ranked_items[:k]:
                hr_acc[k] += 1.0
            # NDCG
            ndcg_acc[k] += dcg_at_k(ranked_items, positive_item, k) / idcg_at_k(k)

    num_users_eval = len(users_in_test)
    if num_users_eval > 0:
        for k in K_values:
            hr_acc[k] /= num_users_eval
            ndcg_acc[k] /= num_users_eval

    return hr_acc, ndcg_acc

if __name__ == "__main__":
    # 1) Load MovieLens-1M (instead of 100k)
    train_df, test_df, users_df, items_df = featurize_movielens_1m()  # Adjust any path/parameters as needed
    print(f"Original train size: {len(train_df)}; test size: {len(test_df)}")

    # 2) Map user_id and item_id to integer IDs
    unique_user_ids_str = sorted(set(train_df["user_id"].astype(str)).union(set(test_df["user_id"].astype(str))))
    unique_item_ids_str = sorted(set(train_df["item_id"].astype(str)).union(set(test_df["item_id"].astype(str))))

    user_id_map = {uid_str: i + 1 for i, uid_str in enumerate(unique_user_ids_str)}
    item_id_map = {iid_str: i + 1 for i, iid_str in enumerate(unique_item_ids_str)}

    train_df["user_id_int"] = train_df["user_id"].astype(str).map(user_id_map).astype(np.int32)
    test_df["user_id_int"] = test_df["user_id"].astype(str).map(user_id_map).astype(np.int32)
    train_df["item_id_int"] = train_df["item_id"].astype(str).map(item_id_map).astype(np.int32)
    test_df["item_id_int"] = test_df["item_id"].astype(str).map(item_id_map).astype(np.int32)

    num_users = len(user_id_map)
    num_items = len(item_id_map)

    # 3) Randomly pick 20% of the users to be cold
    rng = np.random.default_rng(42)
    all_user_ids = list(train_df["user_id_int"].unique())
    cold_count = int(0.20 * len(all_user_ids))
    cold_users = rng.choice(all_user_ids, size=cold_count, replace=False)

    # 4) For cold users, keep only 1 interaction in the training data
    cold_df = train_df[train_df["user_id_int"].isin(cold_users)]
    non_cold_df = train_df[~train_df["user_id_int"].isin(cold_users)]

    # groupby user => pick exactly one row per user
    def pick_single_interaction(group):
        return group.sample(1, random_state=42)

    cold_df_single = cold_df.groupby("user_id_int", as_index=False).apply(pick_single_interaction)
    # groupby(...).apply(...) can produce a multi-index, so fix that
    cold_df_single.reset_index(drop=True, inplace=True)

    # Reassemble our "reduced" training set
    reduced_train_df = pd.concat([non_cold_df, cold_df_single], ignore_index=True)
    print(f"Reduced train size: {len(reduced_train_df)}")

    # 5) Prepare training tf.data.Dataset
    reduced_train_df["label"] = 1.0
    train_ds = df_to_tf_dataset({
        "user_id": reduced_train_df["user_id_int"].values,
        "item_id": reduced_train_df["item_id_int"].values,
    }, shuffle=True, batch_size=1024)

    # 6) Create and train our Two-Tower model
    user_model = UserModel(num_users, embedding_dim=16)
    item_model = ItemModel(num_items, embedding_dim=16)

    # We'll use a BinaryCrossentropy for the TFRS retrieval task
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    model = TwoTowerModel(user_model=user_model, item_model=item_model, loss=loss)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001))

    print("\nTraining the Two-Tower model on the reduced (cold-user) training set...")
    model.fit(train_ds, epochs=5)

    # 7) Evaluate with negative sampling
    print("\nEvaluating the model (Leave-One-Out + Negative Sampling)...")
    hr, ndcg = evaluate_with_negative_sampling(
        model,
        test_df,
        reduced_train_df,
        num_items,
        K_values=[1, 3, 5, 10]
    )

    # 8) Print results
    print("\nEvaluation Results:")
    print("=" * 40)
    print(f"{'k':<10}{'Hit Rate':<15}{'NDCG':<15}")
    print("-" * 40)
    for k in [1, 3, 5, 10]:
        print(f"{k:<10}{hr[k]:.4f}{'':5}{ndcg[k]:.4f}")
    print("=" * 40)
