import tensorflow as tf
import tensorflow_recommenders as tfrs
from tensorflow.keras import layers
import pandas as pd
import numpy as np

# Your function that loads ML-1M splits
from featurize_1m import featurize_movielens_1m

########################################
# Convert DataFrame -> tf.data.Dataset
########################################
def df_to_tf_dataset(df, shuffle=True, batch_size=1024):
    ds = tf.data.Dataset.from_tensor_slices(dict(df))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    return ds

########################################
# UserModel, ItemModel, TwoTowerModel
########################################
class UserModel(tf.keras.Model):
    """User tower of Two-Tower Model."""
    def __init__(self, num_users, embedding_dim=16):
        super().__init__()
        self.user_embedding = layers.Embedding(
            input_dim=num_users + 1,
            output_dim=embedding_dim
        )
    def call(self, inputs):
        return self.user_embedding(inputs["user_id"])

class ItemModel(tf.keras.Model):
    """Item tower of Two-Tower Model."""
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

########################################
# Evaluate: Leave-One-Out + Negative Sampling
########################################
def evaluate_with_negative_sampling(
    model,
    test_df,
    train_df,
    num_items,
    K_values=[1, 3, 5, 10],
    num_neg=99,
    seed=42
):
    """
    Evaluates HR@K / NDCG@K via 'Leave-One-Out + Negative Sampling'.
    Each user in the test set has 1 positive item + num_neg negatives.
    """
    rng = np.random.default_rng(seed)

    users_in_test = set(test_df["user_id_int"])
    user_seen_items = {}

    # Gather items each user saw in train OR test, to avoid picking positives as negatives
    for df_ in [train_df, test_df]:
        for row in df_.itertuples():
            user_seen_items.setdefault(row.user_id_int, set()).add(row.item_id_int)

    hr_acc = {k: 0.0 for k in K_values}
    ndcg_acc = {k: 0.0 for k in K_values}

    def dcg_at_k(ranked_items, pos_item, k):
        """Compute DCG if pos_item is within top k."""
        for i, it in enumerate(ranked_items[:k]):
            if it == pos_item:
                return 1.0 / np.log2(i + 2)
        return 0.0

    def idcg_at_k(k):
        # Since there's only 1 positive, the ideal DCG is always 1.0 if found in top-k.
        return 1.0

    # Evaluate each test user
    for user_id in users_in_test:
        # "Leave-One-Out": each user has 1 test item
        test_items = test_df[test_df["user_id_int"] == user_id]["item_id_int"].unique()
        if len(test_items) == 0:
            continue
        pos_item = test_items[0]

        seen = user_seen_items.get(user_id, set())
        all_candidates = np.arange(1, num_items + 1)
        valid_negatives = np.setdiff1d(all_candidates, list(seen), assume_unique=True)
        if len(valid_negatives) < num_neg:
            # Not enough distinct negatives => skip
            continue

        negs = rng.choice(valid_negatives, size=num_neg, replace=False)
        candidates = np.concatenate(([pos_item], negs))

        # Score them
        user_emb = model.user_model({"user_id": tf.constant([user_id], dtype=tf.int32)})
        item_embs = model.item_model({"item_id": tf.constant(candidates, dtype=tf.int32)})
        scores = tf.reduce_sum(user_emb * item_embs, axis=1).numpy()

        # Sort in descending order
        ranked_items = candidates[np.argsort(-scores)]

        for k in K_values:
            # Hit Rate
            if pos_item in ranked_items[:k]:
                hr_acc[k] += 1.0
            # NDCG
            ndcg_acc[k] += dcg_at_k(ranked_items, pos_item, k) / idcg_at_k(k)

    num_eval_users = len(users_in_test)
    if num_eval_users > 0:
        for k in K_values:
            hr_acc[k] /= num_eval_users
            ndcg_acc[k] /= num_eval_users

    return hr_acc, ndcg_acc

########################################
# Main: Cold-Start w/ MovieLens-1M + Augmented
########################################
if __name__ == "__main__":
    # 1) Load ML-1M splits (train_df, test_df, etc.)
    train_df, test_df, users_df, items_df = featurize_movielens_1m()
    print(f"Loaded ML-1M data: train={len(train_df)}, test={len(test_df)}")

    # 2) Build ID maps
    unique_users_str = sorted(set(train_df["user_id"].astype(str)) | set(test_df["user_id"].astype(str)))
    unique_items_str = sorted(set(train_df["item_id"].astype(str)) | set(test_df["item_id"].astype(str)))

    user_id_map = {u: i + 1 for i, u in enumerate(unique_users_str)}
    item_id_map = {m: i + 1 for i, m in enumerate(unique_items_str)}

    # Map to int IDs
    train_df["user_id_int"] = train_df["user_id"].astype(str).map(user_id_map).astype(np.int32)
    train_df["item_id_int"] = train_df["item_id"].astype(str).map(item_id_map).astype(np.int32)
    test_df["user_id_int"]  = test_df["user_id"].astype(str).map(user_id_map).astype(np.int32)
    test_df["item_id_int"]  = test_df["item_id"].astype(str).map(item_id_map).astype(np.int32)

    # Drop any rows that didn't map
    train_df.dropna(subset=["user_id_int","item_id_int"], inplace=True)
    test_df.dropna(subset=["user_id_int","item_id_int"], inplace=True)

    num_users = len(user_id_map)
    num_items = len(item_id_map)
    print(f"Number of mapped users={num_users}, items={num_items}")

    # 3) Randomly pick 20% of users as cold
    rng = np.random.default_rng(42)
    all_user_ids = train_df["user_id_int"].unique()
    cold_count = int(0.20 * len(all_user_ids))
    cold_users = rng.choice(all_user_ids, size=cold_count, replace=False)

    # Partition: cold vs non-cold
    cold_df = train_df[train_df["user_id_int"].isin(cold_users)]
    non_cold_df = train_df[~train_df["user_id_int"].isin(cold_users)]

    # 4) Scenario A: Keep only 1 real interaction for each cold user
    def pick_one(group):
        return group.sample(1, random_state=42)

    cold_single_df = cold_df.groupby("user_id_int").apply(pick_one).reset_index(drop=True)
    scenarioA_df = pd.concat([non_cold_df, cold_single_df], ignore_index=True)
    scenarioA_df["label"] = 1.0
    print(f"Scenario A: {len(scenarioA_df)} rows (each cold user => 1 real)")

    # Prepare dataset + train model
    scenarioA_ds = df_to_tf_dataset({
        "user_id": scenarioA_df["user_id_int"].values,
        "item_id": scenarioA_df["item_id_int"].values
    }, shuffle=True, batch_size=1024)

    user_model_A = UserModel(num_users, embedding_dim=16)
    item_model_A = ItemModel(num_items, embedding_dim=16)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    model_A = TwoTowerModel(user_model_A, item_model_A, loss_fn)
    model_A.compile(optimizer=tf.keras.optimizers.Adam(0.001))

    print("\n--- Training Scenario A (cold=1 real) ...")
    model_A.fit(scenarioA_ds, epochs=5)

    print("\nEvaluating Scenario A (LOO+Negative Sampling)...")
    hrA, ndcgA = evaluate_with_negative_sampling(
        model=model_A,
        test_df=test_df,
        train_df=scenarioA_df,
        num_items=num_items,
        K_values=[1,3,5,10],
        num_neg=99
    )
    print("==== Scenario A Results ====")
    for k in [1,3,5,10]:
        print(f"  HR@{k}={hrA[k]:.4f}, NDCG@{k}={ndcgA[k]:.4f}")
    print("============================")

    # 5) Scenario B: Add augmented interactions for cold users
    # Instead of randomly picking an item, we load "ml-1m_augmented.csv"

    augmented_path = "augmented_data/ml-1m_augmented.csv"  # Adjust path as needed
    augmented_df = pd.read_csv(augmented_path)
    print(f"\nLoaded augmented data with {len(augmented_df)} rows")

    # Suppose augmented CSV has columns 'user_id', 'movie_id'
    # Rename 'movie_id' => 'item_id' if needed
    augmented_df.rename(columns={"movie_id": "item_id"}, inplace=True)

    # If user_id or item_id might have '.0', remove them
    augmented_df["user_id"] = augmented_df["user_id"].astype(str).str.replace(".0", "", regex=False)
    augmented_df["item_id"] = augmented_df["item_id"].astype(str).str.replace(".0", "", regex=False)

    # Map to user_id_int, item_id_int
    augmented_df["user_id_int"] = augmented_df["user_id"].map(user_id_map).astype(np.int32, errors="ignore")
    augmented_df["item_id_int"] = augmented_df["item_id"].map(item_id_map).astype(np.int32, errors="ignore")

    # Drop rows that fail mapping
    augmented_df.dropna(subset=["user_id_int","item_id_int"], inplace=True)

    # Keep only augmented rows for our cold users
    cold_augmented_df = augmented_df[augmented_df["user_id_int"].isin(cold_users)].copy()
    print(f"Augmented rows for cold users: {len(cold_augmented_df)}")

    # Label these augmented interactions
    cold_augmented_df["label"] = 1.0

    # Scenario B => scenarioA + these augmented rows
    scenarioB_df = pd.concat([scenarioA_df, cold_augmented_df], ignore_index=True)
    print(f"Scenario B: {len(scenarioB_df)} total rows (1 real + augmented for cold)")

    scenarioB_ds = df_to_tf_dataset({
        "user_id": scenarioB_df["user_id_int"].values,
        "item_id": scenarioB_df["item_id_int"].values
    }, shuffle=True, batch_size=1024)

    user_model_B = UserModel(num_users, embedding_dim=16)
    item_model_B = ItemModel(num_items, embedding_dim=16)
    model_B = TwoTowerModel(user_model_B, item_model_B, loss_fn)
    model_B.compile(optimizer=tf.keras.optimizers.Adam(0.001))

    print("\n--- Training Scenario B (cold=1 real + augmented) ...")
    model_B.fit(scenarioB_ds, epochs=5)

    print("\nEvaluating Scenario B (LOO+Negative Sampling)...")
    hrB, ndcgB = evaluate_with_negative_sampling(
        model=model_B,
        test_df=test_df,
        train_df=scenarioB_df,
        num_items=num_items,
        K_values=[1,3,5,10],
        num_neg=99
    )
    print("==== Scenario B Results ====")
    for k in [1,3,5,10]:
        print(f"  HR@{k}={hrB[k]:.4f}, NDCG@{k}={ndcgB[k]:.4f}")
    print("============================")

    # 6) Compare Scenario A vs. B
    print("\n============ FINAL SUMMARY ============")
    print("Scenario A: each cold user => 1 real item")
    for k in [1,3,5,10]:
        print(f"  HR@{k}={hrA[k]:.4f}, NDCG@{k}={ndcgA[k]:.4f}")

    print("\nScenario B: each cold user => 1 real + augmented data")
    for k in [1,3,5,10]:
        print(f"  HR@{k}={hrB[k]:.4f}, NDCG@{k}={ndcgB[k]:.4f}")
    print("========================================")
