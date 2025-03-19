import tensorflow as tf
import tensorflow_recommenders as tfrs
from tensorflow.keras import layers
import pandas as pd
import numpy as np

# Your existing function for loading MovieLens-100k:
from featurize import featurize_movielens_100k

#######################################
# Convert DataFrame -> tf.data.Dataset
#######################################
def df_to_tf_dataset(df, shuffle=True, batch_size=1024):
    ds = tf.data.Dataset.from_tensor_slices(dict(df))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    return ds

#######################################
# User/Item Tower and Two-Tower Model
#######################################
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

#######################################
# Evaluation: Negative Sampling
#######################################
def evaluate_with_negative_sampling(model, test_df, train_df, num_items,
                                    K_values=[1, 3, 5, 10],
                                    num_neg=99,
                                    seed=42):
    """
    Evaluates Hit Rate (HR@K) and NDCG@K using 'Leave-One-Out + Negative Sampling'.
    Each user in test has 1 positive item + `num_neg` negative items.
    """
    rng = np.random.default_rng(seed)

    users_in_test = list(set(test_df["user_id_int"]))

    # Collect the set of all items each user has seen in training or test
    user_seen_items = {}
    for df_ in [train_df, test_df]:
        for row in df_.itertuples():
            user_seen_items.setdefault(row.user_id_int, set()).add(row.item_id_int)

    hr_acc = {k: 0.0 for k in K_values}
    ndcg_acc = {k: 0.0 for k in K_values}

    def dcg_at_k(ranked_items, rel_item, k):
        """Compute DCG if rel_item is within top k of ranked_items."""
        for i, it in enumerate(ranked_items[:k]):
            if it == rel_item:
                # rank is i+1 => DCG = 1/log2(rank+1)
                return 1.0 / np.log2(i + 2)
        return 0.0

    def idcg_at_k(k):
        """Because there's exactly 1 positive, ideal DCG=1.0 for k>=1."""
        return 1.0

    for user_id in users_in_test:
        # Typically each user has 1 "left-out" test item
        test_items = test_df[test_df["user_id_int"] == user_id]["item_id_int"].unique()
        if len(test_items) == 0:
            continue
        positive_item = test_items[0]

        # Gather negatives that user has never interacted with
        seen = user_seen_items.get(user_id, set())
        all_candidates = np.arange(1, num_items + 1)
        valid_negatives = np.setdiff1d(all_candidates, list(seen), assume_unique=True)
        if len(valid_negatives) < num_neg:
            # Not enough unique negatives => skip
            continue
        negs = rng.choice(valid_negatives, size=num_neg, replace=False)
        candidates = np.concatenate(([positive_item], negs))

        # Get model scores for these candidates
        user_emb = model.user_model({"user_id": tf.constant([user_id], dtype=tf.int32)})
        item_embs = model.item_model({"item_id": tf.constant(candidates, dtype=tf.int32)})
        scores = tf.reduce_sum(user_emb * item_embs, axis=1).numpy()

        # Sort in descending order
        ranked_items = candidates[np.argsort(-scores)]

        for k in K_values:
            # HR@K
            if positive_item in ranked_items[:k]:
                hr_acc[k] += 1.0
            # NDCG@K
            ndcg_acc[k] += dcg_at_k(ranked_items, positive_item, k) / idcg_at_k(k)

    num_eval_users = len(users_in_test)
    if num_eval_users > 0:
        for k in K_values:
            hr_acc[k] /= num_eval_users
            ndcg_acc[k] /= num_eval_users

    return hr_acc, ndcg_acc

#######################################
# MAIN SCRIPT
#######################################
if __name__ == "__main__":

    ########################################
    # (1) Load MovieLens-100k data
    ########################################
    train_df, test_df, users_df, items_df = featurize_movielens_100k()
    print(f"\nOriginal train size: {len(train_df)}, test size: {len(test_df)}")

    # Build ID mappings
    unique_user_ids_str = sorted(set(train_df["user_id"].astype(str)) | set(test_df["user_id"].astype(str)))
    unique_item_ids_str = sorted(set(train_df["item_id"].astype(str)) | set(test_df["item_id"].astype(str)))
    user_id_map = {uid: idx + 1 for idx, uid in enumerate(unique_user_ids_str)}
    item_id_map = {iid: idx + 1 for idx, iid in enumerate(unique_item_ids_str)}

    # Map user_id, item_id => user_id_int, item_id_int
    train_df["user_id_int"] = train_df["user_id"].astype(str).map(user_id_map).astype(np.int32)
    train_df["item_id_int"] = train_df["item_id"].astype(str).map(item_id_map).astype(np.int32)
    test_df["user_id_int"]  = test_df["user_id"].astype(str).map(user_id_map).astype(np.int32)
    test_df["item_id_int"]  = test_df["item_id"].astype(str).map(item_id_map).astype(np.int32)

    # Drop any rows that couldn't map
    train_df.dropna(subset=["user_id_int","item_id_int"], inplace=True)
    test_df.dropna(subset=["user_id_int","item_id_int"], inplace=True)

    num_users = len(user_id_map)
    num_items = len(item_id_map)
    print(f"Unique users mapped: {num_users}, unique items mapped: {num_items}")

    ########################################
    # (2) Define cold users (20%)
    ########################################
    rng = np.random.default_rng(42)
    all_user_ids = train_df["user_id_int"].unique()
    cold_count = int(0.20 * len(all_user_ids))
    cold_users = rng.choice(all_user_ids, size=cold_count, replace=False)

    # Partition train data into cold vs. non-cold
    cold_df = train_df[train_df["user_id_int"].isin(cold_users)]
    non_cold_df = train_df[~train_df["user_id_int"].isin(cold_users)]

    ########################################
    # (3) Scenario A: Each cold user => 1 real item
    ########################################
    def pick_single_interaction(group):
        return group.sample(1, random_state=42)

    cold_df_single = cold_df.groupby("user_id_int").apply(pick_single_interaction)
    cold_df_single.reset_index(drop=True, inplace=True)

    scenarioA_train_df = pd.concat([non_cold_df, cold_df_single], ignore_index=True)
    print(f"\nScenario A: {len(scenarioA_train_df)} training rows (each cold user has 1 real interaction)")

    scenarioA_train_df["label"] = 1.0
    scenarioA_ds = df_to_tf_dataset({
        "user_id": scenarioA_train_df["user_id_int"].values,
        "item_id": scenarioA_train_df["item_id_int"].values,
    }, shuffle=True, batch_size=1024)

    # Build & train model for Scenario A
    user_model_A = UserModel(num_users, embedding_dim=16)
    item_model_A = ItemModel(num_items, embedding_dim=16)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    model_A = TwoTowerModel(user_model_A, item_model_A, loss_fn)
    model_A.compile(optimizer=tf.keras.optimizers.Adam(0.001))

    print("\n--- Training Scenario A model (5 epochs) ---")
    model_A.fit(scenarioA_ds, epochs=5)

    print("\nEvaluating Scenario A model (Negative Sampling LOO)...")
    hrA, ndcgA = evaluate_with_negative_sampling(
        model_A,
        test_df,
        scenarioA_train_df,
        num_items,
        K_values=[1, 3, 5, 10],
        num_neg=99
    )
    print("==== Results: Scenario A (Cold = 1 Real) ====")
    for k in [1, 3, 5, 10]:
        print(f" HR@{k}={hrA[k]:.4f}, NDCG@{k}={ndcgA[k]:.4f}")
    print("=============================================")

    ########################################
    # (4) Scenario B: Add augmented interactions
    ########################################
    # In practice, you'd have some CSV of "augmented" user->item rows.
    # e.g. "289G/augmented_data/ml-100k_augmented.csv"
    # For demonstration, let's show how to load & integrate them.

    augmented_path = "augmented_data/ml-100k_augmented.csv"  # or your actual path
    augmented_df = pd.read_csv(augmented_path)
    print(f"\nLoaded augmented dataset with {len(augmented_df)} rows.")

    # Suppose the CSV has columns: "user_id", "movie_id".
    # Rename "movie_id" to "item_id" for consistency:
    augmented_df.rename(columns={"movie_id": "item_id"}, inplace=True)

    # If user_id or item_id might have ".0" decimals, remove them:
    augmented_df["user_id"] = augmented_df["user_id"].astype(str).str.replace(".0", "", regex=False)
    augmented_df["item_id"] = augmented_df["item_id"].astype(str).str.replace(".0", "", regex=False)

    # Map them to our existing user_id_int, item_id_int
    augmented_df["user_id_int"] = augmented_df["user_id"].astype(str).map(user_id_map).astype(np.int32, errors="ignore")
    augmented_df["item_id_int"] = augmented_df["item_id"].astype(str).map(item_id_map).astype(np.int32, errors="ignore")

    # Drop rows that didn't map
    augmented_df.dropna(subset=["user_id_int","item_id_int"], inplace=True)

    # Keep only augmented rows belonging to cold users
    # so each cold user gets 1+ augmented interactions
    augmented_cold_df = augmented_df[augmented_df["user_id_int"].isin(cold_users)].copy()
    print(f"Augmented rows for cold users: {len(augmented_cold_df)}")

    # We'll set label=1.0 for these synthetic interactions
    augmented_cold_df["label"] = 1.0

    # Create "Scenario B" train set: scenarioA + these augmented interactions
    scenarioB_train_df = pd.concat([scenarioA_train_df, augmented_cold_df], ignore_index=True)
    print(f"Scenario B: {len(scenarioB_train_df)} rows (cold has 1 real + augmented items)")

    # Build dataset & re-train
    scenarioB_ds = df_to_tf_dataset({
        "user_id": scenarioB_train_df["user_id_int"].values,
        "item_id": scenarioB_train_df["item_id_int"].values,
    }, shuffle=True, batch_size=1024)

    user_model_B = UserModel(num_users, embedding_dim=16)
    item_model_B = ItemModel(num_items, embedding_dim=16)
    model_B = TwoTowerModel(user_model_B, item_model_B, loss_fn)
    model_B.compile(optimizer=tf.keras.optimizers.Adam(0.001))

    print("\n--- Training Scenario B model (5 epochs) ---")
    model_B.fit(scenarioB_ds, epochs=5)

    # Evaluate Scenario B
    print("\nEvaluating Scenario B model (1 real + augmented) ...")
    hrB, ndcgB = evaluate_with_negative_sampling(
        model_B,
        test_df,
        scenarioB_train_df,
        num_items,
        K_values=[1, 3, 5, 10],
        num_neg=99
    )
    print("==== Results: Scenario B (Cold = 1 Real + Augmented) ====")
    for k in [1, 3, 5, 10]:
        print(f" HR@{k}={hrB[k]:.4f}, NDCG@{k}={ndcgB[k]:.4f}")
    print("=========================================================")

    ########################################
    # Compare A vs B
    ########################################
    print("\n=============== FINAL SUMMARY ===============")
    print("Scenario A: cold = 1 real interaction")
    for k in [1, 3, 5, 10]:
        print(f"   HR@{k}={hrA[k]:.4f}, NDCG@{k}={ndcgA[k]:.4f}")
    print("\nScenario B: cold = 1 real + augmented rows")
    for k in [1, 3, 5, 10]:
        print(f"   HR@{k}={hrB[k]:.4f}, NDCG@{k}={ndcgB[k]:.4f}")
    print("=============================================")
