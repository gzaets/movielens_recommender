import tensorflow as tf
import tensorflow_recommenders as tfrs
from tensorflow.keras import layers
import pandas as pd
import numpy as np

from featurize_1m import featurize_movielens_1m

def df_to_tf_dataset(df, shuffle=True, batch_size=1024):
    ds = tf.data.Dataset.from_tensor_slices(dict(df))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    return ds

class UserModel(tf.keras.Model):
    def __init__(self, num_users, embedding_dim=16):
        super().__init__()
        self.user_embedding = layers.Embedding(
            input_dim=num_users + 1,
            output_dim=embedding_dim
        )

    def call(self, inputs):
        return self.user_embedding(inputs["user_id"])

class ItemModel(tf.keras.Model):
    def __init__(self, max_item_index, embedding_dim=16):
        super().__init__()
        # Use max_item_index (the maximum mapped value) so that input_dim covers all indices.
        self.item_embedding = layers.Embedding(
            input_dim=max_item_index + 1,
            output_dim=embedding_dim
        )

    def call(self, inputs):
        return self.item_embedding(inputs["item_id"])

class TwoTowerModel(tfrs.models.Model):
    def __init__(self, user_model, item_model, loss):
        super().__init__()
        self.user_model = user_model
        self.item_model = item_model
        self.task = tfrs.tasks.Retrieval(loss=loss)

    def compute_loss(self, features, training=False):
        user_embeddings = self.user_model(features)
        item_embeddings = self.item_model(features)
        return self.task(user_embeddings, item_embeddings)

def evaluate_with_negative_sampling(
    model,
    test_df,
    train_df,
    num_users,
    max_item_index,
    K_values=[1, 3, 5, 10],
    num_neg=99,
    seed=42
):
    rng = np.random.default_rng(seed)
    test_user_ids = test_df["user_id_int"].unique()

    user_seen_map = {}
    for row in train_df.itertuples():
        user_seen_map.setdefault(row.user_id_int, set()).add(row.item_id_int)
    for row in test_df.itertuples():
        user_seen_map.setdefault(row.user_id_int, set()).add(row.item_id_int)

    hr_acc = {k: 0.0 for k in K_values}
    ndcg_acc = {k: 0.0 for k in K_values}

    def dcg_at_k(ranked_items, pos_item, k):
        for i, it in enumerate(ranked_items[:k]):
            if it == pos_item:
                return 1.0 / np.log2(i + 2)
        return 0.0

    def idcg_at_k(k):
        return 1.0

    for uid in test_user_ids:
        test_items = test_df[test_df["user_id_int"] == uid]["item_id_int"].unique()
        if len(test_items) == 0:
            continue
        pos_item = test_items[0]

        all_candidates = np.arange(1, max_item_index + 1)
        seen = user_seen_map.get(uid, set())
        valid_negatives = np.setdiff1d(all_candidates, list(seen), assume_unique=True)

        if len(valid_negatives) < num_neg:
            continue

        negs = rng.choice(valid_negatives, size=num_neg, replace=False)
        candidates = np.concatenate(([pos_item], negs))

        user_emb = model.user_model({"user_id": tf.constant([uid], dtype=tf.int32)})
        item_embs = model.item_model({"item_id": tf.constant(candidates, dtype=tf.int32)})
        scores = tf.reduce_sum(user_emb * item_embs, axis=1).numpy()

        ranked_indices = np.argsort(-scores)
        ranked_items = candidates[ranked_indices]

        for k in K_values:
            if pos_item in ranked_items[:k]:
                hr_acc[k] += 1.0
            ndcg_acc[k] += dcg_at_k(ranked_items, pos_item, k) / idcg_at_k(k)

    num_eval_users = len(test_user_ids)
    if num_eval_users > 0:
        for k in K_values:
            hr_acc[k] /= num_eval_users
            ndcg_acc[k] /= num_eval_users

    return hr_acc, ndcg_acc

if __name__ == "__main__":
    # 1) Load data with rating=5 only, 10-core, LOO
    train_df, test_df, users_df, items_df = featurize_movielens_1m()
    print(f"\nFinal Train size: {len(train_df)}, Test size: {len(test_df)}")

    # 2) Build ID maps
    unique_user_ids = sorted(set(train_df["user_id"]) | set(test_df["user_id"]))
    unique_item_ids = sorted(set(train_df["item_id"]) | set(test_df["item_id"]))
    user_id_map = {uid: idx + 1 for idx, uid in enumerate(unique_user_ids)}
    item_id_map = {iid: idx + 1 for idx, iid in enumerate(unique_item_ids, start=1)}

    train_df["user_id_int"] = train_df["user_id"].map(user_id_map).astype(np.int32)
    train_df["item_id_int"] = train_df["item_id"].map(item_id_map).astype(np.int32)
    test_df["user_id_int"]  = test_df["user_id"].map(user_id_map).astype(np.int32)
    test_df["item_id_int"]  = test_df["item_id"].map(item_id_map).astype(np.int32)

    num_users = len(user_id_map)
    # Use the maximum mapped item id as the maximum index for the embedding.
    max_item_index = max(item_id_map.values())
    print("Total # Mapped Users:", num_users, "Total # Mapped Items:", max_item_index)

    # 3) Train dataset
    train_ds = df_to_tf_dataset({
        "user_id": train_df["user_id_int"].values,
        "item_id": train_df["item_id_int"].values,
    }, shuffle=True, batch_size=1024)

    # 4) Model
    user_model = UserModel(num_users)
    item_model = ItemModel(max_item_index)

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    model = TwoTowerModel(user_model=user_model, item_model=item_model, loss=loss)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001))

    print("\nTraining the model (5 epochs)...")
    model.fit(train_ds, epochs=5)

    # 5) Evaluate
    print("\nEvaluating with negative sampling (1 + 99) ...")
    hr, ndcg = evaluate_with_negative_sampling(
        model=model,
        test_df=test_df,
        train_df=train_df,
        num_users=num_users,
        max_item_index=max_item_index,
        K_values=[1, 3, 5, 10],
        num_neg=99,
        seed=42
    )

    # 6) Print results
    print("\nEvaluation Results (rating=5, 10-core, LOO, 1+99):")
    print("=" * 40)
    print(f"{'k':<10}{'HitRate':<15}{'NDCG':<15}")
    print("-" * 40)
    for k in [1, 3, 5, 10]:
        print(f"{k:<10}{hr[k]:.4f}{'':5}{ndcg[k]:.4f}")
    print("=" * 40)
