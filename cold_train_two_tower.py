import tensorflow as tf
import tensorflow_recommenders as tfrs
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import os
from featurize import featurize_movielens_100k
from preprocess import create_negative_samples

class UserModel(tf.keras.Model):
    def __init__(self, num_users, embedding_dim=32):
        super().__init__()
        self.user_embedding = layers.Embedding(
            input_dim=num_users + 1,
            output_dim=embedding_dim
        )

    def call(self, inputs):
        user_id = inputs["user_id"]
        return self.user_embedding(user_id)

class ItemModel(tf.keras.Model):
    def __init__(self, num_items, embedding_dim=32):
        super().__init__()
        self.item_embedding = layers.Embedding(
            input_dim=num_items + 1,
            output_dim=embedding_dim
        )

    def call(self, inputs):
        item_id = inputs["item_id"]
        return self.item_embedding(item_id)

class TwoTowerModel(tfrs.models.Model):
    def __init__(self, user_model, item_model, loss):
        super().__init__()
        self.user_model = user_model
        self.item_model = item_model
        self.loss = loss
        self.task = tfrs.tasks.Retrieval(loss=loss)

    def compute_loss(self, features, training=False):
        user_embeddings = self.user_model(features)
        item_embeddings = self.item_model(features)
        return self.task(user_embeddings, item_embeddings)

def df_to_tf_dataset(df, shuffle=True, batch_size=1024):
    ds = tf.data.Dataset.from_tensor_slices(dict(df))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    return ds

def compute_hit_rate(model, test_ds, candidate_ds, k_values):
    all_user_embs = []
    all_labels = []
    
    for batch in test_ds:
        user_embs = model.user_model(batch)
        item_ids = batch["item_id"]
        all_user_embs.append(user_embs.numpy())
        all_labels.append(item_ids.numpy())
    all_user_embs = np.vstack(all_user_embs)
    all_labels = np.concatenate(all_labels)

    all_item_embs = []
    all_item_ids = []
    for batch in candidate_ds:
        emb = model.item_model(batch)
        all_item_embs.append(emb.numpy())
        all_item_ids.append(batch["item_id"].numpy())
    all_item_embs = np.vstack(all_item_embs)
    all_item_ids = np.concatenate(all_item_ids)

    scores = np.dot(all_user_embs, all_item_embs.T)

    hit_rates = {}
    ndcg_values = {}

    for k in k_values:
        top_k_indices = np.argsort(-scores, axis=1)[:, :k]  
        top_k_items = all_item_ids[top_k_indices]

        hits = np.zeros(len(all_labels))
        for i, (predicted_items, true_item) in enumerate(zip(top_k_items, all_labels)):
            if true_item in predicted_items:
                hits[i] = 1.0
        hit_rate = np.mean(hits)
        hit_rates[k] = hit_rate
        
        ndcg = 0.0
        for i, (predicted_items, true_item) in enumerate(zip(top_k_items, all_labels)):
            if true_item in predicted_items:
                rank = np.where(predicted_items == true_item)[0][0] + 1
                ndcg += 1.0 / np.log2(rank + 1)
        ndcg /= len(all_labels)
        ndcg_values[k] = ndcg

    return hit_rates, ndcg_values

def train_and_evaluate(train_df, test_df, num_users, num_items, embedding_dim=32, epochs=10):
    train_ds = df_to_tf_dataset({
        "user_id": train_df["user_id_int"].values,
        "item_id": train_df["item_id_int"].values,
    }, shuffle=True, batch_size=1024)

    test_ds = df_to_tf_dataset({
        "user_id": test_df["user_id_int"].values,
        "item_id": test_df["item_id_int"].values,
    }, shuffle=False, batch_size=1024)

    user_model = UserModel(num_users, embedding_dim=embedding_dim)
    item_model = ItemModel(num_items, embedding_dim=embedding_dim)

    candidates_ds = tf.data.Dataset.from_tensor_slices({
        "item_id": np.arange(1, num_items + 1, dtype=np.int32)
    }).batch(128)

    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model = TwoTowerModel(
        user_model=user_model,
        item_model=item_model,
        loss=loss_fn
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    model.fit(train_ds, epochs=epochs, verbose=1)

    hit_rates, ndcg_values = compute_hit_rate(model, test_ds, candidates_ds, k_values=[3,5])
    return hit_rates, ndcg_values

if __name__ == "__main__":
    train_df, test_df, users_df, items_df = featurize_movielens_100k()

    unique_user_ids_str = sorted(train_df["user_id"].astype(str).unique())
    unique_item_ids_str = sorted(train_df["item_id"].astype(str).unique())
    user_id_map = {str_id: i+1 for i, str_id in enumerate(unique_user_ids_str)}
    item_id_map = {str_id: i+1 for i, str_id in enumerate(unique_item_ids_str)}

    train_df["user_id_int"] = train_df["user_id"].astype(str).map(user_id_map)
    train_df["item_id_int"] = train_df["item_id"].astype(str).map(item_id_map)
    test_df["user_id_int"] = test_df["user_id"].astype(str).map(user_id_map)
    test_df["item_id_int"] = test_df["item_id"].astype(str).map(item_id_map)

    train_df.dropna(subset=["user_id_int", "item_id_int"], inplace=True)
    test_df.dropna(subset=["user_id_int", "item_id_int"], inplace=True)

    train_df["user_id_int"] = train_df["user_id_int"].astype(np.int32)
    train_df["item_id_int"] = train_df["item_id_int"].astype(np.int32)
    test_df["user_id_int"] = test_df["user_id_int"].astype(np.int32)
    test_df["item_id_int"] = test_df["item_id_int"].astype(np.int32)

    num_users = len(unique_user_ids_str)
    num_items = len(unique_item_ids_str)

    all_user_ids = train_df["user_id_int"].unique()
    rng = np.random.default_rng(seed=42)
    cold_user_count = int(0.2 * len(all_user_ids))  
    cold_users = rng.choice(all_user_ids, size=cold_user_count, replace=False)

    train_df_cold = []
    non_cold_df = train_df[~train_df["user_id_int"].isin(cold_users)]
    train_df_cold.append(non_cold_df)
    cold_df = train_df[train_df["user_id_int"].isin(cold_users)]
    cold_df_one = cold_df.groupby("user_id_int").apply(lambda x: x.sample(1, random_state=42))
    cold_df_one.reset_index(drop=True, inplace=True)
    train_df_cold.append(cold_df_one)
    train_df_cold = pd.concat(train_df_cold, ignore_index=True)

    print("\n=== Training with cold users (only 1 interaction each) ===")
    hit_rates_1, ndcg_values_1 = train_and_evaluate(
        train_df_cold, test_df, num_users, num_items, embedding_dim=32, epochs=5
    )
    print("Results (Cold Users, 1 Interaction):")
    for k in [3,5]:
        print(f"HR@{k} = {hit_rates_1[k]:.4f}, NDCG@{k} = {ndcg_values_1[k]:.4f}")

    synthetic_rows = []
    user_groups = cold_df.groupby("user_id_int")
    for uid, group in user_groups:
        original_items = group["item_id_int"].unique()
        possible_items = np.setdiff1d(all_user_ids, original_items)
        all_item_ids = np.arange(1, num_items + 1)
        possible_items = np.setdiff1d(all_item_ids, original_items)
        if len(possible_items) > 0:
            new_item = rng.choice(possible_items, size=1)[0]
            synthetic_rows.append({
                "user_id_int": uid,
                "item_id_int": new_item,
                "user_id": str(uid),
                "item_id": str(new_item)
            })

    synthetic_df = pd.DataFrame(synthetic_rows)
    train_df_cold_aug = pd.concat([train_df_cold, synthetic_df], ignore_index=True)

    print("\n=== Training with cold users (1 real + 1 synthetic interaction) ===")
    hit_rates_2, ndcg_values_2 = train_and_evaluate(
        train_df_cold_aug, test_df, num_users, num_items, embedding_dim=32, epochs=5
    )
    print("Results (Cold Users, +1 Synthetic):")
    for k in [3,5]:
        print(f"HR@{k} = {hit_rates_2[k]:.4f}, NDCG@{k} = {ndcg_values_2[k]:.4f}")

    print("\n================== Summary ==================")
    print("Model A (1 Interaction for Cold Users):")
    for k in [3,5]:
        print(f"  HR@{k} = {hit_rates_1[k]:.4f}, NDCG@{k} = {ndcg_values_1[k]:.4f}")
    print("\nModel B (1 Real + 1 Synthetic for Cold Users):")
    for k in [3,5]:
        print(f"  HR@{k} = {hit_rates_2[k]:.4f}, NDCG@{k} = {ndcg_values_2[k]:.4f}")
