import tensorflow as tf
import tensorflow_recommenders as tfrs
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import os
from featurize_1m import featurize_movielens_1m
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
    all_predictions = []
    all_labels = []
    for batch in test_ds:
        user_embeddings = model.user_model(batch)
        item_ids = batch["item_id"]
        all_predictions.append(user_embeddings.numpy())
        all_labels.append(item_ids.numpy())
    all_predictions = np.vstack(all_predictions)
    all_labels = np.concatenate(all_labels)

    all_candidates = []
    all_candidate_ids = []
    for batch in candidate_ds:
        item_embeddings = model.item_model(batch)
        all_candidates.append(item_embeddings.numpy())
        all_candidate_ids.append(batch["item_id"].numpy())
    all_candidates = np.vstack(all_candidates)
    all_candidate_ids = np.concatenate(all_candidate_ids)

    scores = np.dot(all_predictions, all_candidates.T)
    hit_rates = {}
    ndcg_values = {}

    for k in k_values:
        top_k_indices = np.argsort(-scores, axis=1)[:, :k]
        top_k_items = all_candidate_ids[top_k_indices]
        hits = np.zeros(len(all_labels))
        for i, (prediction, label) in enumerate(zip(top_k_items, all_labels)):
            if label in prediction:
                hits[i] = 1
        hit_rate = np.mean(hits)
        hit_rates[k] = hit_rate
        
        ndcg = 0.0
        for i, (prediction, label) in enumerate(zip(top_k_items, all_labels)):
            if label in prediction:
                rank = np.where(prediction == label)[0][0] + 1
                ndcg += 1.0 / np.log2(rank + 1)
        ndcg /= len(all_labels)
        ndcg_values[k] = ndcg

    return hit_rates, ndcg_values

if __name__ == "__main__":
    train_df, test_df, users_df, items_df = featurize_movielens_1m()

    augmented_path = "augmented_data/ml-1m_augmented.csv"
    if os.path.exists(augmented_path):
        augmented_df = pd.read_csv(augmented_path)
        
        augmented_df = augmented_df.rename(columns={'user_id': 'user_id', 'movie_id': 'item_id'})
        augmented_df["user_id"] = augmented_df["user_id"].astype(str).str.replace('.0', '')
        augmented_df["item_id"] = augmented_df["item_id"].astype(str).str.replace('.0', '')
    else:
        augmented_df = pd.DataFrame(columns=['user_id', 'item_id'])

    train_df["user_id"] = train_df["user_id"].astype(str)
    train_df["item_id"] = train_df["item_id"].astype(str)
    test_df["user_id"] = test_df["user_id"].astype(str)
    test_df["item_id"] = test_df["item_id"].astype(str)

    unique_user_ids_str = sorted(pd.concat([train_df["user_id"], test_df["user_id"]]).unique())
    unique_item_ids_str = sorted(pd.concat([train_df["item_id"], test_df["item_id"]]).unique())

    if not augmented_df.empty:
        unique_user_ids_str = sorted(pd.concat([
            pd.Series(unique_user_ids_str), 
            augmented_df["user_id"].astype(str)
        ]).unique())
        
        unique_item_ids_str = sorted(pd.concat([
            pd.Series(unique_item_ids_str), 
            augmented_df["item_id"].astype(str)
        ]).unique())

    user_id_map = {str_id: i + 1 for i, str_id in enumerate(unique_user_ids_str)}
    item_id_map = {str_id: i + 1 for i, str_id in enumerate(unique_item_ids_str)}

    def safe_map(id_str, id_map, default=0):
        return id_map.get(id_str, default)

    train_df["user_id"] = train_df["user_id"].apply(lambda x: safe_map(x, user_id_map))
    train_df["item_id"] = train_df["item_id"].apply(lambda x: safe_map(x, item_id_map))
    test_df["user_id"] = test_df["user_id"].apply(lambda x: safe_map(x, user_id_map))
    test_df["item_id"] = test_df["item_id"].apply(lambda x: safe_map(x, item_id_map))

    train_df["user_id"] = train_df["user_id"].astype(np.int32)
    train_df["item_id"] = train_df["item_id"].astype(np.int32)
    test_df["user_id"] = test_df["user_id"].astype(np.int32)
    test_df["item_id"] = test_df["item_id"].astype(np.int32)
    
    num_users = max(user_id_map.values())
    num_items = max(item_id_map.values())

    invalid_train_items = train_df[(train_df["item_id"] <= 0) | (train_df["item_id"] > num_items)]
    invalid_test_items = test_df[(test_df["item_id"] <= 0) | (test_df["item_id"] > num_items)]
    invalid_train_users = train_df[(train_df["user_id"] <= 0) | (train_df["user_id"] > num_users)]
    invalid_test_users = test_df[(test_df["user_id"] <= 0) | (test_df["user_id"] > num_users)]
    
    if not invalid_train_items.empty:
        train_df = train_df[(train_df["item_id"] > 0) & (train_df["item_id"] <= num_items)]
    
    if not invalid_test_items.empty:
        test_df = test_df[(test_df["item_id"] > 0) & (test_df["item_id"] <= num_items)]
    
    if not invalid_train_users.empty:
        train_df = train_df[(train_df["user_id"] > 0) & (train_df["user_id"] <= num_users)]
    
    if not invalid_test_users.empty:
        test_df = test_df[(test_df["user_id"] > 0) & (test_df["user_id"] <= num_users)]
    
    train_df = train_df.dropna(subset=["user_id", "item_id"])
    test_df = test_df.dropna(subset=["user_id", "item_id"])
    
    train_ds = df_to_tf_dataset(train_df, shuffle=True, batch_size=1024)
    test_ds = df_to_tf_dataset(test_df, shuffle=False, batch_size=1024)
    
    embedding_dim = 32
    user_model = UserModel(num_users, embedding_dim=embedding_dim)
    item_model = ItemModel(num_items, embedding_dim=embedding_dim)
    
    candidates = tf.data.Dataset.from_tensor_slices({
        "item_id": np.arange(1, num_items + 1, dtype=np.int32)
    }).batch(128)
    
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model = TwoTowerModel(user_model=user_model, item_model=item_model, loss=loss)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    
    try:
        model.fit(train_ds, validation_data=test_ds, epochs=10)
    except Exception as e:
        print(f"Error during training: {e}")
    
    try:
        loss_result = model.evaluate(test_ds, return_dict=True)
        print("\nOriginal test evaluation loss:", loss_result)
    
        k_values = [1, 3, 5, 10]
        hit_rates, ndcg_values = compute_hit_rate(model, test_ds, candidates, k_values)
        for k in k_values:
            print(f"Hit Rate (Top-{k}): {hit_rates[k]:.4f}")
            print(f"NDCG (Top-{k}): {ndcg_values[k]:.4f}")
    
        index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
    
        item_ids_tensor = tf.constant(np.arange(1, num_items + 1, dtype=np.int32))
        all_item_embeddings = []
        batch_size = 256
    
        for start in range(0, len(item_ids_tensor), batch_size):
            end = min(start + batch_size, len(item_ids_tensor))
            batch_ids = item_ids_tensor[start:end]
            batch_embeddings = model.item_model({"item_id": batch_ids})
            all_item_embeddings.append(batch_embeddings)
    
        all_item_embeddings = tf.concat(all_item_embeddings, axis=0)
        print("Successfully indexed embeddings!")
    except Exception as e:
        print(f"Error during evaluation: {e}")
