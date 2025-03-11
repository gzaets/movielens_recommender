import tensorflow as tf
import tensorflow_recommenders as tfrs
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import os
from featurize_1m import featurize_movielens_1m  # Import the 1M dataset function
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
        ds = ds.shuffle(buffer_size=len(list(df.values())[0]))
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
        
        # Calculate NDCG for all k values
        ndcg = 0.0
        for i, (prediction, label) in enumerate(zip(top_k_items, all_labels)):
            if label in prediction:
                rank = np.where(prediction == label)[0][0] + 1
                ndcg += 1.0 / np.log2(rank + 1)
        ndcg /= len(all_labels)
        ndcg_values[k] = ndcg
    return hit_rates, ndcg_values


if __name__ == "__main__":
    train_df, test_df, users_df, items_df = featurize_movielens_1m(data_path='data/ml-1m')
    
    print(f"Train size: {len(train_df)} Test size: {len(test_df)}")
    
    # Get test users who aren't in train (cold start)
    train_users = set(train_df["user_id"].unique())
    test_users = set(test_df["user_id"].unique())
    cold_users = test_users - train_users
    warm_users = test_users.intersection(train_users)
    
    print(f"Number of cold users in test: {len(cold_users)}")
    print(f"Number of warm users in test: {len(warm_users)}")
    
    # Adjust batch size for the larger dataset
    batch_size = 2048
    
    unique_user_ids_str = sorted(train_df["user_id"].astype(str).unique())
    unique_item_ids_str = sorted(train_df["item_id"].astype(str).unique())
    user_id_map = {str_id: i + 1 for i, str_id in enumerate(unique_user_ids_str)}
    item_id_map = {str_id: i + 1 for i, str_id in enumerate(unique_item_ids_str)}
    
    train_df["user_id_int"] = train_df["user_id"].astype(str).map(user_id_map)
    train_df["item_id_int"] = train_df["item_id"].astype(str).map(item_id_map)
    test_df["user_id_int"] = test_df["user_id"].astype(str).map(user_id_map)
    test_df["item_id_int"] = test_df["item_id"].astype(str).map(item_id_map)
    
    train_df = train_df.dropna(subset=["user_id_int", "item_id_int"])
    test_df = test_df.dropna(subset=["user_id_int", "item_id_int"])
    
    train_df["user_id_int"] = train_df["user_id_int"].astype(np.int32)
    train_df["item_id_int"] = train_df["item_id_int"].astype(np.int32)
    test_df["user_id_int"] = test_df["user_id_int"].astype(np.int32)
    test_df["item_id_int"] = test_df["item_id_int"].astype(np.int32)
    
    # Convert to TensorFlow datasets
    train_ds = df_to_tf_dataset({
        "user_id": train_df["user_id_int"].values,
        "item_id": train_df["item_id_int"].values,
    }, shuffle=True, batch_size=batch_size)
    
    test_ds = df_to_tf_dataset({
        "user_id": test_df["user_id_int"].values,
        "item_id": test_df["item_id_int"].values,
    }, shuffle=False, batch_size=batch_size)
    
    num_users = len(unique_user_ids_str)
    num_items = len(unique_item_ids_str)
    
    print(f"Number of unique users: {num_users}")
    print(f"Number of unique items: {num_items}")
    
    # Increase embedding dimension for the larger dataset
    embedding_dim = 64 
    
    user_model = UserModel(num_users, embedding_dim=embedding_dim)
    item_model = ItemModel(num_items, embedding_dim=embedding_dim)
    
    # Use smaller batches for candidates to avoid memory issues
    candidates = tf.data.Dataset.from_tensor_slices({
        "item_id": np.arange(1, num_items + 1, dtype=np.int32)
    }).batch(64) 
    
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model = TwoTowerModel(
        user_model=user_model,
        item_model=item_model,
        loss=loss
    )
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005))
    
    print("Training the model...")
    # PLAY WITH EPOCHS
    model.fit(train_ds, validation_data=test_ds, epochs=15)
    
    print("Evaluating the model...")
    loss = model.evaluate(test_ds, return_dict=True)
    print("\nEvaluation loss:", loss)
    
    print("\nComputing Hit Rate and NDCG metrics...")
    k_values = [1, 3, 5, 10]
    hit_rates, ndcg_values = compute_hit_rate(model, test_ds, candidates, k_values)
    print("\n----- Hit Rate and NDCG Metrics -----")
    for k in k_values:
        print(f"Hit Rate (Top-{k}): {hit_rates[k]:.4f}")
        print(f"NDCG (Top-{k}): {ndcg_values[k]:.4f}")
    
    # Build index for recommendations
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
    
    print(f"Item IDs tensor shape: {item_ids_tensor.shape}")
    print(f"All item embeddings shape: {all_item_embeddings.shape}")
    
    if len(all_item_embeddings.shape) == 1:
        all_item_embeddings = tf.reshape(all_item_embeddings, [all_item_embeddings.shape[0], embedding_dim])
        print(f"Reshaped all item embeddings to: {all_item_embeddings.shape}")
    elif len(all_item_embeddings.shape) > 2:
        all_item_embeddings = tf.reshape(all_item_embeddings, [all_item_embeddings.shape[0], -1])
        print(f"Flattened all item embeddings to: {all_item_embeddings.shape}")
    
    if item_ids_tensor.shape[0] != all_item_embeddings.shape[0]:
        print("Warning: Dimension mismatch between item_ids_tensor and all_item_embeddings")
        min_dim = min(item_ids_tensor.shape[0], all_item_embeddings.shape[0])
        item_ids_tensor = item_ids_tensor[:min_dim]
        all_item_embeddings = all_item_embeddings[:min_dim]
        print(f"Adjusted dimensions to: {min_dim}")
    
    try:
        index.index_from_dataset(
            tf.data.Dataset.zip((
                tf.data.Dataset.from_tensor_slices(item_ids_tensor),
                tf.data.Dataset.from_tensor_slices(all_item_embeddings)
            )).batch(128)
        )
        print("Successfully indexed embeddings!")
        
        id_to_str_map = {v: k for k, v in item_id_map.items()}
        sample_user_id = user_id_map[unique_user_ids_str[0]]
        sample_user_tensor = tf.constant([sample_user_id], dtype=tf.int32)
        _, top_item_ids = index({"user_id": sample_user_tensor}, k=10)
        top_item_str_ids = [id_to_str_map.get(int(id), "Unknown") for id in top_item_ids.numpy()[0]]
        
        print("\nRecommendations for user", unique_user_ids_str[0])
        print("Top 10 recommended items:", top_item_str_ids)
        
        user_history = train_df[train_df["user_id"] == unique_user_ids_str[0]]["item_id"].tolist()
        print(f"This user has interacted with {len(user_history)} items in the training set")
    except Exception as e:
        print("Error in indexing or recommendation:", e)
        import traceback
        traceback.print_exc()
    
    print("\nID Mapping Examples:")
    for i in range(min(5, len(unique_user_ids_str))):
        print(f"User: Original ID={unique_user_ids_str[i]}, Integer ID={user_id_map[unique_user_ids_str[i]]}")
    for i in range(min(5, len(unique_item_ids_str))):
        print(f"Item: Original ID={unique_item_ids_str[i]}, Integer ID={item_id_map[unique_item_ids_str[i]]}")