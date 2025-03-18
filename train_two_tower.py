import tensorflow as tf
import tensorflow_recommenders as tfrs
from tensorflow.keras import layers
import pandas as pd
import numpy as np

# Dummy imports for illustration (replace with actual data processing functions)
from featurize import featurize_movielens_100k

def df_to_tf_dataset(df, shuffle=True, batch_size=1024):
    ds = tf.data.Dataset.from_tensor_slices(dict(df))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    return ds

class UserModel(tf.keras.Model):
    def __init__(self, num_users, embedding_dim=32):
        super().__init__()
        self.user_embedding = layers.Embedding(input_dim=num_users + 1, output_dim=embedding_dim)

    def call(self, inputs):
        return self.user_embedding(inputs["user_id"])

class ItemModel(tf.keras.Model):
    def __init__(self, num_items, embedding_dim=32):
        super().__init__()
        self.item_embedding = layers.Embedding(input_dim=num_items + 1, output_dim=embedding_dim)

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

def calculate_dcg_at_k(relevant_items, recommended_items, k):
    """Calculate DCG@k for a single user."""
    dcg = 0.0
    for i in range(min(k, len(recommended_items))):
        # If the item at position i is relevant
        if recommended_items[i] in relevant_items:
            # DCG formula: rel_i / log_2(i+2), where position is 0-indexed
            dcg += 1.0 / np.log2(i + 2)
    return dcg

def calculate_idcg_at_k(n_relevant_items, k):
    """Calculate ideal DCG@k when all relevant items are ranked at the top."""
    idcg = 0.0
    for i in range(min(k, n_relevant_items)):
        idcg += 1.0 / np.log2(i + 2)
    return idcg

def calculate_ndcg_at_k(relevant_items, recommended_items, k):
    """Calculate NDCG@k for a single user."""
    if not relevant_items:
        return 0.0
    
    dcg = calculate_dcg_at_k(relevant_items, recommended_items, k)
    idcg = calculate_idcg_at_k(len(relevant_items), k)
    
    if idcg == 0:
        return 0.0
    return dcg / idcg

if __name__ == "__main__":
    train_df, test_df, users_df, items_df = featurize_movielens_100k()

    # Print some info about the data
    print(f"Train size: {len(train_df)} Test size: {len(test_df)}")
    
    # Get unique IDs from both train and test to ensure all IDs have mappings
    all_user_ids = sorted(pd.concat([train_df["user_id"], test_df["user_id"]]).astype(str).unique())
    all_item_ids = sorted(pd.concat([train_df["item_id"], test_df["item_id"]]).astype(str).unique())
    
    user_id_map = {str_id: i + 1 for i, str_id in enumerate(all_user_ids)}
    item_id_map = {str_id: i + 1 for i, str_id in enumerate(all_item_ids)}

    # Count overlap between train and test sets
    train_users = set(train_df["user_id"].astype(str))
    test_users = set(test_df["user_id"].astype(str))
    cold_users = test_users - train_users
    print(f"Number of cold users in test: {len(cold_users)}")
    print(f"Number of warm users in test: {len(test_users) - len(cold_users)}")

    # Convert IDs to integers safely
    train_df["user_id_int"] = train_df["user_id"].astype(str).map(user_id_map).astype(np.int32)
    train_df["item_id_int"] = train_df["item_id"].astype(str).map(item_id_map).astype(np.int32)
    test_df["user_id_int"] = test_df["user_id"].astype(str).map(user_id_map).astype(np.int32)
    test_df["item_id_int"] = test_df["item_id"].astype(str).map(item_id_map).astype(np.int32)

    # Create datasets
    train_df["label"] = 1.0
    train_ds = df_to_tf_dataset({
        "user_id": train_df["user_id_int"].values,
        "item_id": train_df["item_id_int"].values,
    }, shuffle=True, batch_size=1024)
    test_ds = df_to_tf_dataset({
        "user_id": test_df["user_id_int"].values,
        "item_id": test_df["item_id_int"].values,
    }, shuffle=False, batch_size=1024)

    # Create and train model
    num_users = len(all_user_ids)
    num_items = len(all_item_ids)
    embedding_dim = 32

    user_model = UserModel(num_users, embedding_dim)
    item_model = ItemModel(num_items, embedding_dim)

    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model = TwoTowerModel(user_model=user_model, item_model=item_model, loss=loss)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001))

    print("Training the model...")
    model.fit(train_ds, validation_data=test_ds, epochs=10)

    print("\nBuilding retrieval index for evaluation...")
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
    item_dataset = tf.data.Dataset.from_tensor_slices({"item_id": tf.range(1, num_items + 1, dtype=tf.int32)}).batch(256)
    index.index_from_dataset(tf.data.Dataset.zip((
        item_dataset.map(lambda x: x["item_id"]),
        item_dataset.map(lambda x: model.item_model({"item_id": x["item_id"]}))
    )))

    print("\nEvaluating the model...")
    # Only evaluate on warm users (those in the training set)
    warm_user_ids = list(set(test_df["user_id_int"]).intersection(set(train_df["user_id_int"])))
    
    # List of k values to evaluate
    k_values = [1, 3, 5, 10]
    max_k = max(k_values)
    
    # Dictionaries to store metrics
    hit_rates = {k: 0 for k in k_values}
    ndcgs = {k: 0 for k in k_values}
    
    def get_true_items_for_user(user_id, df):
        return set(df[df["user_id_int"] == user_id]["item_id_int"])

    # Evaluate for each user
    for u in warm_user_ids:
        _, top_items = index({"user_id": tf.constant([u])}, k=max_k)
        top_items = top_items[0].numpy()
        true_items = get_true_items_for_user(u, test_df)
        
        # Calculate metrics for each k value
        for k in k_values:
            # Hit Rate calculation
            if any(item in true_items for item in top_items[:k]):
                hit_rates[k] += 1
            
            # NDCG calculation
            ndcgs[k] += calculate_ndcg_at_k(true_items, top_items, k)
    
    # Normalize metrics by number of users
    num_users = len(warm_user_ids)
    if num_users > 0:
        for k in k_values:
            hit_rates[k] /= num_users
            ndcgs[k] /= num_users
    
    # Display results
    print("\nEvaluation Results:")
    print("=" * 40)
    print(f"{'k':<10}{'Hit Rate':<15}{'NDCG':<15}")
    print("-" * 40)
    for k in k_values:
        print(f"{k:<10}{hit_rates[k]:.4f}{'':5}{ndcgs[k]:.4f}")
    print("=" * 40)