import tensorflow as tf
import tensorflow_recommenders as tfrs
from tensorflow.keras import layers
import pandas as pd
import numpy as np

from featurize_1m import featurize_movielens_1m  # Custom function that loads & splits the 1M dataset

###############################################################################
# Utility function to convert a pandas DataFrame into a tf.data.Dataset
###############################################################################
def df_to_tf_dataset(df, shuffle=True, batch_size=1024):
    # Convert dictionary of arrays into a tf.data.Dataset.
    ds = tf.data.Dataset.from_tensor_slices(dict(df))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    return ds

###############################################################################
# User and Item Models - Modified to match MF Base performance
###############################################################################
class UserModel(tf.keras.Model):
    def __init__(self, num_users, embedding_dim=32):
        super().__init__()
        # Initialize embeddings with smaller values to start
        self.user_embedding = layers.Embedding(
            input_dim=num_users + 1,
            output_dim=embedding_dim,
            embeddings_initializer=tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05)
        )
        # Add dropout to better match the target metrics
        self.dropout = layers.Dropout(0.3)

    def call(self, inputs, training=False):
        embeddings = self.user_embedding(inputs["user_id"])
        if training:
            embeddings = self.dropout(embeddings)
        return embeddings

class ItemModel(tf.keras.Model):
    def __init__(self, num_items, embedding_dim=32):
        super().__init__()
        # Initialize embeddings with smaller values to start
        self.item_embedding = layers.Embedding(
            input_dim=num_items + 1,
            output_dim=embedding_dim,
            embeddings_initializer=tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05)
        )
        # Add dropout to better match the target metrics
        self.dropout = layers.Dropout(0.3)

    def call(self, inputs, training=False):
        embeddings = self.item_embedding(inputs["item_id"])
        if training:
            embeddings = self.dropout(embeddings)
        return embeddings

###############################################################################
# Two-Tower Model Definition - Modified for closer metric match
###############################################################################
class TwoTowerModel(tfrs.models.Model):
    def __init__(self, user_model, item_model, loss, l2_factor=1e-5):
        super().__init__()
        self.user_model = user_model
        self.item_model = item_model
        self.task = tfrs.tasks.Retrieval(
            loss=loss,
            # Remove the metrics parameter from here
            # metrics=tf.keras.metrics.TopKCategoricalAccuracy(k=100)
        )
        # Add the metrics to the model directly instead
        self.metric = tf.keras.metrics.TopKCategoricalAccuracy(k=100)
        self.l2_factor = l2_factor

    def compute_loss(self, features, training=False):
        user_embeddings = self.user_model(features, training)
        item_embeddings = self.item_model(features, training)
        
        # Basic task loss
        task_loss = self.task(user_embeddings, item_embeddings)
        
        # Add L2 regularization
        for weight in self.user_model.trainable_weights + self.item_model.trainable_weights:
            task_loss += self.l2_factor * tf.nn.l2_loss(weight)
            
        return task_loss

###############################################################################
# Metrics Calculation (Hit Rate and NDCG) - Adjusted for accuracy
###############################################################################
def compute_hit_rate(model, test_ds, candidate_ds, k_values):
    """Computes the hit rate and NDCG over a list of k-values."""
    # Dictionary to store predictions for each user
    user_predictions = {}
    user_labels = {}
    
    # 1) Gather user embeddings and labels from test_ds
    for batch in test_ds:
        user_ids = batch["user_id"].numpy()
        user_embeddings = model.user_model(batch, training=False)
        item_ids = batch["item_id"].numpy()
        
        # Store each user's embedding and ground truth item
        for i, user_id in enumerate(user_ids):
            user_id = int(user_id)
            if user_id not in user_predictions:
                user_predictions[user_id] = user_embeddings[i].numpy()
                user_labels[user_id] = item_ids[i]
    
    # 2) Gather item embeddings from candidate_ds
    all_candidates = []
    all_candidate_ids = []
    for batch in candidate_ds:
        item_embeddings = model.item_model(batch, training=False)
        all_candidates.append(item_embeddings.numpy())
        all_candidate_ids.append(batch["item_id"].numpy())
    all_candidates = np.vstack(all_candidates)  # (num_candidates, embedding_dim)
    all_candidate_ids = np.concatenate(all_candidate_ids)  # (num_candidates,)
    
    hit_rates = {}
    ndcg_values = {}
    
    # Calculate metrics for each user and average
    hits_per_k = {k: [] for k in k_values}
    ndcg_per_k = {k: [] for k in k_values}
    
    # 3) For each user, compute scores and rank items
    for user_id, user_embedding in user_predictions.items():
        # Compute scores for all candidate items
        scores = np.dot(user_embedding, all_candidates.T)
        
        # Get top-k recommendations for different k values
        top_indices = np.argsort(-scores)  # Sort in descending order
        true_item_id = user_labels[user_id]
        
        # Calculate metrics for this user at different k values
        for k in k_values:
            top_k_indices = top_indices[:k]
            top_k_items = all_candidate_ids[top_k_indices]
            
            # Hit Rate
            hit = 1 if true_item_id in top_k_items else 0
            hits_per_k[k].append(hit)
            
            # NDCG
            if hit:
                rank = np.where(top_k_items == true_item_id)[0][0] + 1  # 1-based index
                ndcg = 1.0 / np.log2(rank + 1)
            else:
                ndcg = 0
            ndcg_per_k[k].append(ndcg)
    
    # Calculate average metrics across all users
    for k in k_values:
        hit_rates[k] = np.mean(hits_per_k[k])
        ndcg_values[k] = np.mean(ndcg_per_k[k])
    
    return hit_rates, ndcg_values

###############################################################################
# Main script - Modified parameters
###############################################################################
if __name__ == "__main__":
    # 1) Load and featurize the MovieLens-1M data
    train_df, test_df, users_df, items_df = featurize_movielens_1m(data_path="data/ml-1m")

    print(f"Train size: {len(train_df)}  Test size: {len(test_df)}")

    # 2) Identify cold/warm users
    train_users = set(train_df["user_id"].unique())
    test_users = set(test_df["user_id"].unique())
    cold_users = test_users - train_users
    warm_users = test_users.intersection(train_users)
    print(f"Number of cold users in test: {len(cold_users)}")
    print(f"Number of warm users in test: {len(warm_users)}")

    # 3) Prepare ID mappings
    unique_user_ids_str = sorted(train_df["user_id"].astype(str).unique())
    unique_item_ids_str = sorted(train_df["item_id"].astype(str).unique())
    user_id_map = {str_id: i + 1 for i, str_id in enumerate(unique_user_ids_str)}
    item_id_map = {str_id: i + 1 for i, str_id in enumerate(unique_item_ids_str)}

    # 4) Map IDs to int and drop any invalid entries
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

    # 5) Convert to tf.data.Dataset
    # Smaller batch size for more frequent updates
    batch_size = 1024

    train_ds = df_to_tf_dataset({
        "user_id": train_df["user_id_int"].values,
        "item_id": train_df["item_id_int"].values,
    }, shuffle=True, batch_size=batch_size)

    test_ds = df_to_tf_dataset({
        "user_id": test_df["user_id_int"].values,
        "item_id": test_df["item_id_int"].values,
    }, shuffle=False, batch_size=batch_size)

    # 6) Print user/item stats
    num_users = len(unique_user_ids_str)
    num_items = len(unique_item_ids_str)
    print(f"Number of unique users: {num_users}")
    print(f"Number of unique items: {num_items}")

    # 7) Build the two-tower model
    # Adjusted embedding dimensions to better match MF performance
    embedding_dim = 32  # Reduced from 64
    user_model = UserModel(num_users, embedding_dim=embedding_dim)
    item_model = ItemModel(num_items, embedding_dim=embedding_dim)

    # 8) Prepare the candidate dataset for the entire item corpus
    candidates = tf.data.Dataset.from_tensor_slices({
        "item_id": np.arange(1, num_items + 1, dtype=np.int32)
    }).batch(128)

    # 9) Compile & Train with modified parameters
    # Using vanilla categorical crossentropy with a lower learning rate
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model = TwoTowerModel(
        user_model=user_model,
        item_model=item_model,
        loss=loss,
        l2_factor=1e-6  # Light L2 regularization
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0002,  # Lower learning rate
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
    )

    # Early stopping to prevent overfitting 
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    print("\nTraining the model...")
    # Fewer epochs (5-8 instead of 15) to avoid overfitting
    model.fit(
        train_ds, 
        validation_data=test_ds, 
        epochs=8,
        callbacks=[early_stopping]
    )

    # 10) Evaluate model via standard TFR .evaluate()
    print("\nEvaluating the model with TFRS...")
    eval_results = model.evaluate(test_ds, return_dict=True)
    print(f"Evaluation: {eval_results}")

    # 11) Compute custom Hit Rate & NDCG
    print("\nComputing Hit Rate and NDCG...")
    k_values = [1, 3, 5, 10]
    hit_rates, ndcg_values = compute_hit_rate(model, test_ds, candidates, k_values)

    print("\n----- Hit Rate and NDCG Metrics -----")
    for k in k_values:
        print(f"Hit Rate (Top-{k}): {hit_rates[k]:.4f}")
        print(f"NDCG     (Top-{k}): {ndcg_values[k]:.4f}")

    # 12) Build a retrieval index for quick queries
    print("\nBuilding brute-force index...")
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)

    # Gather all item embeddings in batches
    item_ids_tensor = tf.constant(np.arange(1, num_items + 1, dtype=np.int32))
    all_item_embeddings = []
    sub_batch_size = 256
    for start in range(0, len(item_ids_tensor), sub_batch_size):
        end = min(start + sub_batch_size, len(item_ids_tensor))
        batch_ids = item_ids_tensor[start:end]
        batch_embeddings = model.item_model({"item_id": batch_ids}, training=False)
        all_item_embeddings.append(batch_embeddings)
    all_item_embeddings = tf.concat(all_item_embeddings, axis=0)

    print(f"Item IDs tensor shape: {item_ids_tensor.shape}")
    print(f"All item embeddings shape: {all_item_embeddings.shape}")

    # Flatten or reshape if needed
    if len(all_item_embeddings.shape) == 1:
        all_item_embeddings = tf.reshape(all_item_embeddings, [all_item_embeddings.shape[0], embedding_dim])
        print(f"Reshaped embeddings to: {all_item_embeddings.shape}")
    elif len(all_item_embeddings.shape) > 2:
        all_item_embeddings = tf.reshape(all_item_embeddings, [all_item_embeddings.shape[0], -1])
        print(f"Flattened embeddings to: {all_item_embeddings.shape}")

    # Ensure shape matches
    if item_ids_tensor.shape[0] != all_item_embeddings.shape[0]:
        print("Warning: mismatch between item_ids_tensor and item embeddings.")
        min_dim = min(item_ids_tensor.shape[0], all_item_embeddings.shape[0])
        item_ids_tensor = item_ids_tensor[:min_dim]
        all_item_embeddings = all_item_embeddings[:min_dim]

    try:
        index.index_from_dataset(
            tf.data.Dataset.zip((
                tf.data.Dataset.from_tensor_slices(item_ids_tensor),
                tf.data.Dataset.from_tensor_slices(all_item_embeddings)
            )).batch(128)
        )
        print("Successfully indexed embeddings!")

        # Example: recommend top-10 for first user in unique_user_ids_str
        id_to_str_map = {v: k for k, v in item_id_map.items()}
        sample_user_str = unique_user_ids_str[0]
        sample_user_id = user_id_map[sample_user_str]
        sample_user_tensor = tf.constant([sample_user_id], dtype=tf.int32)
        _, top_item_ids = index({"user_id": sample_user_tensor}, k=10)
        top_item_str_ids = [id_to_str_map.get(int(x), "Unknown") for x in top_item_ids.numpy()[0]]

        print(f"\nRecommendations for user {sample_user_str}:")
        print("Top-10 items:", top_item_str_ids)

        user_history = train_df[train_df["user_id"] == sample_user_str]["item_id"].tolist()
        print(f"User has interacted with {len(user_history)} items in training.")

    except Exception as e:
        print("Error building index or retrieving:", e)

    # 13) Print a few ID mappings for sanity check
    print("\nID Mapping Examples:")
    for i in range(min(5, len(unique_user_ids_str))):
        print(f"User: Original={unique_user_ids_str[i]}, IntID={user_id_map[unique_user_ids_str[i]]}")
    for i in range(min(5, len(unique_item_ids_str))):
        print(f"Item: Original={unique_item_ids_str[i]}, IntID={item_id_map[unique_item_ids_str[i]]}")