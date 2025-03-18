import tensorflow as tf
import tensorflow_recommenders as tfrs
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from featurize_1m import featurize_movielens_1m
from preprocess import create_negative_samples

class UserModel(tf.keras.Model):
    def __init__(self, num_users, embedding_dim=64):
        super().__init__()
        self.user_embedding = layers.Embedding(
            input_dim=num_users + 1,
            output_dim=embedding_dim
        )
    def call(self, inputs):
        return self.user_embedding(inputs["user_id"])

class ItemModel(tf.keras.Model):
    def __init__(self, num_items, embedding_dim=64):
        super().__init__()
        self.item_embedding = layers.Embedding(
            input_dim=num_items + 1,
            output_dim=embedding_dim
        )
    def call(self, inputs):
        return self.item_embedding(inputs["item_id"])

class TwoTowerModel(tfrs.models.Model):
    def __init__(self, user_model, item_model, loss):
        super().__init__()
        self.user_model = user_model
        self.item_model = item_model
        self.loss = loss
        self.task = tfrs.tasks.Retrieval(loss=loss)
    def compute_loss(self, features, training=False):
        return self.task(self.user_model(features), self.item_model(features))

def df_to_tf_dataset(df, shuffle=True, batch_size=2048):
    ds = tf.data.Dataset.from_tensor_slices(dict(df))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    return ds

def compute_hit_rate(model, test_ds, candidate_ds, k_values):
    a, b = [], []
    for batch in test_ds:
        a.append(model.user_model(batch).numpy())
        b.append(batch["item_id"].numpy())
    a = np.vstack(a)
    b = np.concatenate(b)
    c, d = [], []
    for batch in candidate_ds:
        c.append(model.item_model(batch).numpy())
        d.append(batch["item_id"].numpy())
    c = np.vstack(c)
    d = np.concatenate(d)
    s = np.dot(a, c.T)
    hr, ndcg = {}, {}
    for k in k_values:
        idx = np.argsort(-s, axis=1)[:, :k]
        top_k_items = d[idx]
        h = np.zeros(len(b))
        r = 0.0
        for i, (pred, true) in enumerate(zip(top_k_items, b)):
            if true in pred:
                h[i] = 1
                rank = np.where(pred == true)[0][0] + 1
                r += 1.0 / np.log2(rank + 1)
        hr[k] = np.mean(h)
        ndcg[k] = r / len(b)
    return hr, ndcg

def train_and_evaluate(train_df, test_df, num_users, num_items, embedding_dim=64, epochs=5):
    ds_train = df_to_tf_dataset({
        "user_id": train_df["user_id_int"].values,
        "item_id": train_df["item_id_int"].values
    }, shuffle=True)
    ds_test = df_to_tf_dataset({
        "user_id": test_df["user_id_int"].values,
        "item_id": test_df["item_id_int"].values
    }, shuffle=False)
    u = UserModel(num_users, embedding_dim)
    i = ItemModel(num_items, embedding_dim)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    m = TwoTowerModel(u, i, loss_fn)
    m.compile(optimizer=tf.keras.optimizers.Adam(0.0005))
    m.fit(ds_train, epochs=epochs, verbose=1)
    c = tf.data.Dataset.from_tensor_slices({"item_id": np.arange(1, num_items + 1, dtype=np.int32)}).batch(64)
    hr, ndcg = compute_hit_rate(m, ds_test, c, [3,5])
    return hr, ndcg

if __name__ == "__main__":
    train_df, test_df, u_df, it_df = featurize_movielens_1m(data_path="data/ml-1m")
    uids = sorted(train_df["user_id"].astype(str).unique())
    iids = sorted(train_df["item_id"].astype(str).unique())
    umap = {x: i+1 for i, x in enumerate(uids)}
    imap = {x: i+1 for i, x in enumerate(iids)}
    train_df["user_id_int"] = train_df["user_id"].astype(str).map(umap)
    train_df["item_id_int"] = train_df["item_id"].astype(str).map(imap)
    test_df["user_id_int"] = test_df["user_id"].astype(str).map(umap)
    test_df["item_id_int"] = test_df["item_id"].astype(str).map(imap)
    train_df.dropna(subset=["user_id_int","item_id_int"], inplace=True)
    test_df.dropna(subset=["user_id_int","item_id_int"], inplace=True)
    train_df["user_id_int"] = train_df["user_id_int"].astype(np.int32)
    train_df["item_id_int"] = train_df["item_id_int"].astype(np.int32)
    test_df["user_id_int"] = test_df["user_id_int"].astype(np.int32)
    test_df["item_id_int"] = test_df["item_id_int"].astype(np.int32)
    nu = len(uids)
    ni = len(iids)
    all_user_ids = train_df["user_id_int"].unique()
    rng = np.random.default_rng(42)
    cold_count = int(0.2 * len(all_user_ids))
    cold_users = rng.choice(all_user_ids, size=cold_count, replace=False)
    x = []
    y = train_df[~train_df["user_id_int"].isin(cold_users)]
    x.append(y)
    z = train_df[train_df["user_id_int"].isin(cold_users)]
    w = z.groupby("user_id_int").apply(lambda p: p.sample(1, random_state=42))
    w.reset_index(drop=True, inplace=True)
    x.append(w)
    cold_df_1 = pd.concat(x, ignore_index=True)
    hr1, ndcg1 = train_and_evaluate(cold_df_1, test_df, nu, ni, embedding_dim=64, epochs=5)
    s = []
    g = z.groupby("user_id_int")
    for uid, group in g:
        orig = group["item_id_int"].unique()
        all_items = np.arange(1, ni+1)
        pickable = np.setdiff1d(all_items, orig)
        if len(pickable) > 0:
            item_new = rng.choice(pickable, 1)[0]
            s.append({"user_id_int": uid, "item_id_int": item_new, "user_id": str(uid), "item_id": str(item_new)})
    s = pd.DataFrame(s)
    cold_df_2 = pd.concat([cold_df_1, s], ignore_index=True)
    hr2, ndcg2 = train_and_evaluate(cold_df_2, test_df, nu, ni, embedding_dim=64, epochs=5)
    print("Cold Users, 1 Interaction:")
    print(f"HR@3={hr1[3]:.4f} NDCG@3={ndcg1[3]:.4f}  HR@5={hr1[5]:.4f} NDCG@5={ndcg1[5]:.4f}")
    print("Cold Users, +1 Synthetic:")
    print(f"HR@3={hr2[3]:.4f} NDCG@3={ndcg2[3]:.4f}  HR@5={hr2[5]:.4f} NDCG@5={ndcg2[5]:.4f}")
