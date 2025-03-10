import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# Ratings data.
ratings = tfds.load("movielens/100k-ratings", split="train")
# Features of all the available movies.
# movies = tfds.load("movielens/100k-movies", split="train")
for x in ratings.take(1).as_numpy_iterator():
  pprint.pprint(x)
# for x in movies.take(1).as_numpy_iterator():
#   pprint.pprint(x)


# Extract user ID, item ID, and timestamp
user_item_timestamp = []
for example in tfds.as_numpy(ratings):
    # print(type(example['user_id']))
    user_id = int(example['user_id'].decode('utf-8'))
    item_id = int(example['movie_id'].decode('utf-8'))
    timestamp = example['timestamp']
    user_item_timestamp.append((user_id, item_id, timestamp))
# Sort by user ID, then by timestamp
sorted_data = sorted(user_item_timestamp, key=lambda x: (x[0], x[2]))
f_name = 'ml-100k.txt'
with open(f_name, 'w') as f:
   for user_id, item_id, _ in sorted_data:
        f.write(f"{user_id} {item_id}\n")

print(f"Data has been written to {f_name}")
  