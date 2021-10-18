from src.vae import run_vae_augmentation
from tqdm import tqdm
import random
import pandas as pd

# splits a dataframe into chunks of equal size
def split_dataframe(df, chunk_size):
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i * chunk_size:(i + 1) * chunk_size])
    return chunks

def augment_with_vae(data, times_to_aug):
    return run_vae_augmentation(data, times_to_aug)

def augment_sets_numerical_rand_percentage(dfs, data_name, times_to_aug, percentage):
    print('Start Augmentation by changing values randomly for:', data_name)
    l = [transformations_numerical_values_rand_percentage(dfs, percentage) for _ in tqdm(range(times_to_aug))]
    return pd.concat(l)

def augment_sets_timesets(dfs, data_name, times_to_aug, split_size):
    print('Start Augmentation by mixing timesets for:', data_name)
    l = [transformations_timesets(dfs, split_size) for _ in tqdm(range(times_to_aug))]
    return pd.concat(l)

# uses percentage to change values by a random amount between 0 and the given percentage
def transformations_numerical_values_rand_percentage(df_for_aug, percentage):
    for column in df_for_aug:
        _min = df_for_aug[column].min()
        _max = df_for_aug[column].max()
        for count, val in enumerate(df_for_aug[column]):
            if val == 0:
                continue
            amount = val * percentage
            amount = round(random.uniform(0, amount), 6)
            rand = random.randint(0,1)
            if rand == 0:
                df_for_aug[column][count] = val - amount
                if df_for_aug[column][count] < _min:
                    df_for_aug[column][count] = _min
            else:
                df_for_aug[column][count] = val + amount
                if df_for_aug[column][count] > _max:
                    df_for_aug[column][count] = _max
    return df_for_aug

# splits data into chunks, shuffles these
def transformations_timesets(df_for_aug, split_size):
    res_df = df_for_aug
    if len(df_for_aug) > 0:
        chunk_size = int(len(df_for_aug)/split_size)
        df_chunks = split_dataframe(df_for_aug, chunk_size)
        random.shuffle(df_chunks)
        res_df = pd.concat(df_chunks)
    return res_df