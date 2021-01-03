import pandas as pd

def get_names_from_indexes(indexes):
    df = pd.read_csv('./data/train_filtered.csv')
    names = df.loc[indexes,'name'].values
    return names
