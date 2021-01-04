import pandas as pd

def get_names_from_indexes(indexes, names_df):
    names = names_df.loc[indexes,'name'].values
    return names
