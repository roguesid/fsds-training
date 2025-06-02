import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


def stratified_split(housing, stratify_col="income_cat"):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing[stratify_col]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    for dataset in (strat_train_set, strat_test_set):
        dataset.drop(stratify_col, axis=1, inplace=True)
    return strat_train_set, strat_test_set


def add_income_cat(housing):
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    return housing


def add_features(housing):
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]
    return housing
