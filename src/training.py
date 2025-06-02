from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import randint
import pandas as pd


def prepare_data(housing, imputer=None):
    housing_num = housing.drop("ocean_proximity", axis=1)
    if not imputer:
        imputer = SimpleImputer(strategy="median")
        imputer.fit(housing_num)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )
    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))
    return housing_prepared, imputer


def train_linear_regression(X, y):
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    return lin_reg


def train_decision_tree(X, y):
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(X, y)
    return tree_reg


def train_random_forest_with_search(X, y):
    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }
    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    rnd_search.fit(X, y)
    return rnd_search


def grid_search_rf(X, y):
    param_grid = [
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]
    forest_reg = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(X, y)
    return grid_search
