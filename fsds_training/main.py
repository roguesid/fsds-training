from data_ingestion import fetch_housing_data, load_housing_data
from preprocessing import add_income_cat, stratified_split, add_features
from training import prepare_data, train_linear_regression, grid_search_rf
from scoring import evaluate_model


def main():
    fetch_housing_data()
    housing = load_housing_data()
    housing = add_income_cat(housing)
    strat_train_set, strat_test_set = stratified_split(housing)
    strat_train_set = add_features(strat_train_set)
    strat_test_set = add_features(strat_test_set)

    housing_labels = strat_train_set["median_house_value"].copy()
    housing_prepared, imputer = prepare_data(strat_train_set)

    model = train_linear_regression(housing_prepared, housing_labels)
    metrics = evaluate_model(model, housing_prepared, housing_labels)
    print("Linear Regression training metrics:", metrics)


if __name__ == "__main__":
    main()
