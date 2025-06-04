import argparse
import logging
import os
import pickle

import pandas as pd

from fsds_training import preprocessing, training


def setup_logging(log_level, log_path=None, no_console_log=False):
    logger = logging.getLogger()
    logger.setLevel(log_level)

    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if not no_console_log:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logging.info("Console logging enabled.")

    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logging.info(f"Logging to file: {log_path}")

    logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train machine learning models.")
    parser.add_argument(
        "--input_dataset_path",
        type=str,
        default="datasets/housing_train.csv",
        help="Path to the input training dataset CSV file.",
    )
    parser.add_argument(
        "--output_model_dir",
        type=str,
        default="models",
        help="Path to the directory where trained models (pickles) will be saved.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (e.g., INFO, DEBUG).",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Path to a file to write logs to. If not specified, logs are not written to a file.",
    )
    parser.add_argument(
        "--no-console-log",
        action="store_true",
        help="Do not write logs to the console.",
    )
    args = parser.parse_args()

    setup_logging(args.log_level.upper(), args.log_path, args.no_console_log)
    logger = logging.getLogger(__name__)

    output_model_dir = args.output_model_dir
    os.makedirs(output_model_dir, exist_ok=True)
    logger.info(f"Ensured output model directory '{output_model_dir}' exists.")

    input_dataset_path = args.input_dataset_path
    logger.info(f"Loading training dataset from {input_dataset_path}...")
    try:
        housing = pd.read_csv(input_dataset_path)
        logger.info("Training dataset loaded.")
    except FileNotFoundError:
        logger.error(f"Error: Training dataset file not found at {input_dataset_path}.")
        return

    logger.info("Adding features to the dataset...")
    housing_with_features = preprocessing.add_features(housing.copy())
    logger.info("Features added.")

    housing_labels = housing_with_features["median_house_value"].copy()
    housing_features = housing_with_features.drop("median_house_value", axis=1)

    logger.info("Preparing data (imputation and one-hot encoding)...")
    housing_prepared, imputer = training.prepare_data(housing_features)
    logger.info("Data preparation complete.")

    logger.info("Training Linear Regression model...")
    lin_reg_model = training.train_linear_regression(housing_prepared, housing_labels)
    lin_reg_model_path = os.path.join(output_model_dir, "linear_regression_model.pkl")
    with open(lin_reg_model_path, "wb") as f:
        pickle.dump(lin_reg_model, f)
    logger.info(f"Linear Regression model saved to {lin_reg_model_path}")

    logger.info(
        "Training Random Forest model with Grid Search (this may take a while)..."
    )
    rf_grid_search_model = training.grid_search_rf(housing_prepared, housing_labels)
    rf_model_path = os.path.join(output_model_dir, "random_forest_model.pkl")
    with open(rf_model_path, "wb") as f:
        pickle.dump(rf_grid_search_model.best_estimator_, f)
    logger.info(f"Random Forest model saved to {rf_model_path}")
    logger.info(f"Best Random Forest parameters: {rf_grid_search_model.best_params_}")
    logger.info(
        f"Best Random Forest RMSE: {(-rf_grid_search_model.best_score_)**0.5:.4f}"
    )

    imputer_path = os.path.join(output_model_dir, "imputer.pkl")
    with open(imputer_path, "wb") as f:
        pickle.dump(imputer, f)
    logger.info(f"Imputer saved to {imputer_path}")

    logger.info("Model training complete.")


if __name__ == "__main__":
    main()
