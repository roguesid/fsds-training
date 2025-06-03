import argparse
import logging
import os
import pickle
import pandas as pd
import mlflow # Import mlflow
import mlflow.sklearn # Import mlflow.sklearn for logging models

from fsds_training import preprocessing, training

# Initialize a logger for this module
logger = logging.getLogger(__name__)

def setup_logging(log_level, log_path=None, no_console_log=False):
    """
    Sets up logging for the script, allowing for console and file logging.

    Args:
        log_level (str): The desired logging level (e.g., "INFO", "DEBUG").
        log_path (str, optional): Path to a file to write logs to. Defaults to None.
        no_console_log (bool, optional): If True, disables console logging. Defaults to False.
    """
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to prevent duplicate logs
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if not no_console_log:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def run_training(input_dataset_path: str, output_model_dir: str, log_level: str = "INFO",
                 log_path: str = None, no_console_log: bool = False,
                 timestamp: str = "", parent_run_id: str = ""):
    """
    Performs model training, including data preparation,
    training Linear Regression and Random Forest models, and saving them.
    MLflow tracking is integrated to log parameters, metrics, and models.

    Args:
        input_dataset_path (str): Path to the input training dataset CSV file.
        output_model_dir (str): Path to the directory where trained models will be saved.
        log_level (str, optional): The desired logging level. Defaults to "INFO".
        log_path (str, optional): Path to a file to write logs to. Defaults to None.
        no_console_log (bool, optional): If True, disables console logging. Defaults to False.
        timestamp (str, optional): Timestamp for unique MLflow run names. Defaults to "".
        parent_run_id (str, optional): ID of the parent MLflow run for naming. Defaults to "".
    """
    run_name = f"model_training_parent_{parent_run_id}_{timestamp}" if parent_run_id and timestamp else (f"model_training_{timestamp}" if timestamp else "model_training")

    # Start an MLflow run, nested=True makes it a child run
    with mlflow.start_run(run_name=run_name, nested=True):
        setup_logging(log_level.upper(), log_path, no_console_log)

        os.makedirs(output_model_dir, exist_ok=True)
        logger.info(f"Ensured output model directory '{output_model_dir}' exists.")

        logger.info(f"Loading training dataset from {input_dataset_path}...")
        try:
            housing = pd.read_csv(input_dataset_path)
        except FileNotFoundError:
            logger.error(f"Error: Training dataset file not found at {input_dataset_path}.")
            return

        # Log parameters to MLflow
        mlflow.log_param("input_dataset_path", input_dataset_path)
        mlflow.log_param("output_model_directory", output_model_dir)
        mlflow.log_param("log_level", log_level)
        mlflow.log_param("log_path", log_path if log_path else "None")
        mlflow.log_param("no_console_log", no_console_log)
        mlflow.log_param("parent_run_id", parent_run_id)

        logger.info("Adding features to the dataset...")
        housing_with_features = preprocessing.add_features(housing.copy())

        housing_labels = housing_with_features["median_house_value"].copy()
        housing_features = housing_with_features.drop("median_house_value", axis=1)

        logger.info("Preparing data (imputation and one-hot encoding)...")
        housing_prepared, imputer = training.prepare_data(housing_features)

        logger.info("Training Linear Regression model...")
        lin_reg_model = training.train_linear_regression(housing_prepared, housing_labels)
        lin_reg_model_path = os.path.join(output_model_dir, "linear_regression_model.pkl")
        with open(lin_reg_model_path, "wb") as f:
            pickle.dump(lin_reg_model, f)
        logger.info(f"Linear Regression model saved to {lin_reg_model_path}")

        # Log Linear Regression model to MLflow
        mlflow.sklearn.log_model(lin_reg_model, "linear_regression_model")

        logger.info(
            "Training Random Forest model with Grid Search (this may take a while)..."
        )
        rf_grid_search_model = training.grid_search_rf(housing_prepared, housing_labels)
        rf_model_path = os.path.join(output_model_dir, "random_forest_model.pkl")
        with open(rf_model_path, "wb") as f:
            pickle.dump(rf_grid_search_model.best_estimator_, f)
        logger.info(f"Random Forest model saved to {rf_model_path}")

        # Log Random Forest best parameters and RMSE to MLflow
        mlflow.log_params(rf_grid_search_model.best_params_)
        mlflow.log_metric("rf_best_rmse", (-rf_grid_search_model.best_score_)**0.5)
        logger.info(f"Best Random Forest parameters: {rf_grid_search_model.best_params_}")
        logger.info(
            f"Best Random Forest RMSE: {(-rf_grid_search_model.best_score_)**0.5:.4f}"
        )

        # Log Random Forest model to MLflow
        mlflow.sklearn.log_model(rf_grid_search_model.best_estimator_, "random_forest_model")


        imputer_path = os.path.join(output_model_dir, "imputer.pkl")
        with open(imputer_path, "wb") as f:
            pickle.dump(imputer, f)
        logger.info(f"Imputer saved to {imputer_path}")

        # Log imputer as an artifact
        mlflow.log_artifact(imputer_path, "imputer")

        logger.info("Model training complete.")


if __name__ == "__main__":
    # This block allows the script to be run standalone via command line
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
    parser.add_argument(
        "--timestamp",
        type=str,
        default="",
        help="Timestamp to append to the MLflow run name for uniqueness.",
    )
    parser.add_argument(
        "--parent_run_id",
        type=str,
        default="",
        help="ID of the parent MLflow run, used for naming child runs.",
    )
    args = parser.parse_args()
    run_training(args.input_dataset_path, args.output_model_dir, args.log_level,
                 args.log_path, args.no_console_log, args.timestamp, args.parent_run_id)
