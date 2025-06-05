import argparse
import logging
import os
import pickle
import pandas as pd
import mlflow # Import mlflow

from fsds_training import preprocessing, scoring, training

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


def run_scoring(model_dir: str, dataset_path: str, model_name: str,
                output_scores_path: str = None, log_level: str = "INFO",
                log_path: str = None, no_console_log: bool = False,
                timestamp: str = "", parent_run_id: str = ""):
    """
    Performs model scoring, including loading model and imputer,
    preparing test data, and evaluating model performance.
    MLflow tracking is integrated to log parameters and evaluation metrics.

    Args:
        model_dir (str): Path to the directory containing the trained model and imputer.
        dataset_path (str): Path to the input dataset (e.g., test set) CSV file.
        model_name (str): Name of the model pickle file to load.
        output_scores_path (str, optional): Path to a file to save the evaluation scores. Defaults to None.
        log_level (str, optional): The desired logging level. Defaults to "INFO".
        log_path (str, optional): Path to a file to write logs to. Defaults to None.
        no_console_log (bool, optional): If True, disables console logging. Defaults to False.
        timestamp (str, optional): Timestamp for unique MLflow run names. Defaults to "".
        parent_run_id (str, optional): ID of the parent MLflow run for naming. Defaults to "".
    """
    # Construct a dynamic run name including the model name, parent run ID, and timestamp
    model_display_name = model_name.replace(".pkl", "").replace("_", " ").title()
    run_name = f"model_scoring_{model_display_name}_parent_{parent_run_id}_{timestamp}" if parent_run_id and timestamp else (f"model_scoring_{model_display_name}_{timestamp}" if timestamp else f"model_scoring_{model_display_name}")

    # Start an MLflow run, nested=True makes it a child run
    with mlflow.start_run(run_name=run_name, nested=True):
        setup_logging(log_level, log_path, no_console_log)

        # Log parameters to MLflow
        mlflow.log_param("model_directory", model_dir)
        mlflow.log_param("dataset_path", dataset_path)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("log_level", log_level)
        mlflow.log_param("log_path", log_path if log_path else "None")
        mlflow.log_param("no_console_log", no_console_log)
        mlflow.log_param("parent_run_id", parent_run_id)


        model_path = os.path.join(model_dir, model_name)
        logger.info(f"Loading model from {model_path}...")
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        except FileNotFoundError:
            logger.error(
                f"Error: Model file not found at {model_path}. Please check the path and model_name."
            )
            return

        imputer_path = os.path.join(model_dir, "imputer.pkl")
        logger.info(f"Loading imputer from {imputer_path}...")
        try:
            with open(imputer_path, "rb") as f:
                imputer = pickle.load(f)
        except FileNotFoundError:
            logger.error(
                f"Error: Imputer file not found at {imputer_path}. Please ensure 'train.py' was run."
            )
            return

        logger.info(f"Loading dataset from {dataset_path}...")
        try:
            housing_test = pd.read_csv(dataset_path)
        except FileNotFoundError:
            logger.error(
                f"Error: Dataset file not found at {dataset_path}. Please check the path."
            )
            return

        logger.info("Adding features to the test dataset...")
        housing_test_with_features = preprocessing.add_features(housing_test.copy())

        housing_test_labels = housing_test_with_features["median_house_value"].copy()
        housing_test_features = housing_test_with_features.drop(
            "median_house_value", axis=1
        )

        logger.info("Preparing test data using the loaded imputer...")
        housing_test_prepared, _ = training.prepare_data(
            housing_test_features, imputer=imputer
        )

        logger.info("Evaluating model performance...")
        metrics = scoring.evaluate_model(model, housing_test_prepared, housing_test_labels)
        logger.info("\n--- Model Evaluation Metrics ---")
        for metric, value in metrics.items():
            logger.info(f"{metric.upper()}: {value:.4f}")
            mlflow.log_metric(f"test_{metric}", value) # Log each metric to MLflow
        logger.info("------------------------------")

        if output_scores_path:
            logger.info(f"Saving scores to {output_scores_path}...")
            with open(output_scores_path, "w") as f:
                f.write("--- Model Evaluation Metrics ---\n")
                for metric, value in metrics.items():
                    f.write(f"{metric.upper()}: {value:.4f}\n")
                f.write("------------------------------\n")
            mlflow.log_artifact(output_scores_path, "evaluation_scores")

        logger.info("Model scoring complete.")


if __name__ == "__main__":
    # This block allows the script to be run standalone via command line
    parser = argparse.ArgumentParser(description="Score machine learning models.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Path to the directory containing the trained model (pickle) and imputer.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="datasets/housing_test.csv",
        help="Path to the input dataset (e.g., test set) CSV file.",
    )
    parser.add_argument(
        "--output_scores_path",
        type=str,
        default=None,
        help="Optional: Path to a file to save the evaluation scores (e.g., scores.txt).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="random_forest_model.pkl",
        help="Name of the model pickle file to load (e.g., linear_regression_model.pkl or random_forest_model.pkl).",
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
    run_scoring(args.model_dir, args.dataset_path, args.model_name,
                args.output_scores_path, args.log_level, args.log_path,
                args.no_console_log, args.timestamp, args.parent_run_id)
