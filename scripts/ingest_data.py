import argparse
import logging
import os
import pandas as pd
import mlflow # Import mlflow

from fsds_training import data_ingestion, preprocessing

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


def run_ingestion(output_dir: str, log_level: str = "INFO", log_path: str = None,
                  no_console_log: bool = False, timestamp: str = "", parent_run_id: str = ""):
    """
    Performs data ingestion and preparation, including downloading data,
    adding income categories, performing stratified split, and saving datasets.
    MLflow tracking is integrated to log parameters and metrics.

    Args:
        output_dir (str): Path to the directory where the processed datasets will be saved.
        log_level (str, optional): The desired logging level. Defaults to "INFO".
        log_path (str, optional): Path to a file to write logs to. Defaults to None.
        no_console_log (bool, optional): If True, disables console logging. Defaults to False.
        timestamp (str, optional): Timestamp for unique MLflow run names. Defaults to "".
        parent_run_id (str, optional): ID of the parent MLflow run for naming. Defaults to "".
    """
    run_name = f"data_ingestion_parent_{parent_run_id}_{timestamp}" if parent_run_id and timestamp else (f"data_ingestion_{timestamp}" if timestamp else "data_ingestion")

    # Start an MLflow run, nested=True makes it a child run
    with mlflow.start_run(run_name=run_name, nested=True):
        setup_logging(log_level.upper(), log_path, no_console_log)

        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Ensured output directory '{output_dir}' exists.")

        # Log parameters to MLflow
        mlflow.log_param("output_directory", output_dir)
        mlflow.log_param("log_level", log_level)
        mlflow.log_param("log_path", log_path if log_path else "None")
        mlflow.log_param("no_console_log", no_console_log)
        mlflow.log_param("parent_run_id", parent_run_id)

        logger.info("Fetching housing data...")
        data_ingestion.fetch_housing_data()

        logger.info("Loading housing data...")
        housing = data_ingestion.load_housing_data()

        logger.info("Adding income categories...")
        housing = preprocessing.add_income_cat(housing)

        logger.info("Performing stratified split...")
        strat_train_set, strat_test_set = preprocessing.stratified_split(housing)

        train_output_path = os.path.join(output_dir, "housing_train.csv")
        test_output_path = os.path.join(output_dir, "housing_test.csv")

        logger.info(f"Saving training dataset to {train_output_path}...")
        strat_train_set.to_csv(train_output_path, index=False)

        logger.info(f"Saving test dataset to {test_output_path}...")
        strat_test_set.to_csv(test_output_path, index=False)

        # Log metrics (e.g., number of samples)
        mlflow.log_metric("train_samples", len(strat_train_set))
        mlflow.log_metric("test_samples", len(strat_test_set))
        mlflow.log_metric("total_samples", len(housing))

        # Log artifacts (the datasets themselves)
        mlflow.log_artifact(train_output_path, "datasets")
        mlflow.log_artifact(test_output_path, "datasets")

        logger.info("Data ingestion and preparation complete.")


if __name__ == "__main__":
    # This block allows the script to be run standalone via command line
    parser = argparse.ArgumentParser(
        description="Download and prepare housing data for training and validation."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets",
        help="Path to the directory where the processed datasets will be saved.",
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
    run_ingestion(args.output_dir, args.log_level, args.log_path,
                  args.no_console_log, args.timestamp, args.parent_run_id)
