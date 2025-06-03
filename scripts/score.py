import argparse
import logging
import os
import pickle

import pandas as pd

from fsds_training import preprocessing, scoring, training


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
    args = parser.parse_args()

    setup_logging(args.log_level, args.log_path, args.no_console_log)
    logger = logging.getLogger(__name__)

    model_dir = args.model_dir
    dataset_path = args.dataset_path
    output_scores_path = args.output_scores_path
    model_name = args.model_name

    model_path = os.path.join(model_dir, model_name)
    logger.info(f"Loading model from {model_path}...")
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully.")
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
        logger.info("Imputer loaded successfully.")
    except FileNotFoundError:
        logger.error(
            f"Error: Imputer file not found at {imputer_path}. Please ensure 'train.py' was run."
        )
        return

    logger.info(f"Loading dataset from {dataset_path}...")
    try:
        housing_test = pd.read_csv(dataset_path)
        logger.info("Dataset loaded successfully.")
    except FileNotFoundError:
        logger.error(
            f"Error: Dataset file not found at {dataset_path}. Please check the path."
        )
        return

    logger.info("Adding features to the test dataset...")
    housing_test_with_features = preprocessing.add_features(housing_test.copy())
    logger.info("Features added to test dataset.")

    housing_test_labels = housing_test_with_features["median_house_value"].copy()
    housing_test_features = housing_test_with_features.drop(
        "median_house_value", axis=1
    )

    logger.info("Preparing test data using the loaded imputer...")
    housing_test_prepared, _ = training.prepare_data(
        housing_test_features, imputer=imputer
    )
    logger.info("Test data prepared.")

    logger.info("Evaluating model performance...")
    metrics = scoring.evaluate_model(model, housing_test_prepared, housing_test_labels)
    logger.info("\n--- Model Evaluation Metrics ---")
    for metric, value in metrics.items():
        logger.info(f"{metric.upper()}: {value:.4f}")
    logger.info("------------------------------")

    if output_scores_path:
        logger.info(f"Saving scores to {output_scores_path}...")
        with open(output_scores_path, "w") as f:
            f.write("--- Model Evaluation Metrics ---\n")
            for metric, value in metrics.items():
                f.write(f"{metric.upper()}: {value:.4f}\n")
            f.write("------------------------------\n")
        logger.info("Scores saved.")

    logger.info("Model scoring complete.")


if __name__ == "__main__":
    main()
