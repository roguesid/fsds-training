import argparse
import logging
import os
import pickle
import pandas as pd

from fsds_training import preprocessing, training

logger = logging.getLogger(__name__)

def setup_logging(log_level, log_path=None, no_console_log=False):
    """
    Sets up logging for the script, allowing for console and file logging.

    Args:
        log_level (str): The desired logging level (e.g., "INFO", "DEBUG").
        log_path (str, optional): Path to a file to write logs to. Defaults to None.
        no_console_log (bool, optional): If True, disables console logging. Defaults to False.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

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


def run_inference(input_data_path: str, model_path: str, imputer_path: str,
                  output_predictions_path: str, log_level: str = "INFO",
                  log_path: str = None, no_console_log: bool = False):
    """
    Performs inference using a trained model and saves predictions to a CSV file.

    Args:
        input_data_path (str): Path to the input data CSV file for inference.
        model_path (str): Path to the trained model pickle file.
        imputer_path (str): Path to the imputer pickle file used during training.
        output_predictions_path (str): Path to the CSV file where predictions will be saved.
        log_level (str, optional): The desired logging level. Defaults to "INFO".
        log_path (str, optional): Path to a file to write logs to. Defaults to None.
        no_console_log (bool, optional): If True, disables console logging. Defaults to False.
    """
    setup_logging(log_level.upper(), log_path, no_console_log)

    logger.info(f"Loading input data from {input_data_path}...")
    try:
        input_data = pd.read_csv(input_data_path)
    except FileNotFoundError:
        logger.error(f"Error: Input data file not found at {input_data_path}.")
        return

    logger.info(f"Loading model from {model_path}...")
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        logger.error(f"Error: Model file not found at {model_path}. Please check the path.")
        return
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        return

    logger.info(f"Loading imputer from {imputer_path}...")
    try:
        with open(imputer_path, "rb") as f:
            imputer = pickle.load(f)
    except FileNotFoundError:
        logger.error(
            f"Error: Imputer file not found at {imputer_path}. Please ensure 'train.py' was run and imputer was saved."
        )
        return
    except Exception as e:
        logger.error(f"Error loading imputer from {imputer_path}: {e}")
        return

    logger.info("Adding features to the input data...")
    input_data_with_features = preprocessing.add_features(input_data.copy())

    logger.info("Preparing input data using the loaded imputer...")
    if "median_house_value" in input_data_with_features.columns:
        inference_features = input_data_with_features.drop("median_house_value", axis=1)
    else:
        inference_features = input_data_with_features

    housing_prepared, _ = training.prepare_data(inference_features, imputer=imputer)

    logger.info("Making predictions...")
    predictions = model.predict(housing_prepared)

    input_data["predicted_median_house_value"] = predictions

    logger.info(f"Saving predictions to {output_predictions_path}...")
    output_dir = os.path.dirname(output_predictions_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    input_data.to_csv(output_predictions_path, index=False)
    logger.info("Inference complete and predictions saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform inference using a trained model and save predictions."
    )
    parser.add_argument(
        "--input_data_path",
        type=str,
        required=True,
        help="Path to the input data CSV file for inference.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/random_forest_model.pkl",
        help="Path to the trained model pickle file (e.g., models/random_forest_model.pkl).",
    )
    parser.add_argument(
        "--imputer_path",
        type=str,
        default="models/imputer.pkl",
        help="Path to the imputer pickle file (e.g., models/imputer.pkl).",
    )
    parser.add_argument(
        "--output_predictions_path",
        type=str,
        default="predictions.csv",
        help="Path to the CSV file where predictions will be saved.",
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
    run_inference(args.input_data_path, args.model_path, args.imputer_path,
                  args.output_predictions_path, args.log_level,
                  args.log_path, args.no_console_log)

