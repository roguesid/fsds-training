import argparse
import os
import pandas as pd
from fsds_training import data_ingestion
from fsds_training import preprocessing
import logging


def setup_logging(log_level, log_path=None, no_console_log=False):
    logger = logging.getLogger()
    logger.setLevel(log_level)

    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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
    args = parser.parse_args()

    setup_logging(args.log_level.upper(), args.log_path, args.no_console_log)

    logger = logging.getLogger(__name__)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Ensured output directory '{output_dir}' exists.")

    logger.info("Fetching housing data...")
    data_ingestion.fetch_housing_data()
    logger.info("Housing data fetched.")

    logger.info("Loading housing data...")
    housing = data_ingestion.load_housing_data()
    logger.info("Housing data loaded.")

    logger.info("Adding income categories...")
    housing = preprocessing.add_income_cat(housing)
    logger.info("Income categories added.")

    logger.info("Performing stratified split...")
    strat_train_set, strat_test_set = preprocessing.stratified_split(housing)
    logger.info("Stratified split complete.")

    train_output_path = os.path.join(output_dir, "housing_train.csv")
    test_output_path = os.path.join(output_dir, "housing_test.csv")

    logger.info(f"Saving training dataset to {train_output_path}...")
    strat_train_set.to_csv(train_output_path, index=False)
    logger.info("Training dataset saved.")

    logger.info(f"Saving test dataset to {test_output_path}...")
    strat_test_set.to_csv(test_output_path, index=False)
    logger.info("Test dataset saved.")

    logger.info("Data ingestion and preparation complete.")

if __name__ == "__main__":
    main()
