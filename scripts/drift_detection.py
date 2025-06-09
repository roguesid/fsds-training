import argparse
import logging
import pandas as pd
import os
import json # Import json module

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

logger = logging.getLogger(__name__)

def setup_logging(log_level):
    """Sets up basic logging for the script."""
    # Ensure logging doesn't interfere with JSON output
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

def run_monitoring(current_predictions_path: str, reference_predictions_path: str,
                   report_output_path: str,
                   drift_threshold: float = 0.05, log_level: str = "INFO",
                   prediction_column: str = "predicted_median_house_value"):
    """
    Performs model monitoring by detecting drift in model predictions using EvidentlyAI.

    Compares current predictions with reference predictions, generates an HTML report,
    and checks for significant prediction drift. Prints a JSON object to stdout
    indicating drift status and details.

    Args:
        current_predictions_path (str): Path to the CSV file of current model predictions.
        reference_predictions_path (str): Path to the CSV file of reference/baseline predictions.
        report_output_path (str): Path to save the EvidentlyAI HTML report.
        drift_threshold (float): Threshold for the 'share_of_drifted_features' metric from Evidently.
        log_level (str): Logging level.
        prediction_column (str): The name of the prediction column.
    """
    setup_logging(log_level.upper())

    output_status = {"drift_detected": False, "timestamp": pd.Timestamp.now().isoformat()} # Default output status

    logger.info(f"Loading current predictions from {current_predictions_path}...")
    try:
        current_data = pd.read_csv(current_predictions_path)
    except FileNotFoundError:
        logger.error(f"Error: Current predictions file not found at {current_predictions_path}.")
        print(json.dumps(output_status)) # Print default status
        return
    except Exception as e:
        logger.error(f"Error loading current predictions: {e}")
        print(json.dumps(output_status)) # Print default status
        return

    logger.info(f"Loading reference predictions from {reference_predictions_path}...")
    try:
        reference_data = pd.read_csv(reference_predictions_path)
    except FileNotFoundError:
        logger.error(f"Error: Reference predictions file not found at {reference_predictions_path}.")
        print(json.dumps(output_status)) # Print default status
        return
    except Exception as e:
        logger.error(f"Error loading reference predictions: {e}")
        print(json.dumps(output_status)) # Print default status
        return

    if prediction_column not in current_data.columns or prediction_column not in reference_data.columns:
        logger.error(f"Error: Prediction column '{prediction_column}' not found in one or both datasets.")
        print(json.dumps(output_status)) # Print default status
        return

    logger.info("Generating EvidentlyAI Data Drift Report...")
    data_drift_report = Report(metrics=[
        DataDriftPreset()
    ])
    data_drift_report.run(
        current_data=current_data,
        reference_data=reference_data,
        column_mapping=None
    )

    # Save the HTML report regardless of drift detection
    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
    data_drift_report.save_html(report_output_path)
    logger.info(f"EvidentlyAI HTML report saved to {report_output_path}")

    report_dict = data_drift_report.as_dict()
    # Note: Evidently report structure can change. This assumes DataDriftPreset structure.
    dataset_drift_detected = report_dict['metrics'][0]['result']['dataset_drift']
    share_of_drifted_columns = report_dict['metrics'][0]['result']['share_of_drifted_columns']
    number_of_drifted_columns = report_dict['metrics'][0]['result']['number_of_drifted_columns']
    number_of_features = report_dict['metrics'][0]['result']['number_of_columns']

    drifting_features_info = []
    # Check if 'drift_by_columns' exists under the main DataDriftPreset metric result
    if 'drift_by_columns' in report_dict['metrics'][0]['result']:
        for feature_name, feature_metrics in report_dict['metrics'][1]['result']['drift_by_columns'].items():
            if feature_metrics.get('drift_detected', False): # Use .get for robustness
                drifting_features_info.append({
                    'name': feature_name,
                    'p_value': feature_metrics.get('p_value', 'N/A'),
                    'stattest_name': feature_metrics.get('stattest_name', 'N/A'),
                    'current_mean': feature_metrics.get('current_stattest', {}).get('current_distribution_mean', 'N/A'),
                    'reference_mean': feature_metrics.get('current_stattest', {}).get('reference_distribution_mean', 'N/A')
                })

    if dataset_drift_detected or share_of_drifted_columns > drift_threshold:
        logger.warning(f"MODEL PREDICTION DRIFT DETECTED: Dataset drift={dataset_drift_detected}, Share of drifting features={share_of_drifted_columns:.2f} (Threshold: {drift_threshold:.2f}).")
        output_status = {
            "drift_detected": True,
            "dataset_drift_detected": dataset_drift_detected,
            "share_of_drifted_columns": share_of_drifted_columns,
            "number_of_drifted_columns": number_of_drifted_columns,
            "number_of_features": number_of_features,
            "drift_threshold": drift_threshold,
            "report_path": report_output_path,
            "drifting_features": drifting_features_info,
            "timestamp": pd.Timestamp.now().isoformat()
        }
    else:
        logger.info("No significant model prediction drift detected.")
        output_status = {"drift_detected": False, "timestamp": pd.Timestamp.now().isoformat()}

    # Print the JSON status to stdout for the calling script/workflow to parse
    print(json.dumps(output_status))

# The __main__ block remains unchanged
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run EvidentlyAI model prediction monitoring."
    )
    parser.add_argument(
        "--current_predictions_path",
        type=str,
        required=True,
        help="Path to the CSV file of current model predictions (output of inference.py).",
    )
    parser.add_argument(
        "--reference_predictions_path",
        type=str,
        required=True,
        help="Path to the CSV file of reference/baseline predictions.",
    )
    parser.add_argument(
        "--report_output_path",
        type=str,
        default="evidently_reports/prediction_drift_report.html",
        help="Path to save the EvidentlyAI HTML report.",
    )
    parser.add_argument(
        "--drift_threshold",
        type=float,
        default=0.05,
        help="Share of drifting features that triggers a drift detection.",
    )
    parser.add_argument(
        "--prediction_column",
        type=str,
        default="predicted_median_house_value",
        help="Name of the prediction column in the datasets.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (e.g., INFO, DEBUG).",
    )
    args = parser.parse_args()
    run_monitoring(args.current_predictions_path, args.reference_predictions_path,
                   args.report_output_path,
                   args.drift_threshold, args.log_level,
                   args.prediction_column)