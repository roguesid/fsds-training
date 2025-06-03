import mlflow
import os
import logging
from datetime import datetime

# Import the refactored modules.
# Assuming 'scripts' is a package or the current working directory is set up
# such that these imports resolve correctly.
# If 'scripts' is a directory and not a Python package, you might need to
# add its parent directory to PYTHONPATH or adjust imports.
# For simplicity, assuming 'scripts' is a package or directly importable.
import ingest_data
import train
import score

# Set up basic logging for the main script
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_ml_pipeline():
    """
    Orchestrates the entire machine learning pipeline (data ingestion, training, scoring)
    under a single parent MLflow run, with each stage as a nested child run.
    """
    # Configure MLflow Tracking URI and Experiment
    mlflow.set_tracking_uri("http://localhost:5000") # Set the MLflow server URI
    mlflow.set_experiment("first exp for random forest") # Set the MLflow experiment name

    # Generate a timestamp for unique run names
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Define paths for datasets and models
    datasets_dir = "datasets"
    models_dir = "models"
    train_dataset_path = os.path.join(datasets_dir, "housing_train.csv")
    test_dataset_path = os.path.join(datasets_dir, "housing_test.csv")
    scores_path = "scores.txt"

    # Ensure output directories exist
    os.makedirs(datasets_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Start the parent MLflow run with a dynamic name
    with mlflow.start_run(run_name=f"full_ml_pipeline_{timestamp}") as parent_run:
        parent_run_id = parent_run.info.run_id # Get the parent run ID
        logger.info(f"Parent MLflow Run ID: {parent_run_id}")
        mlflow.log_param("pipeline_start_time", mlflow.active_run().info.start_time)

        # Step 1: Data Ingestion and Preparation
        logger.info("--- Starting Data Ingestion and Preparation ---")
        ingest_data.run_ingestion(
            output_dir=datasets_dir,
            log_level="INFO",
            timestamp=timestamp,
            parent_run_id=parent_run_id
        )
        logger.info("Data Ingestion and Preparation completed successfully.")

        # Step 2: Model Training
        logger.info("--- Starting Model Training ---")
        train.run_training(
            input_dataset_path=train_dataset_path,
            output_model_dir=models_dir,
            log_level="INFO",
            timestamp=timestamp,
            parent_run_id=parent_run_id
        )
        logger.info("Model Training completed successfully.")

        # Step 3: Model Scoring (for Random Forest model as an example)
        logger.info("--- Starting Model Scoring (Random Forest) ---")
        score.run_scoring(
            model_dir=models_dir,
            dataset_path=test_dataset_path,
            model_name="random_forest_model.pkl",
            output_scores_path=scores_path,
            log_level="INFO",
            timestamp=timestamp,
            parent_run_id=parent_run_id
        )
        logger.info("Model Scoring (Random Forest) completed successfully.")

        mlflow.log_param("pipeline_end_time", mlflow.active_run().info.end_time)
        mlflow.set_tag("status", "completed")
        logger.info(f"Full ML pipeline completed. View runs at: mlflow ui")


if __name__ == "__main__":
    run_ml_pipeline()
