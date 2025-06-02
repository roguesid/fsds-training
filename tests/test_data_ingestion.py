import os
import sys
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data_ingestion import fetch_housing_data, load_housing_data


HOUSING_PATH = "datasets/housing"


def test_fetch_and_load_housing_data():
    fetch_housing_data()
    assert os.path.isdir(HOUSING_PATH), "Housing directory should exist after fetch."
    csv_path = os.path.join(HOUSING_PATH, "housing.csv")
    assert os.path.isfile(csv_path), "housing.csv should exist after extraction."
    
    df = load_housing_data()
    assert not df.empty, "Loaded DataFrame should not be empty."
    assert "ocean_proximity" in df.columns, "Expected column 'ocean_proximity' missing."
    assert df.shape[0] > 0 and df.shape[1] > 0, "DataFrame should have rows and columns."

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
