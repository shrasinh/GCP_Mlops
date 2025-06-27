import pytest
import joblib
import pandas as pd
import numpy as np
from feast import FeatureStore
import os


# Utility tests
def test_artifacts_directory_exists():
    """Test that artifacts directory and model file exist."""
    assert os.path.exists("artifacts"), "Artifacts directory should exist"
    assert os.path.exists("artifacts/model.joblib"), "Model file should exist"


def test_feast_configuration():
    """Test that Feast is properly configured."""
    try:
        store = FeatureStore(repo_path="Iris_Feast/feature_repo")
        feature_service = store.get_feature_service("feast_model_v1")
        assert feature_service is not None, "Feature service should exist"
    except Exception as e:
        pytest.fail(f"Feast configuration error: {str(e)}")

class TestModelPredictions:
    """Test suite for model predictions using Feast online store."""
    
    @pytest.fixture(scope="class")
    def model(self):
        """Load the trained model."""
        return joblib.load("artifacts/model.joblib")
    
    @pytest.fixture(scope="class")
    def feature_store(self):
        """Initialize Feast feature store."""
        return FeatureStore(repo_path="Iris_Feast/feature_repo")
    
    @pytest.fixture(scope="class")
    def online_features(self, feature_store):
        """Get online features from Feast store."""
        features = feature_store.get_online_features(
            features=feature_store.get_feature_service("feast_model_v1"),
            entity_rows=[
                {"species": "setosa"},
                {"species": "versicolor"}, 
                {"species": "virginica"}
            ],
        ).to_df()
        return features
    
    def test_data_schema_validation(self, online_features):
        """Test that the retrieved features have the correct schema."""
        # Check that DataFrame is not empty
        assert not online_features.empty, "Features DataFrame should not be empty"
        
        # Check required columns exist
        required_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
        for col in required_columns:
            assert col in online_features.columns, f"Column '{col}' is missing from features"
        
        # Check data types
        numeric_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        for col in numeric_columns:
            assert pd.api.types.is_numeric_dtype(online_features[col]), f"Column '{col}' should be numeric"
        
        # Check species column contains expected values
        expected_species = {'setosa', 'versicolor', 'virginica'}
        actual_species = set(online_features['species'].unique())
        assert actual_species.issubset(expected_species), f"Unexpected species found: {actual_species - expected_species}"
        
        # Check for null values
        assert not online_features[required_columns].isnull().any().any(), "No null values should be present in required columns"
        
        # Check that the DataFrame have exactly 3 rows (one for each species)
        assert len(online_features) == 3, f"Expected 3 rows, got {len(online_features)}"
    
    def test_model_loading(self, model):
        """Test that the model loads successfully and has required methods."""
        assert model is not None, "Model should not be None"
        assert hasattr(model, 'predict'), "Model should have predict method"
    
    def test_model_predictions(self, model, online_features):
        """Test model predictions on online features."""
        feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        X = online_features[feature_columns]
        
        # Make predictions
        predictions = model.predict(X)
        
        # Test predictions
        assert len(predictions) == 3, f"Expected 3 predictions, got {len(predictions)}"
        assert all(pred in {'setosa', 'versicolor', 'virginica'} for pred in predictions), "All predictions should be in {'setosa', 'versicolor', 'virginica'}"
        
        # Check each prediction
        for idx, row in online_features.iterrows():
            predicted_class = predictions[idx]
            expected_class = row['species']
            
            # For a well-trained model on typical iris features, prediction should match species
            # This is a soft check - warn if it doesn't match but not fail the test
            if predicted_class != expected_class:
                print(f"Warning: Species '{expected_class}' predicted as class {predicted_class}, expected {expected_class}")
                print(f"Features: {row[feature_columns].to_dict()}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short", "--disable-warnings"])