import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import sys
from io import StringIO

class TestHeartDiseasePredictor(unittest.TestCase):
    """Test suite for Heart Disease Predictor application"""
    
    @classmethod
    def setUpClass(cls):
        """Load and train model once for all tests"""
        print("Setting up test environment...")
        
        # Define column names
        cls.column_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
            'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]
        
        # Load data
        try:
            cls.df = pd.read_csv('heart_data.txt', header=None, names=cls.column_names, na_values='?')
            print(f"✓ Data loaded successfully: {len(cls.df)} rows")
        except FileNotFoundError:
            print("✗ Error: heart_data.txt not found")
            raise
        
        # Handle missing values
        for col in ['ca', 'thal']:
            if cls.df[col].isnull().any():
                cls.df[col] = cls.df[col].fillna(cls.df[col].median())
        
        # Binarize target
        cls.df['target'] = cls.df['target'].apply(lambda x: 1 if x > 0 else 0)
        
        # Prepare features and target
        cls.X = cls.df.drop('target', axis=1)
        cls.y = cls.df['target']
        
        # Scale features
        cls.scaler = StandardScaler()
        cls.X_scaled = cls.scaler.fit_transform(cls.X)
        
        # Train model
        cls.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42
        )
        cls.model.fit(cls.X_scaled, cls.y)
        print("✓ Model trained successfully\n")
    
    def test_01_data_loading(self):
        """Test data file loads correctly"""
        print("TEST 1: Data Loading")
        self.assertIsNotNone(self.df, "Data should load successfully")
        self.assertEqual(len(self.df.columns), 14, "Should have 14 columns")
        self.assertGreater(len(self.df), 0, "Should have at least one row")
        print(f"  ✓ Data has {len(self.df)} rows and {len(self.df.columns)} columns\n")
    
    def test_02_missing_values(self):
        """Test missing value handling"""
        print("TEST 2: Missing Value Handling")
        missing_before = self.df.isnull().sum().sum()
        self.assertEqual(missing_before, 0, "All missing values should be filled")
        print(f"  ✓ No missing values in dataset\n")
    
    def test_03_target_binarization(self):
        """Test target variable is properly binarized"""
        print("TEST 3: Target Binarization")
        unique_targets = self.y.unique()
        self.assertTrue(all(t in [0, 1] for t in unique_targets), "Target should be binary (0 or 1)")
        print(f"  ✓ Target values: {sorted(unique_targets)}")
        print(f"  ✓ Class distribution: {self.y.value_counts().to_dict()}\n")
    
    def test_04_scaler_transformation(self):
        """Test feature scaling"""
        print("TEST 4: Feature Scaling")
        self.assertEqual(self.X_scaled.shape, self.X.shape, "Scaled data should have same shape")
        # Check if data is standardized (mean ≈ 0, std ≈ 1)
        means = np.mean(self.X_scaled, axis=0)
        stds = np.std(self.X_scaled, axis=0)
        self.assertTrue(np.allclose(means, 0, atol=1e-10), "Scaled features should have mean ≈ 0")
        self.assertTrue(np.allclose(stds, 1, atol=0.1), "Scaled features should have std ≈ 1")
        print(f"  ✓ Features scaled correctly (mean≈0, std≈1)\n")
    
    def test_05_model_training(self):
        """Test model is trained and can make predictions"""
        print("TEST 5: Model Training")
        self.assertIsNotNone(self.model, "Model should be trained")
        self.assertTrue(hasattr(self.model, 'predict'), "Model should have predict method")
        
        # Test prediction on first sample
        sample = self.X_scaled[:1]
        prediction = self.model.predict(sample)
        self.assertIn(prediction[0], [0, 1], "Prediction should be 0 or 1")
        print(f"  ✓ Model can make predictions\n")
    
    def test_06_model_accuracy(self):
        """Test model achieves reasonable accuracy"""
        print("TEST 6: Model Performance")
        predictions = self.model.predict(self.X_scaled)
        accuracy = np.mean(predictions == self.y)
        self.assertGreater(accuracy, 0.5, "Model accuracy should be better than random")
        print(f"  ✓ Training accuracy: {accuracy:.2%}\n")
    
    def test_07_low_risk_patient(self):
        """Test prediction for a low-risk patient profile"""
        print("TEST 7: Low Risk Patient Prediction")
        # Healthy profile: young, low cholesterol, no symptoms
        low_risk_patient = pd.DataFrame({
            'age': [35], 'sex': [1], 'cp': [4], 'trestbps': [120],
            'chol': [200], 'fbs': [0], 'restecg': [0], 'thalach': [180],
            'exang': [0], 'oldpeak': [0.0], 'slope': [1], 'ca': [0], 'thal': [3]
        })
        
        scaled = self.scaler.transform(low_risk_patient)
        prediction = self.model.predict(scaled)[0]
        proba = self.model.predict_proba(scaled)[0]
        
        print(f"  Patient Profile: Age 35, Normal indicators")
        print(f"  Prediction: {'High Risk' if prediction == 1 else 'Low Risk'}")
        print(f"  Confidence: Low={proba[0]:.1%}, High={proba[1]:.1%}\n")
    
    def test_08_high_risk_patient(self):
        """Test prediction for a high-risk patient profile"""
        print("TEST 8: High Risk Patient Prediction")
        # High risk profile: older, high cholesterol, symptoms present
        high_risk_patient = pd.DataFrame({
            'age': [70], 'sex': [1], 'cp': [1], 'trestbps': [180],
            'chol': [350], 'fbs': [1], 'restecg': [2], 'thalach': [100],
            'exang': [1], 'oldpeak': [4.0], 'slope': [3], 'ca': [3], 'thal': [7]
        })
        
        scaled = self.scaler.transform(high_risk_patient)
        prediction = self.model.predict(scaled)[0]
        proba = self.model.predict_proba(scaled)[0]
        
        print(f"  Patient Profile: Age 70, Multiple risk factors")
        print(f"  Prediction: {'High Risk' if prediction == 1 else 'Low Risk'}")
        print(f"  Confidence: Low={proba[0]:.1%}, High={proba[1]:.1%}\n")
    
    def test_09_boundary_values(self):
        """Test prediction with boundary values"""
        print("TEST 9: Boundary Value Testing")
        # Test with minimum values
        min_patient = pd.DataFrame({
            'age': [29], 'sex': [0], 'cp': [1], 'trestbps': [94],
            'chol': [126], 'fbs': [0], 'restecg': [0], 'thalach': [71],
            'exang': [0], 'oldpeak': [0.0], 'slope': [1], 'ca': [0], 'thal': [3]
        })
        
        scaled = self.scaler.transform(min_patient)
        prediction = self.model.predict(scaled)[0]
        print(f"  Min boundary values - Prediction: {'High Risk' if prediction == 1 else 'Low Risk'}")
        
        # Test with maximum values
        max_patient = pd.DataFrame({
            'age': [77], 'sex': [1], 'cp': [4], 'trestbps': [200],
            'chol': [564], 'fbs': [1], 'restecg': [2], 'thalach': [202],
            'exang': [1], 'oldpeak': [6.2], 'slope': [3], 'ca': [3], 'thal': [7]
        })
        
        scaled = self.scaler.transform(max_patient)
        prediction = self.model.predict(scaled)[0]
        print(f"  Max boundary values - Prediction: {'High Risk' if prediction == 1 else 'Low Risk'}\n")
    
    def test_10_feature_importance(self):
        """Test and display feature importance"""
        print("TEST 10: Feature Importance Analysis")
        feature_importance = pd.DataFrame({
            'feature': self.column_names[:-1],
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("  Top 5 Most Important Features:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"    {row['feature']:12s}: {row['importance']:.4f}")
        print()
    
    def test_11_prediction_probability(self):
        """Test prediction probability output"""
        print("TEST 11: Prediction Probability Testing")
        sample = self.X_scaled[:1]
        proba = self.model.predict_proba(sample)
        
        self.assertEqual(proba.shape, (1, 2), "Should return probability for 2 classes")
        self.assertAlmostEqual(proba.sum(), 1.0, places=5, msg="Probabilities should sum to 1")
        print(f"  ✓ Probabilities sum to 1.0")
        print(f"  ✓ Output shape correct: {proba.shape}\n")
    
    def test_12_multiple_predictions(self):
        """Test batch predictions"""
        print("TEST 12: Batch Prediction Testing")
        # Test with 5 random samples
        sample_size = 5
        samples = self.X_scaled[:sample_size]
        predictions = self.model.predict(samples)
        
        self.assertEqual(len(predictions), sample_size, f"Should return {sample_size} predictions")
        print(f"  ✓ Batch prediction works for {sample_size} samples")
        print(f"  Results: {predictions}\n")


def run_tests():
    """Run all tests with detailed output"""
    print("=" * 70)
    print("HEART DISEASE PREDICTOR - TEST SUITE")
    print("=" * 70)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestHeartDiseasePredictor)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✓ ALL TESTS PASSED! The application is working correctly.")
    else:
        print("\n✗ SOME TESTS FAILED. Please review the errors above.")
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    run_tests()