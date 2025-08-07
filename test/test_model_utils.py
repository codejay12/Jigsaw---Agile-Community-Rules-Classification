import unittest
import numpy as np
from sklearn.metrics import roc_auc_score
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

class TestModelUtils(unittest.TestCase):
    
    def test_auc_calculation(self):
        """Test AUC score calculation"""
        # Test perfect predictions
        y_true = [1, 1, 0, 0]
        y_pred = [0.9, 0.8, 0.2, 0.1]
        auc = roc_auc_score(y_true, y_pred)
        self.assertEqual(auc, 1.0)
        
        # Test random predictions
        y_true = [1, 0, 1, 0]
        y_pred = [0.5, 0.5, 0.5, 0.5]
        auc = roc_auc_score(y_true, y_pred)
        self.assertAlmostEqual(auc, 0.5, places=1)
    
    def test_probability_normalization(self):
        """Test probability normalization"""
        # Test softmax-like normalization
        logprobs = np.array([0.7, 0.3])
        normalized = logprobs / logprobs.sum()
        
        self.assertAlmostEqual(normalized.sum(), 1.0)
        self.assertTrue(all(0 <= p <= 1 for p in normalized))
    
    def test_ensemble_averaging(self):
        """Test ensemble prediction averaging"""
        # Multiple model predictions
        predictions = [
            [0.7, 0.3],
            [0.8, 0.2],
            [0.6, 0.4]
        ]
        
        # Simple average
        avg_pred = np.mean(predictions, axis=0)
        expected = [0.7, 0.3]
        
        np.testing.assert_array_almost_equal(avg_pred, expected)
    
    def test_weighted_ensemble(self):
        """Test weighted ensemble predictions"""
        predictions = np.array([
            [0.7, 0.3],
            [0.8, 0.2],
            [0.6, 0.4]
        ])
        
        weights = [0.5, 0.3, 0.2]
        weighted_pred = np.average(predictions, axis=0, weights=weights)
        
        expected = [0.71, 0.29]
        np.testing.assert_array_almost_equal(weighted_pred, expected)

if __name__ == '__main__':
    unittest.main()