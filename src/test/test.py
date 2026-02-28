"""
Unit tests for ML-PROJECT: regression and classification models.
Tests training, prediction, and evaluation functions.
"""

import unittest

import numpy as np


class TestEvaluationFunctions(unittest.TestCase):
    """Test the evaluation.py utility functions."""

    def setUp(self):
        from src.modeling.evaluation import (
            confusion_matrix_scratch,
            evaluate_classification,
            evaluate_regression,
        )

        self.confusion_matrix_scratch = confusion_matrix_scratch
        self.evaluate_classification = evaluate_classification
        self.evaluate_regression = evaluate_regression

    def test_confusion_matrix_perfect(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        tp, tn, fp, fn = self.confusion_matrix_scratch(y_true, y_pred)
        self.assertEqual(tp, 2)
        self.assertEqual(tn, 2)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 0)

    def test_confusion_matrix_all_wrong(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 1, 1])
        tp, tn, fp, fn = self.confusion_matrix_scratch(y_true, y_pred)
        self.assertEqual(tp, 0)
        self.assertEqual(tn, 0)
        self.assertEqual(fp, 2)
        self.assertEqual(fn, 2)

    def test_evaluate_classification_perfect(self):
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0, 1, 0])
        metrics = self.evaluate_classification(y_true, y_pred)
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 1.0)

    def test_evaluate_regression_perfect(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        metrics = self.evaluate_regression(y_true, y_pred)
        self.assertAlmostEqual(metrics["mse"], 0.0)
        self.assertAlmostEqual(metrics["rmse"], 0.0)
        self.assertAlmostEqual(metrics["mae"], 0.0)
        self.assertAlmostEqual(metrics["r2"], 1.0)

    def test_evaluate_regression_nonperfect(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1])
        metrics = self.evaluate_regression(y_true, y_pred)
        self.assertGreater(metrics["mse"], 0)
        self.assertGreater(metrics["r2"], 0.9)


class TestDataLoading(unittest.TestCase):
    """Test data loading functions."""

    def test_load_regression_data(self):
        from src.data.regression_data import load_regression_data

        X, y = load_regression_data()
        self.assertGreater(len(X), 0)
        self.assertEqual(len(X), len(y))
        self.assertEqual(X.ndim, 2)

    def test_load_classification_data(self):
        from src.data.classification_data import load_classification_data

        X, y, feature_cols = load_classification_data()
        self.assertGreater(len(X), 0)
        self.assertEqual(len(X), len(y))
        self.assertEqual(X.ndim, 2)
        self.assertEqual(X.shape[1], len(feature_cols))
        # Target should be binary
        unique_vals = np.unique(y)
        self.assertTrue(set(unique_vals).issubset({0, 1}))


class TestRegressionModels(unittest.TestCase):
    """Test regression model training and prediction imports."""

    def test_linear_regression_train_import(self):
        from src.modeling.regression.linear_regression import train

        self.assertTrue(hasattr(train, "train"))

    def test_poly_regression_train_import(self):
        from src.modeling.regression.poly_regression import train

        self.assertTrue(hasattr(train, "train"))

    def test_multiple_regression_train_import(self):
        from src.modeling.regression.multiple_regression import train

        self.assertTrue(hasattr(train, "train"))

    def test_mlp_regression_train_import(self):
        from src.modeling.regression.mlp_regression import train

        self.assertTrue(hasattr(train, "train"))

    def test_scratch_linear_predict_import(self):
        try:
            from src.modeling.regression.scratch.linear_regression import predict

            self.assertTrue(hasattr(predict, "predict"))
        except FileNotFoundError:
            self.skipTest("Model file not trained yet")

    def test_lib_linear_predict_import(self):
        try:
            from src.modeling.regression.lib.linear_regression import predict

            self.assertTrue(hasattr(predict, "predict"))
        except FileNotFoundError:
            self.skipTest("Model file not trained yet")


class TestClassificationModels(unittest.TestCase):
    """Test classification model training and prediction imports."""

    def test_logistic_regression_train_import(self):
        from src.modeling.classification.scratch.logistic_regression import train

        self.assertTrue(hasattr(train, "train"))

    def test_perceptron_train_import(self):
        from src.modeling.classification.scratch.perceptron import train

        self.assertTrue(hasattr(train, "train"))

    def test_mlp_train_import(self):
        from src.modeling.classification.scratch.mlp import train

        self.assertTrue(hasattr(train, "train"))

    def test_decision_tree_train_import(self):
        from src.modeling.classification.scratch.decision_tree import train

        self.assertTrue(hasattr(train, "train"))

    def test_random_forest_train_import(self):
        from src.modeling.classification.scratch.random_forest import train

        self.assertTrue(hasattr(train, "train"))

    def test_svm_train_import(self):
        from src.modeling.classification.scratch.svm import train

        self.assertTrue(hasattr(train, "train"))

    def test_custom_classification_train_import(self):
        from src.modeling.classification.scratch.custom_classification import train

        self.assertTrue(hasattr(train, "train"))

    def test_clustering_train_import(self):
        from src.modeling.classification.scratch.clustering import train

        self.assertTrue(hasattr(train, "train"))

    def test_lib_logistic_regression_import(self):
        try:
            from src.modeling.classification.lib.logistic_regression import predict

            self.assertTrue(hasattr(predict, "predict"))
        except FileNotFoundError:
            self.skipTest("Model file not trained yet")

    def test_lib_decision_tree_import(self):
        try:
            from src.modeling.classification.lib.decision_tree import predict

            self.assertTrue(hasattr(predict, "predict"))
        except FileNotFoundError:
            self.skipTest("Model file not trained yet")

    def test_lib_svm_import(self):
        try:
            from src.modeling.classification.lib.svm import predict

            self.assertTrue(hasattr(predict, "predict"))
        except FileNotFoundError:
            self.skipTest("Model file not trained yet")

    def test_lib_clustering_import(self):
        try:
            from src.modeling.classification.lib.clustering import predict

            self.assertTrue(hasattr(predict, "predict"))
        except FileNotFoundError:
            self.skipTest("Model file not trained yet")


class TestClassificationModelStructure(unittest.TestCase):
    """Test classification scratch model classes."""

    def test_logistic_regression_model(self):
        from src.modeling.classification.scratch.logistic_regression.model import LogisticRegressionScratch

        model = LogisticRegressionScratch(learning_rate=0.01, n_iterations=10)
        self.assertTrue(hasattr(model, "fit"))
        self.assertTrue(hasattr(model, "predict"))

    def test_decision_tree_model(self):
        from src.modeling.classification.scratch.decision_tree.model import DecisionTreeScratch

        model = DecisionTreeScratch(max_depth=3)
        self.assertTrue(hasattr(model, "fit"))
        self.assertTrue(hasattr(model, "predict"))

    def test_perceptron_model(self):
        from src.modeling.classification.scratch.perceptron.model import PerceptronScratch

        model = PerceptronScratch(learning_rate=0.01, n_iterations=10)
        self.assertTrue(hasattr(model, "fit"))
        self.assertTrue(hasattr(model, "predict"))


if __name__ == "__main__":
    unittest.main()
