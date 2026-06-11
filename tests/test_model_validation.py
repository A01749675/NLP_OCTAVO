import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from model_validation import (
    parse_model_filename,
    align_features_to_training,
    get_positive_label,
    evaluate_model,
)


class DummyBinaryModel:
    def __init__(self, classes_, predictions, proba_scores):
        self.classes_ = classes_
        self._predictions = np.array(predictions)
        self._proba_scores = np.array(proba_scores)

    def predict(self, X):
        return self._predictions

    def predict_proba(self, X):
        return self._proba_scores


class TestModelValidation(unittest.TestCase):

    def test_parse_model_filename_valid(self):
        model_name, representation = parse_model_filename("rf-tfidf.pkl")
        self.assertEqual(model_name, "rf")
        self.assertEqual(representation, "tfidf")

    def test_parse_model_filename_invalid_raises(self):
        with self.assertRaises(ValueError):
            parse_model_filename("invalidfilename.pkl")

    def test_align_features_to_training_adds_missing_and_removes_extra(self):
        X_test = pd.DataFrame({"a": [1], "b": [2]})
        aligned = align_features_to_training(X_test, ["a", "c"])
        self.assertEqual(list(aligned.columns), ["a", "c"])
        self.assertEqual(aligned.loc[0, "c"], 0)

    def test_align_features_to_training_returns_same_when_no_features(self):
        X_test = pd.DataFrame({"a": [1]})
        aligned = align_features_to_training(X_test, None)
        self.assertTrue(aligned.equals(X_test))

    def test_get_positive_label_from_model_classes(self):
        model = MagicMock()
        model.classes_ = ["control", "anorexia"]
        positive_label = get_positive_label(["control", "anorexia"], model)
        self.assertEqual(positive_label, "anorexia")

    def test_get_positive_label_from_labels(self):
        model = MagicMock(spec=[])
        positive_label = get_positive_label(["anorexia", "control"], model)
        self.assertEqual(positive_label, "control")

    def test_get_positive_label_invalid_labels_raises(self):
        model = MagicMock()
        with self.assertRaises(ValueError):
            get_positive_label(["control"], model)

    def test_evaluate_model_computes_metrics_and_auc(self):
        y_test = ["control", "anorexia"]
        X_test = pd.DataFrame({"f1": [1.0, 2.0]})
        model = DummyBinaryModel(
            classes_=["control", "anorexia"],
            predictions=["control", "anorexia"],
            proba_scores=[[0.8, 0.2], [0.1, 0.9]],
        )

        y_pred, results = evaluate_model(model, X_test, y_test)

        self.assertEqual(list(y_pred), ["control", "anorexia"])
        self.assertAlmostEqual(results["accuracy"], 1.0)
        self.assertAlmostEqual(results["specificity"], 1.0)
        self.assertAlmostEqual(results["auc"], 1.0)
        self.assertIn("precision_macro", results)
        self.assertIn("recall_macro", results)
        self.assertIn("f1_macro", results)
