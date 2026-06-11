import os
import tempfile
import unittest
import pandas as pd

from ai_results import load_gemma3_predictions, compute_specificity, evaluate_gemma3


class TestAiResults(unittest.TestCase):

    def test_load_gemma3_predictions_reads_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "predictions.csv")
            df = pd.DataFrame({
                "real_class": ["control", "anorexia"],
                "new_class": ["control", "anorexia"],
            })
            df.to_csv(csv_path, index=False)

            y_true, y_pred = load_gemma3_predictions(csv_path)
            self.assertEqual(list(y_true), ["control", "anorexia"])
            self.assertEqual(list(y_pred), ["control", "anorexia"])

    def test_load_gemma3_predictions_missing_columns_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "predictions.csv")
            pd.DataFrame({"real_class": ["control"]}).to_csv(csv_path, index=False)

            with self.assertRaises(ValueError):
                load_gemma3_predictions(csv_path)

    def test_compute_specificity_binary(self):
        y_true = ["control", "control", "anorexia", "anorexia"]
        y_pred = ["control", "anorexia", "anorexia", "control"]

        specificity = compute_specificity(y_true, y_pred, positive_label="anorexia")
        self.assertAlmostEqual(specificity, 0.5)

    def test_evaluate_gemma3_returns_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "predictions.csv")
            df = pd.DataFrame({
                "real_class": ["control", "control", "anorexia", "anorexia"],
                "new_class": ["control", "anorexia", "anorexia", "control"],
            })
            df.to_csv(csv_path, index=False)

            results = evaluate_gemma3(csv_path, model_name="test_model", positive_label="anorexia")
            self.assertEqual(results["model"], "test_model")
            self.assertIn("accuracy", results)
            self.assertIn("specificity", results)
            self.assertIn("precision_macro", results)
            self.assertIn("recall_macro", results)
            self.assertIn("f1_macro", results)
