import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from evaluation import evaluate_model, print_metrics, plot_class_distribution, plot_confusion_matrix, plot_random_forest_feature_importance, plot_logistic_regression_coefficients, plot_train_vs_test_accuracy_rf, plot_roc_auc, calculate_auc

# ---------------------------------------------------------
# Tests for the evaluate_model() function
# ---------------------------------------------------------

class TestEvaluateModel(unittest.TestCase):

    def test_evaluate_model_calculates_correct_metrics(self):
        """Test that metrics are calculated correctly based on controlled predictions."""

        # 1. Set up fake data
        X_test_fake = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]

        # Real values: 2 Zeros, 2 Ones
        y_test_fake = [0, 0, 1, 1]

        # Fake model predictions:
        # Misses one 0 (predicts 1) and misses one 1 (predicts 0)
        mock_predictions = np.array([0, 1, 1, 0])

        # Create the simulated model and define the .predict() output
        mock_model = MagicMock()
        mock_model.predict.return_value = mock_predictions

        # 2. Execute the function
        y_pred, results = evaluate_model(mock_model, X_test_fake, y_test_fake)

        # 3. Call verifications
        mock_model.predict.assert_called_once_with(X_test_fake)
        np.testing.assert_array_equal(y_pred, mock_predictions)

        # 4. Math verifications
        # - Accuracy: 2 out of 4 correct = 0.5 (50%)
        self.assertEqual(results["accuracy"], 0.5)

        # - Specificity: TN / (TN + FP)
        # True Negatives (Was 0, predicted 0): 1
        # False Positives (Was 0, predicted 1): 1
        # Spec = 1 / (1 + 1) = 0.5
        self.assertEqual(results["specificity"], 0.5)

        # - Check for macro keys existence
        self.assertIn("precision_macro", results)
        self.assertIn("recall_macro", results)
        self.assertIn("f1_macro", results)

    def test_perfect_predictions(self):
        """Test an extreme case: the model predicts everything perfectly."""
        X_test_fake = [[1], [2], [3], [4]]
        y_test_fake = [0, 1, 0, 1]

        mock_model = MagicMock()
        # Predict exactly the same as the real values
        mock_model.predict.return_value = np.array([0, 1, 0, 1])

        _, results = evaluate_model(mock_model, X_test_fake, y_test_fake)

        # If everything is perfect, all metrics should be 1.0 (100%)
        self.assertEqual(results["accuracy"], 1.0)
        self.assertEqual(results["specificity"], 1.0)
        self.assertEqual(results["precision_macro"], 1.0)
        self.assertEqual(results["recall_macro"], 1.0)
        self.assertEqual(results["f1_macro"], 1.0)

# ---------------------------------------------------------
# Tests for the print_metrics() function
# ---------------------------------------------------------

class TestPrintMetrics(unittest.TestCase):

    @patch("builtins.print")
    @patch("builtins.print")
    def test_print_metrics_with_all_data(self, mock_print):
        """Test that all metrics are printed with the correct format."""

        # 1. Set up fake data
        fake_results = {
            "accuracy": 0.95,
            "precision_macro": 0.94,
            "recall_macro": 0.93,
            "f1_macro": 0.94,
            "specificity": 0.96,
            "auc": 0.99
        }

        # 2. Execute the function
        print_metrics(fake_results)

        # 3. Verify that the calls to print() were made in the correct order
        expected_calls = [
            call("Accuracy:", 0.95),
            call("Precision:", 0.94),
            call("Recall:", 0.93),
            call("F1 Score:", 0.94),
            call("Specificity:", 0.96),
            call("AUC:", 0.99)
        ]

        mock_print.assert_has_calls(expected_calls, any_order=False)

    @patch("builtins.print")
    @patch("builtins.print")
    def test_print_metrics_missing_auc(self, mock_print):
        """Test that the AUC fallback to 'N/A' works correctly."""

        fake_results = {
            "accuracy": 0.80,
            "precision_macro": 0.80,
            "recall_macro": 0.80,
            "f1_macro": 0.80,
            "specificity": 0.80
        }

        print_metrics(fake_results)

        # assert_any_call verifies that this line was printed at some point during execution
        mock_print.assert_any_call("AUC:", "N/A")

# ---------------------------------------------------------
# Tests for the plot_class_distribution() function
# ---------------------------------------------------------

class TestPlotClassDistribution(unittest.TestCase):

    def tearDown(self):
        """Close in-memory plots to avoid saturating RAM."""
        plt.close('all')

    @patch("matplotlib.pyplot.show")
    def test_plot_class_distribution_generates_correct_elements(self, mock_show):
        """Test that the class distribution plot generates the correct visual elements."""
        # 1. Set up fake data
        y_fake = pd.Series(["A", "A", "A", "B", "B"])

        # 2. Call the function
        plot_class_distribution(y_fake)

        # 3. Extract the plot from memory
        ax = plt.gca()

        # 4. Verifications
        self.assertEqual(ax.get_title(), "Class Distribution")
        self.assertEqual(ax.get_xlabel(), "Class")
        self.assertEqual(ax.get_ylabel(), "Count")

        # Check that 2 bars were drawn (for class A and B)
        bars = ax.patches
        self.assertEqual(len(bars), 2)

        # Verify that plt.show() was called
        mock_show.assert_called_once()

# ---------------------------------------------------------
# Tests for the plot_confusion_matrix() function
# ---------------------------------------------------------

class TestPlotConfusionMatrix(unittest.TestCase):

    def tearDown(self):
        """Clean up graphical memory after the test."""
        plt.close('all')

    @patch("matplotlib.pyplot.show")
    def test_plot_confusion_matrix_generates_correctly(self, mock_show):
        """Test that the confusion matrix is drawn with the correct title without blocking execution."""

        # 1. Prepare fake data
        y_test_fake = [0, 1, 0, 1]
        y_pred_fake = [0, 0, 1, 1]
        test_title = "Test Matrix"

        # 2. Call the function
        plot_confusion_matrix(y_test_fake, y_pred_fake, title=test_title)

        # 3. Extract the drawn plot from memory
        ax = plt.gca()

        # 4. Basic verifications
        self.assertEqual(ax.get_title(), test_title)

        # 5. Advanced verification: Check that the heatmap was drawn
        # If the ax.images list has elements, it means the heatmap exists
        self.assertGreater(len(ax.images), 0, "The matrix heatmap was not drawn.")

        # 6. Check that plt.show() was intercepted
        mock_show.assert_called_once()

# ---------------------------------------------------------
# Tests for the plot_random_forest_feature_importance() function
# ---------------------------------------------------------

class TestPlotFeatureImportance(unittest.TestCase):

    def tearDown(self):
        """Clean up graphical memory after the test."""
        plt.close('all')

    @patch("matplotlib.pyplot.show")
    def test_plot_generates_correct_top_n_bars(self, mock_show):
        """Test that the plot draws exactly the 'top_n' features."""

        # 1. Prepare a fake model WITH feature_importances_
        mock_model = MagicMock()
        # Provide 5 fake importance values
        mock_model.feature_importances_ = [0.1, 0.5, 0.2, 0.15, 0.05]

        # 2. Prepare fake data (X must have the same 5 columns)
        X_fake = pd.DataFrame(columns=["feat_1", "feat_2", "feat_3", "feat_4", "feat_5"])

        # 3. Call the function, requesting only the TOP 3
        top_n_test = 3
        plot_random_forest_feature_importance(mock_model, X_fake, top_n=top_n_test)

        # 4. Extract the plot from memory
        ax = plt.gca()

        # 5. Verifications
        self.assertEqual(ax.get_title(), f"Top {top_n_test} Feature Importances")
        self.assertEqual(ax.get_xlabel(), "Importance")

        # Verify the number of bars drawn as patches
        # If 5 features exist but top 3 is requested, only 3 bars should exist
        bars = ax.patches
        self.assertEqual(len(bars), top_n_test)

        # Verify that the window did not actually open
        mock_show.assert_called_once()

    @patch("builtins.print")
    @patch("matplotlib.pyplot.show")
    def test_early_exit_for_invalid_models(self, mock_show, mock_print):
        """Test that the function aborts gracefully if the model does not support feature_importances."""

        # 1. Create a "dummy" model that lacks the property (e.g., KNN)
        class ModelWithoutImportance:
            pass

        invalid_model = ModelWithoutImportance()
        X_fake = pd.DataFrame(columns=["feat_1", "feat_2"])

        # 2. Call the function
        plot_random_forest_feature_importance(invalid_model, X_fake)

        # 3. Verify safety behavior
        # Ensure the exact message is printed
        mock_print.assert_called_once_with("This model does not have feature_importances_.")

        # Ensure plt.show() is never called
        mock_show.assert_not_called()

# ---------------------------------------------------------
# Tests for the plot_logistic_regression_coefficients() function
# ---------------------------------------------------------

class TestPlotLogisticRegressionCoefficients(unittest.TestCase):

    def tearDown(self):
        """Clean up graphical memory after the test."""
        plt.close('all')

    @patch("matplotlib.pyplot.show")
    def test_plot_generates_correct_positive_and_negative_bars(self, mock_show):
        """Test that the plot correctly extracts the Top N positive and negative coefficients."""

        # 1. Simulate the "Classifier" (the final model inside the pipeline)
        mock_classifier = MagicMock()

        # Coefficients in Scikit-Learn are typically a 2D array: [[coef1, coef2, ...]]
        # Provide 6 coefficients: 3 highly positive, 3 highly negative
        mock_classifier.coef_ = np.array([[5.0, 4.0, 1.0, -1.0, -4.0, -5.0]])

        # 2. Simulate the "Pipeline" containing the classifier
        mock_model = MagicMock()
        mock_model.named_steps = {"classifier": mock_classifier}

        # 3. Prepare fake data with 6 columns
        X_fake = pd.DataFrame(
            columns=["F1", "F2", "F3", "F4", "F5", "F6"]
        )

        # 4. Call the function requesting Top 2
        # Since it extracts Top 2 positive and Top 2 negative, it should draw 4 bars total.
        top_n_test = 2
        plot_logistic_regression_coefficients(mock_model, X_fake, top_n=top_n_test)

        # 5. Extract the plot from memory
        ax = plt.gca()

        # 6. Text verifications
        self.assertEqual(ax.get_title(), "Top Logistic Regression Coefficients")
        self.assertEqual(ax.get_xlabel(), "Coefficient Value")

        # 7. Visual mathematical logic verification
        bars = ax.patches
        self.assertEqual(
            len(bars),
            top_n_test * 2,
            "There should be 4 bars (2 positive + 2 negative)"
        )

        # Confirm that the graphical window was intercepted
        mock_show.assert_called_once()

    @patch("builtins.print")
    @patch("matplotlib.pyplot.show")
    def test_early_exit_for_models_without_coefficients(self, mock_show, mock_print):
        """Test the safety system if the model lacks coefficients."""

        # Create an empty fake class so hasattr() naturally returns False
        class ClassifierWithoutCoefficients:
            pass

        # Construct the fake pipeline
        mock_model = MagicMock()
        mock_model.named_steps = {"classifier": ClassifierWithoutCoefficients()}

        X_fake = pd.DataFrame(columns=["feat_1", "feat_2"])

        # Call the function
        plot_logistic_regression_coefficients(mock_model, X_fake)

        # Verify that the operation was correctly aborted
        mock_print.assert_called_once_with("This model does not have coefficients.")
        mock_show.assert_not_called()

# ---------------------------------------------------------
# Tests for the plot_train_vs_test_accuracy_rf() function
# ---------------------------------------------------------

class TestPlotTrainVsTestAccuracy(unittest.TestCase):

    def tearDown(self):
        """Clean up graphical memory after the test."""
        plt.close('all')

    @patch("matplotlib.pyplot.show")
    def test_plot_generates_train_and_test_lines(self, mock_show):
        """Test that the loop trains the models and draws both accuracy lines."""

        # 1. Create a Micro-Dataset (4 rows to train, 2 to test)
        # This is small enough for Random Forest to process in a millisecond
        X_train_fake = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
        y_train_fake = np.array([0, 1, 0, 1])

        X_test_fake = np.array([[0.2, 0.3], [0.6, 0.7]])
        y_test_fake = np.array([0, 1])

        # 2. Call the function
        plot_train_vs_test_accuracy_rf(
            X_train_fake, X_test_fake,
            y_train_fake, y_test_fake,
            random_state=42
        )

        # 3. Extract the plot from memory
        ax = plt.gca()

        # 4. Text and label verifications
        self.assertEqual(ax.get_title(), "Random Forest Train vs Test Accuracy")
        self.assertEqual(ax.get_xlabel(), "Number of Trees")
        self.assertEqual(ax.get_ylabel(), "Accuracy")

        # 5. Check that the legend is activated (Train Accuracy / Test Accuracy)
        self.assertIsNotNone(ax.get_legend(), "The plot does not have the legend activated.")

        # 6. Visual verification
        # Matplotlib stores line plots inside the 'lines' property.
        # Since plt.plot() was called twice, exactly 2 lines should exist.
        drawn_lines = ax.lines
        self.assertEqual(len(drawn_lines), 2, "Exactly 2 lines should have been drawn.")

        # Optional: Verify that each line has 8 points (one for each value in n_trees_list)
        train_line_points = drawn_lines[0].get_xdata()
        self.assertEqual(len(train_line_points), 8)

        # 7. Verify that the graphical window was blocked
        mock_show.assert_called_once()

# ---------------------------------------------------------
# Tests for the plot_roc_auc() function
# ---------------------------------------------------------

class TestPlotRocAuc(unittest.TestCase):

    def tearDown(self):
        """Clean up graphical memory after the test."""
        plt.close('all')

    @patch("matplotlib.pyplot.show")
    def test_plot_roc_auc_calculates_and_draws_correctly(self, mock_show):
        """Test the happy path: AUC is calculated correctly and the plot is drawn."""

        # 1. Configure a PERFECT fake model
        mock_model = MagicMock()
        mock_model.classes_ = ["control", "anorexia"]

        # Simulate probabilities.
        # Column 0: control probability. Column 1: anorexia probability.
        # Make the model highly confident in its correct predictions.
        mock_model.predict_proba.return_value = np.array([
            [0.1, 0.9],  # Row 0: 90% anorexia (Real: anorexia)
            [0.8, 0.2],  # Row 1: 80% control  (Real: control)
            [0.2, 0.8],  # Row 2: 80% anorexia (Real: anorexia)
            [0.9, 0.1]  # Row 3: 90% control  (Real: control)
        ])

        # 2. Set up fake data
        X_test_fake = pd.DataFrame({"feat": [1, 2, 3, 4]})
        y_test_fake = pd.Series(["anorexia", "control", "anorexia", "control"])

        # 3. Call the function
        auc_score = plot_roc_auc(
            model=mock_model,
            X_test=X_test_fake,
            y_test=y_test_fake,
            positive_label="anorexia",
            title="ROC Test"
        )

        # 4. Math verifications
        # Since the fake model separated classes perfectly, the AUC must be 1.0
        self.assertEqual(auc_score, 1.0)

        # 5. Graphic verifications
        ax = plt.gca()
        self.assertEqual(ax.get_title(), "ROC Test")
        self.assertEqual(ax.get_xlabel(), "False Positive Rate")
        self.assertEqual(ax.get_ylabel(), "True Positive Rate / Recall")

        # Two lines should be drawn: the ROC curve and the baseline dotted line
        lines = ax.lines
        self.assertEqual(len(lines), 2)

        # Verify that plt.show() was intercepted
        mock_show.assert_called_once()

    def test_model_without_predict_proba_raises_error(self):
        """Test that the function rejects models that do not support probabilities."""

        # Create a basic class lacking the predict_proba method
        class InvalidModel:
            pass

        model = InvalidModel()

        with self.assertRaises(ValueError) as context:
            plot_roc_auc(model, None, None)

        self.assertIn("does not support predict_proba()", str(context.exception))

    def test_missing_positive_label_raises_error(self):
        """Test that the function fails if the 'positive_label' does not exist in the model classes."""

        mock_model = MagicMock()
        mock_model.classes_ = ["cat", "dog"]  # 'anorexia' does not exist
        mock_model.predict_proba.return_value = np.array([[0.5, 0.5]])

        with self.assertRaises(ValueError) as context:
            plot_roc_auc(
                model=mock_model,
                X_test=pd.DataFrame(),
                y_test=pd.Series(),
                positive_label="anorexia"
            )

        self.assertIn("not found in model classes", str(context.exception))

# ---------------------------------------------------------
# Tests for the calculate_auc() function
# ---------------------------------------------------------

class TestCalculateAuc(unittest.TestCase):

    def test_calculate_auc_standard_model(self):
        """Test that the AUC is calculated correctly for a standard model."""

        # 1. Configure a perfect fake model
        mock_model = MagicMock()
        mock_model.classes_ = ["control", "anorexia"]

        # Probabilities: [Prob_Control, Prob_Anorexia]
        mock_model.predict_proba.return_value = np.array([
            [0.1, 0.9],  # Real: anorexia
            [0.8, 0.2]  # Real: control
        ])

        X_fake = pd.DataFrame({"feat": [1, 2]})
        y_fake = pd.Series(["anorexia", "control"])

        # 2. Execute the function
        auc_score = calculate_auc(mock_model, X_fake, y_fake, positive_label="anorexia")

        # 3. Verification
        # A perfect separation yields an AUC of 1.0
        self.assertEqual(auc_score, 1.0)
        # Check that predict_proba is called with the correct data
        mock_model.predict_proba.assert_called_once_with(X_fake)

    def test_calculate_auc_pipeline_model(self):
        """Test that the function extracts classes correctly if the model is a Pipeline."""

        # 1. Configure a fake model to act as a Pipeline
        mock_pipeline = MagicMock()

        # Delete the top-level 'classes_' attribute to force the 'else' block
        del mock_pipeline.classes_

        # Configure the internal classifier
        mock_classifier = MagicMock()
        mock_classifier.classes_ = ["control", "anorexia"]
        mock_pipeline.named_steps = {"classifier": mock_classifier}

        mock_pipeline.predict_proba.return_value = np.array([
            [0.1, 0.9],
            [0.8, 0.2]
        ])

        X_fake = pd.DataFrame({"feat": [1, 2]})
        y_fake = pd.Series(["anorexia", "control"])

        # 2. Execute
        auc_score = calculate_auc(mock_pipeline, X_fake, y_fake)

        # 3. Verification
        self.assertEqual(auc_score, 1.0)

    def test_model_without_predict_proba_raises_error(self):
        """Test the safety validation for models without predict_proba."""

        class InvalidModel:
            pass

        with self.assertRaises(ValueError) as context:
            calculate_auc(InvalidModel(), None, None)

        self.assertIn("does not support predict_proba()", str(context.exception))

    def test_missing_positive_label_raises_error(self):
        """Test the safety validation for missing labels."""
        mock_model = MagicMock()
        mock_model.classes_ = ["healthy", "ill"]  # "anorexia" does not exist
        mock_model.predict_proba.return_value = np.array([[0.5, 0.5]])

        with self.assertRaises(ValueError) as context:
            calculate_auc(mock_model, None, None, positive_label="anorexia")

        self.assertIn("not found in model classes", str(context.exception))


if __name__ == '__main__':
    unittest.main()