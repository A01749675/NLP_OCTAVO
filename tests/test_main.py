import unittest
from unittest.mock import patch, MagicMock

from main import select_model, get_test_size, test_knn_model, save_model, train_and_plot, run_experiments

# ---------------------------------------------------------
# Tests for the select_model() function
# ---------------------------------------------------------

class TestSelectModel(unittest.TestCase):

    @patch("main.get_random_forest")
    def test_select_random_forest(self, mock_get_rf):
        """Tests the routing for Random Forest using all its possible aliases."""
        mock_get_rf.return_value = "Mocked_RF_Model"
        aliases = ["rf", "random_forest", "Random Forest", "RF", "Random_Forest"]

        for alias in aliases:
            model = select_model(alias, random_state=42)

            self.assertEqual(model, "Mocked_RF_Model")
            mock_get_rf.assert_called_with(random_state=42)
            mock_get_rf.reset_mock()

    @patch("main.get_logistic_regression")
    def test_select_logistic_regression(self, mock_get_lr):
        """Tests the routing for Logistic Regression using its aliases."""
        mock_get_lr.return_value = "Mocked_LR_Model"
        aliases = ["lr", "logistic_regression", "logistic regression", "LR"]

        for alias in aliases:
            model = select_model(alias, random_state=99)

            self.assertEqual(model, "Mocked_LR_Model")
            mock_get_lr.assert_called_with(random_state=99)
            mock_get_lr.reset_mock()

    @patch("main.get_knn")
    def test_select_knn(self, mock_get_knn):
        """Tests the routing for KNN using its aliases."""
        mock_get_knn.return_value = "Mocked_KNN_Model"
        aliases = ["knn", "k_nearest_neighbors", "k nearest neighbors", "KNN"]

        for alias in aliases:
            model = select_model(alias)

            self.assertEqual(model, "Mocked_KNN_Model")
            mock_get_knn.assert_called_with()
            mock_get_knn.reset_mock()

    def test_invalid_model_raises_error(self):
        """Tests that requesting a non-existent model raises a ValueError."""
        with self.assertRaises(ValueError) as context:
            select_model("red_neuronal_inventada")

        self.assertIn("Invalid model_name", str(context.exception))
        self.assertIn("'rf', 'lr', or 'knn'", str(context.exception))

# ---------------------------------------------------------
# Tests for the get_test_size() function
# ---------------------------------------------------------

class TestGetTestSize(unittest.TestCase):

    def test_random_forest_size(self):
        """Tests that Random Forest aliases return 0.20 (80/20 split)."""
        aliases = ["rf", "random_forest", "Random Forest", "RF"]

        for alias in aliases:
            size = get_test_size(alias)
            self.assertEqual(size, 0.20, f"Failed for Random Forest alias: '{alias}'")

    def test_other_models_size(self):
        """Tests that other models (LR, KNN) return 0.20 (80/20 split)."""
        aliases = ["lr", "logistic_regression", "knn", "K nearest neighbors", "anything_else"]

        for alias in aliases:
            size = get_test_size(alias)
            self.assertEqual(size, 0.20, f"Failed for other model alias: '{alias}'")

# ---------------------------------------------------------
# Tests for the test_knn_model() function
# ---------------------------------------------------------

class TestKnnModelExperiment(unittest.TestCase):

    # The order of the arguments in the function goes from BOTTOM to TOP with respect to the @patch decorators
    @patch("main.process_csv")
    @patch("main.get_data")
    @patch("main.get_test_size")
    @patch("main.train_test_split")
    @patch("main.get_knn")
    @patch("main.evaluate_model")
    @patch("main.print_metrics")
    @patch("pandas.DataFrame.to_csv")
    def test_test_knn_model_orchestration(
            self,
            mock_to_csv,
            mock_print_metrics,
            mock_evaluate_model,
            mock_get_knn,
            mock_train_test_split,
            mock_get_test_size,
            mock_get_data,
            mock_process_csv
    ):
        """Tests that the experimentation loop works without training real models."""

        # 1. Configure the mock return values
        mock_process_csv.return_value = "dummy_file.csv"
        mock_get_data.return_value = ("X_dummy", "y_dummy")
        mock_get_test_size.return_value = 0.20
        mock_train_test_split.return_value = ("X_train", "X_test", "y_train", "y_test")

        # Simulates that get_knn returns an object with a .fit() method that does nothing
        mock_model = MagicMock()
        mock_get_knn.return_value = mock_model

        # Simulates the results of evaluate_model (returns predictions and a metrics dict)
        fake_metrics = {"accuracy": 0.95, "f1_score": 0.92}
        mock_evaluate_model.return_value = (["fake_preds"], fake_metrics)

        # 2. Execute the function
        performance = test_knn_model(input_file="test.csv")

        # 3. Orchestrator logic verifications

        # There are 4 vectorizers and 9 values of K (4 * 9 = 36 iterations)
        self.assertEqual(len(performance), 36)

        # Verify that process_csv was called exactly 4 times
        self.assertEqual(mock_process_csv.call_count, 4)

        # Check that the data was correctly added to the list
        # The first element should be TF-IDF with 1 neighbor
        self.assertEqual(performance[0]["representation"], "tfidf")
        self.assertEqual(performance[0]["n_neighbors"], 1)
        self.assertEqual(performance[0]["accuracy"], 0.95)

        # Verify that the final file attempt to save occurred
        mock_to_csv.assert_called_once_with('knn_performance.csv', index=False)

# ---------------------------------------------------------
# Tests for the save_model() function
# ---------------------------------------------------------

class TestSaveModel(unittest.TestCase):

    @patch("main.joblib.dump")
    def test_save_model_creates_correct_filename(self, mock_dump):
        """Tests that the filename is correctly concatenated and joblib is called."""

        # Create a fake model
        mock_model = MagicMock()

        # Call function
        save_model(model=mock_model, representation="tfidf", target="knn")

        # Check that joblib.dump attempted to execute with the model and the exact name
        mock_dump.assert_called_once_with(mock_model, "tfidf-knn.pkl")

# ---------------------------------------------------------
# Tests for the train_and_plot() function
# ---------------------------------------------------------

class TestTrainAndPlot(unittest.TestCase):

    def test_invalid_vectorized_file_raises_error(self):
        """Tests that the orchestrator stops if process_csv fails (returns None or empty)."""
        with patch("main.process_csv", return_value=""):
            with self.assertRaises(ValueError) as context:
                train_and_plot(input_file="dummy.csv", target="tfidf", model_name="rf")

            self.assertIn("did not return a valid vectorized file name", str(context.exception))

    @patch("main.process_csv")
    @patch("main.get_data")
    @patch("main.get_test_size")
    @patch("main.train_test_split")
    @patch("main.select_model")
    @patch("main.save_model")
    @patch("main.evaluate_model")
    @patch("main.calculate_auc")
    @patch("main.print_metrics")
    def test_full_pipeline_execution_without_plots(
            self,
            mock_print, mock_calc_auc, mock_eval, mock_save,
            mock_select, mock_split, mock_get_size, mock_get_data, mock_process
    ):
        """Tests the full execution of the base pipeline assuming PLOTS_ACTIVE = False."""

        # 1. Configure mock return values
        mock_process.return_value = "vectorized_dummy.csv"
        mock_get_data.return_value = ("X_dummy", "y_dummy")
        mock_get_size.return_value = 0.30
        mock_split.return_value = ("X_train", "X_test", "y_train", "y_test")

        # Simulate the trained model
        mock_model = MagicMock()
        mock_select.return_value = mock_model

        # Simulate evaluation
        mock_eval.return_value = (["fake_preds"], {"accuracy": 0.90})
        mock_calc_auc.return_value = 0.85

        # 2. Temporarily disable plots
        with patch("main.PLOTS_ACTIVE", False):
            model, results = train_and_plot(
                input_file="input.csv",
                target="tfidf",
                model_name="rf"
            )

        # 3. Verifications that everything was called in order
        mock_process.assert_called_once_with(input_file="input.csv", target="tfidf")
        mock_get_data.assert_called_once_with("vectorized_dummy.csv")
        mock_select.assert_called_once_with(model_name="rf", random_state=42)

        # Verify that the model was trained and saved
        mock_model.fit.assert_called_once_with("X_train", "y_train")
        mock_save.assert_called_once_with(mock_model, "rf", "tfidf")

        # Verify that the results were calculated correctly
        self.assertEqual(results["accuracy"], 0.90)
        self.assertEqual(results["test_size"], 0.30)
        self.assertEqual(results["auc"], 0.85)

    @patch("main.process_csv", return_value="dummy.csv")
    @patch("main.get_data", return_value=("X", "y"))
    @patch("main.get_test_size", return_value=0.2)
    @patch("main.train_test_split", return_value=("X_tr", "X_te", "y_tr", "y_te"))
    @patch("main.select_model")
    @patch("main.save_model")
    @patch("main.evaluate_model", return_value=(["preds"], {"acc": 0.8}))
    @patch("main.calculate_auc", return_value=0.8)
    @patch("main.print_metrics")
    @patch("main.plot_class_distribution")
    @patch("main.plot_confusion_matrix")
    @patch("main.plot_roc_auc", return_value=0.85)
    @patch("main.plot_random_forest_feature_importance")
    @patch("main.plot_train_vs_test_accuracy_rf")
    def test_pipeline_with_plots_random_forest(
            self,
            mock_plot_dist, mock_plot_cm, mock_plot_roc, mock_plot_feat, mock_plot_acc,
            mock_print, mock_calc_auc, mock_eval, mock_save, mock_select,
            mock_split, mock_get_size, mock_get_data, mock_process
    ):
        """Tests that if PLOTS_ACTIVE is True and the model is 'rf', its specific plots are called."""
        mock_model = MagicMock()
        mock_select.return_value = mock_model

        # Mock return values for required pipeline steps
        mock_process.return_value = "vectorized.csv"
        mock_get_data.return_value = ("X", "y")
        mock_split.return_value = ("X_train", "X_test", "y_train", "y_test")
        mock_eval.return_value = (["preds"], {"accuracy": 0.90})

        # Turn on plots
        with patch("main.PLOTS_ACTIVE", True):
            model, results = train_and_plot(model_name="rf")

        # Verify that general plots were called
        mock_plot_dist.assert_called_once()
        mock_plot_cm.assert_called_once()
        mock_plot_roc.assert_called_once()

        # Verify that Random Forest SPECIFIC plots were called
        mock_plot_feat.assert_called_once()
        mock_plot_acc.assert_called_once()

        # The AUC value should be updated with what plot_roc_auc returns
        self.assertEqual(results["auc"], 0.85)

# ---------------------------------------------------------
# Tests for the run_experiments() function
# ---------------------------------------------------------

class TestRunExperiments(unittest.TestCase):

    @patch("pandas.DataFrame.to_csv")
    @patch("main.train_and_plot")
    @patch("builtins.print")  # Prevents the console from filling with prints during the test
    def test_run_experiments_orchestration(self, mock_print, mock_train_and_plot, mock_to_csv):
        """Tests that the experimentation loop runs 15 times and saves the results."""

        # 1. Configure the mock values
        # train_and_plot returns a tuple: (model, results_dictionary)
        fake_model = MagicMock()
        fake_results = {"accuracy": 0.88, "f1_score": 0.85}
        mock_train_and_plot.return_value = (fake_model, fake_results)

        # 2. Execute the function
        all_results = run_experiments()

        # 3. Verifications
        self.assertEqual(len(all_results), 15)

        # Check that train_and_plot was called exactly 15 times
        self.assertEqual(mock_train_and_plot.call_count, 15)

        # Check the structure of the first saved result
        first_result = all_results[0]
        self.assertEqual(first_result["representation"], "tfidf")
        self.assertEqual(first_result["model"], "rf")
        self.assertEqual(first_result["accuracy"], 0.88)  # The mock value we injected

        # Check that at the end of the loop, the CSV was saved
        mock_to_csv.assert_called_once_with('all_experiments.csv')

if __name__ == '__main__':
    unittest.main()