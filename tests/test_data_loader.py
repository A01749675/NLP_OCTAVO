import unittest
import pandas as pd
import os
import tempfile

from data_loader import get_data

class TestGetData(unittest.TestCase):

    def setUp(self):
        """Runs BEFORE each test. Creates a clean temporary directory and file path."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.vectorized_csv = os.path.join(self.test_dir.name, "vectorized_test.csv")

    def tearDown(self):
        """Runs AFTER each test. Deletes the temporary files and folder."""
        self.test_dir.cleanup()

    def test_successful_data_extraction(self):
        """Test that features (X) and target (y) are extracted and filtered correctly."""
        # 1. Create mock data including columns to drop, target, numeric features, and string features
        df_mock = pd.DataFrame({
            "tweet_id": [101, 102, 103],
            "user_id": [1, 2, 3],
            "tweet_text": ["text a", "text b", "text c"],
            "class": [0, 1, 0],  # Target column
            "numeric_feat_1": [0.5, 0.8, 0.2],  # Valid feature (float)
            "numeric_feat_2": [10, 20, 30],  # Valid feature (int)
            "string_feat": ["a", "b", "c"]  # Invalid feature (not numeric)
        })
        df_mock.to_csv(self.vectorized_csv, index=False, encoding="utf-8")

        # 2. Run the function
        X, y = get_data(self.vectorized_csv)

        # 3. Assertions for X (Features)
        # It should drop the specified columns AND drop 'string_feat' because it's not a number
        expected_x_columns = ["numeric_feat_1", "numeric_feat_2"]
        self.assertEqual(list(X.columns), expected_x_columns)
        self.assertEqual(X.shape, (3, 2))  # 3 rows, 2 valid feature columns

        # 4. Assertions for y (Target)
        self.assertEqual(list(y), [0, 1, 0])

    def test_missing_class_raises_error(self):
        """Test that a ValueError is raised if the 'class' column is missing."""
        df_mock = pd.DataFrame({
            "tweet_id": [101, 102],
            "numeric_feat": [0.5, 0.8]
            # 'class' column is intentionally missing
        })
        df_mock.to_csv(self.vectorized_csv, index=False, encoding="utf-8")

        with self.assertRaises(ValueError) as context:
            get_data(self.vectorized_csv)

        self.assertIn("must contain a 'class' column", str(context.exception))

    def test_no_numeric_features_raises_error(self):
        """Test that a ValueError is raised if there are no numeric features left to train on."""
        df_mock = pd.DataFrame({
            "tweet_id": [101, 102],
            "class": [1, 0],
            "tweet_text": ["hello", "world"],
            "string_feat": ["abc", "def"]
            # No numeric columns are present outside of the ones explicitly dropped
        })
        df_mock.to_csv(self.vectorized_csv, index=False, encoding="utf-8")

        with self.assertRaises(ValueError) as context:
            get_data(self.vectorized_csv)

        self.assertIn("No numeric feature columns were found", str(context.exception))


if __name__ == '__main__':
    unittest.main()