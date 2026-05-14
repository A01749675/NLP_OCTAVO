import unittest
import os
import tempfile
from unittest.mock import patch
import pandas as pd

from vectorizers import tfidf_vectorize, ngram_vectorize, word2vec_vectorize, tfidf_ngram_vectorize, all_vectorize, process_csv

# ---------------------------------------------------------
# Tests for the tfidf_vectorize() function
# ---------------------------------------------------------

class TestTfidfVectorize(unittest.TestCase):

    def setUp(self):
        """Runs BEFORE each test. Creates a clean temporary directory."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.output_csv = os.path.join(self.test_dir.name, "test_tfidf.csv")

    def tearDown(self):
        """Runs AFTER each test. Deletes the temporary files and folder."""
        self.test_dir.cleanup()

    def test_basic_vectorization(self):
        """Test vectorization with all parameters provided (texts, ids, classes)."""
        texts = ["hola mundo", "hola perro"]
        tweet_ids = [101, 102]
        classes = [0, 1]

        # Run function
        df = tfidf_vectorize(
            texts=texts,
            tweet_ids=tweet_ids,
            classes=classes,
            output_file=self.output_csv
        )

        # 1. Check if file was created
        self.assertTrue(os.path.exists(self.output_csv))

        # 2. Check DataFrame structure
        # Expected features: tfidf_hola, tfidf_mundo, tfidf_perro
        # Expected metadata: class, tweet_id, tweet_text_clean
        self.assertEqual(len(df), 2)

        expected_columns = [
            "class", "tweet_id", "tweet_text_clean",
            "tfidf_hola", "tfidf_mundo", "tfidf_perro"
        ]

        # Verify all expected columns are present
        self.assertTrue(set(expected_columns).issubset(set(df.columns)))

        # 3. Check values
        self.assertEqual(df.loc[0, "tweet_text_clean"], "hola mundo")
        self.assertEqual(df.loc[1, "tweet_id"], 102)

    def test_optional_parameters_omitted(self):
        """Test that the function works correctly when tweet_ids and classes are None."""
        texts = ["gato feliz", "perro triste"]

        df = tfidf_vectorize(
            texts=texts,
            tweet_ids=None,
            classes=None,
            output_file=self.output_csv
        )

        # Check that the optional columns were NOT added
        self.assertNotIn("class", df.columns)
        self.assertNotIn("tweet_id", df.columns)

        # The text column should still be there
        self.assertIn("tweet_text_clean", df.columns)

    def test_ngram_range(self):
        """Test that the ngram_range parameter correctly extracts bigrams."""
        texts = ["muy buen dia"]

        # Run with bigrams enabled (1, 2)
        df = tfidf_vectorize(
            texts=texts,
            tweet_ids=None,
            classes=None,
            output_file=self.output_csv,
            ngram_range=(1, 2)
        )

        # It should extract unigrams ('muy', 'buen', 'dia') AND bigrams ('muy buen', 'buen dia')
        # A column called 'tfidf_muy buen' is expected
        self.assertIn("tfidf_muy buen", df.columns)
        self.assertIn("tfidf_buen dia", df.columns)

# ---------------------------------------------------------
# Tests for the ngram_vectorize() function
# ---------------------------------------------------------

class TestNgramVectorize(unittest.TestCase):

    def setUp(self):
        """Runs BEFORE each test. Creates a clean temporary directory."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.output_csv = os.path.join(self.test_dir.name, "test_ngrams.csv")

    def tearDown(self):
        """Runs AFTER each test. Deletes the temporary files and folder."""
        self.test_dir.cleanup()

    def test_basic_vectorization(self):
        """Test n-gram generation with all parameters provided."""
        texts = ["gato negro", "perro blanco"]
        tweet_ids = [1, 2]
        classes = [0, 1]

        # Run function with default ngram_range (1, 3)
        df = ngram_vectorize(
            texts=texts,
            tweet_ids=tweet_ids,
            classes=classes,
            output_file=self.output_csv
        )

        # 1. Check if file was created
        self.assertTrue(os.path.exists(self.output_csv))

        # 2. Check DataFrame structure
        self.assertEqual(len(df), 2)

        # Expected metadata columns
        self.assertIn("class", df.columns)
        self.assertIn("tweet_id", df.columns)
        self.assertIn("tweet_text_clean", df.columns)

        # Expected n-gram features (unigrams and bigrams from the text)
        expected_ngrams = [
            "ngram_gato", "ngram_negro", "ngram_gato negro",
            "ngram_perro", "ngram_blanco", "ngram_perro blanco"
        ]

        # Verify all expected n-gram columns are present
        self.assertTrue(set(expected_ngrams).issubset(set(df.columns)))

        # 3. Check values mapping correctly
        self.assertEqual(df.loc[0, "tweet_text_clean"], "gato negro")
        self.assertEqual(df.loc[1, "tweet_id"], 2)

    def test_optional_parameters_omitted(self):
        """Test that the function works correctly when tweet_ids and classes are None."""
        texts = ["sol brillante", "luna llena"]

        df = ngram_vectorize(
            texts=texts,
            tweet_ids=None,
            classes=None,
            output_file=self.output_csv
        )

        # Check that the optional metadata columns were NOT added
        self.assertNotIn("class", df.columns)
        self.assertNotIn("tweet_id", df.columns)

        # The text column should still be there
        self.assertIn("tweet_text_clean", df.columns)

    def test_ngram_range(self):
        """Test that the ngram_range parameter accurately restricts/allows n-grams."""
        texts = ["un dia muy feliz"]

        # Run with trigrams enabled explicitly (1, 3)
        df = ngram_vectorize(
            texts=texts,
            tweet_ids=None,
            classes=None,
            output_file=self.output_csv,
            ngram_range=(1, 3)
        )

        # Unigram
        self.assertIn("ngram_dia", df.columns)
        # Bigram
        self.assertIn("ngram_dia muy", df.columns)
        # Trigram
        self.assertIn("ngram_dia muy feliz", df.columns)

        # Ensure it doesn't create a 4-gram (which would be the whole string)
        self.assertNotIn("ngram_un dia muy feliz", df.columns)


# ---------------------------------------------------------
# Tests for the word2vec_vectorize() function
# ---------------------------------------------------------

class TestWord2VecVectorize(unittest.TestCase):

    def setUp(self):
        """Runs BEFORE each test."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.output_csv = os.path.join(self.test_dir.name, "test_word2vec.csv")

        # Physically create the file that the function expects.
        self.txt_filename = "word2vecText.txt"
        self.file_already_existed = os.path.exists(self.txt_filename)

        # BACKUP: If a real file with that name already existed, save its content
        if self.file_already_existed:
            with open(self.txt_filename, "r", encoding="utf-8") as f:
                self.backup_content = f.read()

        # Write the test content for Word2Vec to read
        with open(self.txt_filename, "w", encoding="utf-8") as f:
            f.write("anorexia salud dieta peso")

    def tearDown(self):
        """Runs AFTER each test."""
        self.test_dir.cleanup()

        # RESTORATION: Delete the test file, or restore yours if it existed
        if self.file_already_existed:
            with open(self.txt_filename, "w", encoding="utf-8") as f:
                f.write(self.backup_content)
        else:
            if os.path.exists(self.txt_filename):
                os.remove(self.txt_filename)

    def test_basic_vectorization(self):
        """Tests basic vector creation with all parameters."""
        texts = ["hola mundo", "dieta saludable"]
        tweet_ids = [101, 102]
        classes = [0, 1]
        v_size = 5  # Small size to make the test instantaneous

        df = word2vec_vectorize(
            texts=texts,
            tweet_ids=tweet_ids,
            classes=classes,
            output_file=self.output_csv,
            vector_size=v_size,
            epochs=1
        )

        # Verify that the CSV was created
        self.assertTrue(os.path.exists(self.output_csv))

        # Check column structure
        self.assertEqual(len(df), 2)
        self.assertIn("class", df.columns)
        self.assertIn("tweet_id", df.columns)
        self.assertIn("tweet_text_clean", df.columns)

        # Check that exactly 'v_size' vector columns were created
        expected_vector_cols = [f"word2vec_{i}" for i in range(v_size)]
        self.assertTrue(set(expected_vector_cols).issubset(set(df.columns)))

    def test_optional_params_and_fallback(self):
        """
        Tests what happens without optional parameters and what happens if a text
        becomes empty after cleaning (it should be filled with zeros).
        """
        texts = ["hola normal", "!!!"]
        v_size = 3

        df = word2vec_vectorize(
            texts=texts,
            tweet_ids=None,
            classes=None,
            output_file=self.output_csv,
            vector_size=v_size,
            epochs=1
        )

        self.assertNotIn("class", df.columns)
        self.assertNotIn("tweet_id", df.columns)

        # Verify that row 1 (the one with "!!!") triggered the np.zeros() fallback
        fila_vacia = df.iloc[1]
        for i in range(v_size):
            self.assertEqual(fila_vacia[f"word2vec_{i}"], 0.0)

# ---------------------------------------------------------
# Tests for the tfidf_ngram_vectorize() function
# ---------------------------------------------------------

class TestTfidfNgramVectorize(unittest.TestCase):

    def setUp(self):
        """Runs BEFORE each test. Creates a clean temporary environment."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.output_csv = os.path.join(self.test_dir.name, "test_combined.csv")

    def tearDown(self):
        """Runs AFTER each test. Deletes generated files."""
        self.test_dir.cleanup()

        archivos_temporales = ["temporary_tfidf.csv", "temporary_ngrams.csv"]
        for archivo in archivos_temporales:
            if os.path.exists(archivo):
                os.remove(archivo)

    def test_combined_vectorization(self):
        """Tests the integration of both vectorizations with all parameters."""
        # Texts with enough words to test n-grams
        texts = ["un gato negro", "un perro blanco"]
        tweet_ids = [101, 102]
        classes = [0, 1]

        # Execute with Unigrams for TF-IDF (1,1) and Bigrams for Count (2,2)
        df = tfidf_ngram_vectorize(
            texts=texts,
            tweet_ids=tweet_ids,
            classes=classes,
            output_file=self.output_csv,
            tfidf_ngram_range=(1, 1),
            count_ngram_range=(2, 2)
        )

        # 1. Verify final file creation
        self.assertTrue(os.path.exists(self.output_csv))

        # 2. Check column structure (metadata)
        self.assertEqual(len(df), 2)
        self.assertIn("class", df.columns)
        self.assertIn("tweet_id", df.columns)
        self.assertIn("tweet_text_clean", df.columns)

        # 3. Check TF-IDF features (must have the tfidf_ prefix)
        self.assertIn("tfidf_gato", df.columns)
        self.assertIn("tfidf_perro", df.columns)

        # 4. Check N-gram features (must have the ngram_ prefix)
        # Using range (2,2) extracts bigrams
        self.assertIn("ngram_un gato", df.columns)
        self.assertIn("ngram_perro blanco", df.columns)

    def test_optional_parameters_omitted(self):
        """Tests that the function properly handles the absence of IDs and classes."""
        texts = ["hola mundo feliz", "adios mundo cruel"]

        df = tfidf_ngram_vectorize(
            texts=texts,
            tweet_ids=None,
            classes=None,
            output_file=self.output_csv,
            tfidf_ngram_range=(1, 1),
            count_ngram_range=(2, 2)
        )

        # Verify that optional columns do not exist
        self.assertNotIn("class", df.columns)
        self.assertNotIn("tweet_id", df.columns)

        # The text and combined features should still be there
        self.assertIn("tweet_text_clean", df.columns)
        self.assertIn("tfidf_hola", df.columns)
        self.assertIn("ngram_mundo cruel", df.columns)

# ---------------------------------------------------------
# Tests for the all_vectorize() function
# ---------------------------------------------------------

class TestAllVectorize(unittest.TestCase):

    def setUp(self):
        """Runs BEFORE each test. Creates a clean temporary environment."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.output_csv = os.path.join(self.test_dir.name, "test_all.csv")

        # Temporarily create the file that Word2Vec needs
        self.txt_filename = "word2vecText.txt"
        self.file_already_existed = os.path.exists(self.txt_filename)

        # Backup if the original file already existed
        if self.file_already_existed:
            with open(self.txt_filename, "r", encoding="utf-8") as f:
                self.backup_content = f.read()

        # Write a mock text
        with open(self.txt_filename, "w", encoding="utf-8") as f:
            f.write("anorexia salud dieta peso")

    def tearDown(self):
        """Runs AFTER each test. Deletes generated files and restores the environment."""
        self.test_dir.cleanup()

        # Restoration of the original txt
        if self.file_already_existed:
            with open(self.txt_filename, "w", encoding="utf-8") as f:
                f.write(self.backup_content)
        else:
            if os.path.exists(self.txt_filename):
                os.remove(self.txt_filename)

        # Cleanup of the hardcoded file generated by your internal function
        if os.path.exists("temporary_word2vec.csv"):
            os.remove("temporary_word2vec.csv")

    def test_combined_vectorization_all_features(self):
        """Tests the massive integration of TF-IDF, N-grams, and Word2Vec."""
        texts = ["un gato negro", "un perro blanco"]
        tweet_ids = [101, 102]
        classes = [0, 1]

        # Run the function
        df = all_vectorize(
            texts=texts,
            tweet_ids=tweet_ids,
            classes=classes,
            output_file=self.output_csv,
            tfidf_ngram_range=(1, 1),
            count_ngram_range=(2, 2)
        )

        # 1. Verify final file creation
        self.assertTrue(os.path.exists(self.output_csv))

        # 2. Check that metadata columns are present and in order
        self.assertEqual(len(df), 2)
        self.assertIn("class", df.columns)
        self.assertIn("tweet_id", df.columns)
        self.assertIn("tweet_text_clean", df.columns)

        # 3. Verify features generated by TF-IDF
        self.assertIn("tfidf_gato", df.columns)
        self.assertIn("tfidf_perro", df.columns)

        # 4. Verify features generated by CountVectorizer (N-grams)
        # Requested range (2, 2), so bigrams should exist
        self.assertIn("ngram_un gato", df.columns)
        self.assertIn("ngram_perro blanco", df.columns)

        # 5. Verify features generated by Word2Vec
        # Checks the first and last to ensure they were concatenated correctly
        self.assertIn("word2vec_0", df.columns)
        self.assertIn("word2vec_49", df.columns)

    def test_optional_parameters_omitted(self):
        """Tests that the pipeline does not fail if the IDs and classes are omitted."""
        texts = ["hola mundo", "adios universo"]

        df = all_vectorize(
            texts=texts,
            tweet_ids=None,
            classes=None,
            output_file=self.output_csv,
            tfidf_ngram_range=(1, 1),
            count_ngram_range=(1, 1)
        )

        # Confirm that they were not added
        self.assertNotIn("class", df.columns)
        self.assertNotIn("tweet_id", df.columns)

        # But the text and vectors must still be there
        self.assertIn("tweet_text_clean", df.columns)
        self.assertIn("tfidf_hola", df.columns)
        self.assertIn("ngram_universo", df.columns)
        self.assertIn("word2vec_0", df.columns)

# ---------------------------------------------------------
# Tests for the process_csv() function
# ---------------------------------------------------------

class TestProcessCSV(unittest.TestCase):

    def setUp(self):
        """Prepares a generic valid CSV file for routing tests."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.valid_input_csv = os.path.join(self.test_dir.name, "valid_input.csv")

        # Perfect CSV with all columns
        df_valid = pd.DataFrame({
            "tweet_id": [101, 102],
            "tweet_text_clean": ["texto uno", "texto dos"],
            "class": [0, 1]
        })
        df_valid.to_csv(self.valid_input_csv, index=False, encoding="utf-8")

    def tearDown(self):
        """Cleans up the temporary directory."""
        self.test_dir.cleanup()

    def test_missing_text_column_raises_error(self):
        """Tests that it fails if the 'tweet_text_clean' column is missing."""
        bad_csv = os.path.join(self.test_dir.name, "bad1.csv")
        pd.DataFrame({"class": [1, 0]}).to_csv(bad_csv, index=False)

        with self.assertRaises(ValueError) as context:
            process_csv(bad_csv, "tfidf")
        self.assertIn("must contain a 'tweet_text_clean' column", str(context.exception))

    def test_missing_class_column_raises_error(self):
        """Tests that it fails if the 'class' column is missing."""
        bad_csv = os.path.join(self.test_dir.name, "bad2.csv")
        pd.DataFrame({"tweet_text_clean": ["hola"]}).to_csv(bad_csv, index=False)

        with self.assertRaises(ValueError) as context:
            process_csv(bad_csv, "tfidf")
        self.assertIn("must contain a 'class' column", str(context.exception))

    def test_invalid_target_raises_error(self):
        """Tests that an unknown target raises an error in the default case (_)."""
        with self.assertRaises(ValueError) as context:
            process_csv(self.valid_input_csv, "vectorizador_falso")
        self.assertIn("Invalid target", str(context.exception))

    # @patch is used to intercept calls to the vectorizers.
    # The order of the mock_ arguments in the function is the REVERSE of the order of the decorators.
    @patch("vectorizers.tfidf_vectorize")
    def test_target_tfidf(self, mock_tfidf):
        """Tests that the 'tfidf' target calls the correct function."""
        file_name = process_csv(self.valid_input_csv, "tfidf")

        self.assertEqual(file_name, "data_train_tfidf.csv")
        mock_tfidf.assert_called_once()

        # Verify that it was called with the lists extracted from the CSV
        kwargs = mock_tfidf.call_args.kwargs
        self.assertEqual(kwargs['texts'], ["texto uno", "texto dos"])
        self.assertEqual(kwargs['classes'], [0, 1])

    @patch("vectorizers.ngram_vectorize")
    def test_target_ngrams(self, mock_ngrams):
        """Tests that the 'ngrams' target calls the correct function."""
        file_name = process_csv(self.valid_input_csv, "ngrams")
        self.assertEqual(file_name, "data_train_ngrams.csv")
        mock_ngrams.assert_called_once()

    @patch("vectorizers.word2vec_vectorize")
    def test_target_word2vec(self, mock_word2vec):
        """Tests that the 'word2vec' target calls the correct function without taking 100 epochs."""
        file_name = process_csv(self.valid_input_csv, "word2vec")
        self.assertEqual(file_name, "data_train_word2vec.csv")
        mock_word2vec.assert_called_once()

    @patch("vectorizers.all_vectorize")
    def test_target_all(self, mock_all):
        """Tests that the 'all' target calls the correct function."""
        file_name = process_csv(self.valid_input_csv, "all")
        self.assertEqual(file_name, "data_train_all.csv")
        mock_all.assert_called_once()

    @patch("vectorizers.tfidf_ngram_vectorize")
    def test_target_tfidf_ngrams(self, mock_tfidf_ngram):
        """Tests that the 'tfidf_ngrams' target calls the correct function."""
        file_name = process_csv(self.valid_input_csv, "tfidf_ngrams")
        self.assertEqual(file_name, "data_train_tfidf_ngrams.csv")
        mock_tfidf_ngram.assert_called_once()

    @patch("vectorizers.tfidf_vectorize")
    def test_optional_tweet_ids(self, mock_tfidf):
        """Tests the behavior if the input CSV does NOT have 'tweet_id'."""
        no_id_csv = os.path.join(self.test_dir.name, "no_id.csv")
        pd.DataFrame({
            "tweet_text_clean": ["sin id"],
            "class": [1]
        }).to_csv(no_id_csv, index=False)

        process_csv(no_id_csv, "tfidf")

        kwargs = mock_tfidf.call_args.kwargs
        self.assertIsNone(kwargs['tweet_ids'], "If there is no tweet_id column, None should be passed.")

if __name__ == '__main__':
    unittest.main()