import unittest
import pandas as pd
import numpy as np
import nltk
import os
import tempfile

from text_cleaner import clean_text, stopword_remover_nltk, text_stemming, text_lemmatization, text_stemming_pystemmer, text_filtering, process_csv

# ---------------------------------------------------------
# Tests for the clean_text() function
# ---------------------------------------------------------

class TestCleanText(unittest.TestCase):

    def test_missing_values(self):
        """Test that NaN values are handled and returned correctly."""
        self.assertTrue(pd.isna(clean_text(np.nan)))
        self.assertTrue(pd.isna(clean_text(pd.NA)))

    def test_lowercase(self):
        """Test that text is converted to lowercase."""
        self.assertEqual(clean_text("UPPERCASE TEXT"), "uppercase text")

    def test_remove_urls(self):
        """Test the regex that removes URLs."""
        self.assertEqual(clean_text("check http://example.com and www.test.com here"), "check and here")

    def test_remove_hashtags(self):
        """Test the regex that removes hashtags."""
        self.assertEqual(clean_text("coding #python #machinelearning today"), "coding today")

    def test_remove_numbers(self):
        """Test the regex that removes digits."""
        self.assertEqual(clean_text("i have 2 apples and 100 bananas"), "i have apples and bananas")

    def test_remove_mentions(self):
        """Test the regex that removes mentions."""
        self.assertEqual(clean_text("hello @user123 and @admin"), "hello and")

    def test_remove_punctuation(self):
        """Test string translation for punctuation removal."""
        self.assertEqual(clean_text("wait, what? yes!"), "wait what yes")

    def test_special_chars(self):
        """Test the regex (r"\W+") that removes non-word characters (symbols, emojis)."""
        # Note: \W+ removes things like ★ (\u2605) and replaces them with a space
        self.assertEqual(clean_text("stars \u2605 and heart \u2764"), "stars and heart")

    def test_extra_spaces(self):
        """Test the regex (r"\s+") that collapses tabs, newlines, and multiple spaces."""
        self.assertEqual(clean_text("  too   many \t spaces \n and tabs  "), "too many spaces and tabs")

    def test_combined(self):
        """Test all operations combined into a single messy string."""
        messy_text = "  WOW! @JohnDoe check out 100 cool tips on www.cool-tips.com #awesome \n\t \u2605  "
        expected_text = "wow check out cool tips on"
        self.assertEqual(clean_text(messy_text), expected_text)

# ---------------------------------------------------------
# Tests for the stopword_remover_nltk() function
# ---------------------------------------------------------

class TestStopwordRemoverNLTK(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Ensure required NLTK resources are downloaded before running tests."""
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)

    def test_remove_basic_stopwords(self):
        """Test the removal of common Spanish stopwords (el, de, la, etc.)."""
        text = "el perro de la casa es muy grande"
        expected = "perro casa grande"
        self.assertEqual(stopword_remover_nltk(text), expected)

    def test_no_stopwords(self):
        """Test that a string with no stopwords remains completely unchanged."""
        text = "computadora teclado monitor"
        expected = "computadora teclado monitor"
        self.assertEqual(stopword_remover_nltk(text), expected)

    def test_only_stopwords(self):
        """Test that a string composed entirely of stopwords returns an empty string."""
        text = "el la de que y en un una"
        expected = ""
        self.assertEqual(stopword_remover_nltk(text), expected)

    def test_empty_string(self):
        """Test that an empty string is handled safely."""
        self.assertEqual(stopword_remover_nltk(""), "")

    def test_case_sensitivity(self):
        """
        Test how the function handles capitalized stopwords.
        Note: Because the function does not explicitly call .lower(),
        capitalized stopwords (like 'El') will NOT be removed.
        """
        text = "El perro de la casa"
        expected = "El perro casa"
        self.assertEqual(stopword_remover_nltk(text), expected)

    def test_with_punctuation(self):
        """
        Test how the function behaves with punctuation.
        Note: word_tokenize separates punctuation into distinct tokens.
        """
        text = "perro, gato y ratón."
        expected = "perro , gato ratón ."
        self.assertEqual(stopword_remover_nltk(text), expected)

# ---------------------------------------------------------
# Tests for the text_stemming() function
# ---------------------------------------------------------

class TestTextStemming(unittest.TestCase):

    def test_basic_stemming(self):
        """Test stemming of standard Spanish verbs, plurals, and adverbs."""
        # Snowball typically stems: 'corriendo' -> 'corr', 'perros' -> 'perr', 'rápidamente' -> 'rapid'
        text = "corriendo perros rapidamente"
        expected = "corr perr rapid"
        self.assertEqual(text_stemming(text), expected)

    def test_empty_string(self):
        """Test that an empty string returns an empty string safely."""
        self.assertEqual(text_stemming(""), "")

    def test_whitespace_only(self):
        """Test that a string of only spaces tokenizes and returns empty."""
        self.assertEqual(text_stemming("     "), "")

    def test_case_handling(self):
        """Test that capitalized words are handled (Snowball stems to lowercase)."""
        text = "Gatos y Perros"
        # 'Gatos' -> 'gat', 'y' -> 'y', 'Perros' -> 'perr'
        expected = "gat y perr"
        self.assertEqual(text_stemming(text), expected)

    def test_with_punctuation(self):
        """Test behavior with punctuation. word_tokenize separates them, stemmer ignores them."""
        text = "perros, gatos."
        # word_tokenize makes: ['perros', ',', 'gatos', '.']
        expected = "perr , gat ."
        self.assertEqual(text_stemming(text), expected)

    def test_no_change_needed(self):
        """Test words that are already stems or don't get changed by the algorithm."""
        text = "sal mar sol"
        expected = "sal mar sol"
        self.assertEqual(text_stemming(text), expected)

# ---------------------------------------------------------
# Tests for the text_lemmatization() function
# ---------------------------------------------------------

class TestTextLemmatization(unittest.TestCase):

    def test_verbs_and_plurals(self):
        """Test lemmatization of conjugated verbs and plural nouns."""
        # 'comiendo' -> 'comer', 'gatos' -> 'gato', 'cantaron' -> 'cantar'
        text = "comiendo gatos cantaron"
        expected = "comer gato cantar"
        self.assertEqual(text_lemmatization(text), expected)

    def test_empty_string(self):
        """Test that an empty string is handled safely."""
        self.assertEqual(text_lemmatization(""), "")

    def test_articles_and_gender(self):
        """Test how spaCy handles articles and gendered adjectives in Spanish."""
        # 'Los' -> 'el', 'perros' -> 'perro', 'blancos' -> 'blanco'
        text = "los perros blancos"
        expected = "el perro blanco"
        self.assertEqual(text_lemmatization(text), expected)

    def test_case_handling(self):
        """Test that capitalized words are lemmatized correctly (usually to lowercase base)."""
        text = "Estaban Jugando"
        # 'Estaban' -> 'estar', 'Jugando' -> 'jugar'
        expected = "estar jugar"
        self.assertEqual(text_lemmatization(text), expected)

    def test_with_punctuation(self):
        """Test behavior with punctuation. spaCy tokenizes punctuation as individual tokens."""
        text = "hola, mundo."
        expected = "hola , mundo ."
        self.assertEqual(text_lemmatization(text), expected)

    def test_no_change_needed(self):
        """Test words that are already in their base form (lemmas)."""
        text = "yo comer pan"
        # 'yo' -> 'yo', 'comer' -> 'comer', 'pan' -> 'pan'
        expected = "yo comer pan"
        self.assertEqual(text_lemmatization(text), expected)

# ---------------------------------------------------------
# Tests for the text_stemming_pystemmer() function
# ---------------------------------------------------------

class TestTextStemmingPyStemmer(unittest.TestCase):

    def test_basic_stemming(self):
        """Test stemming of standard Spanish verbs, plurals, and adverbs."""
        # PyStemmer for Spanish typically stems: 'corriendo' -> 'corr', 'perros' -> 'perr'
        text = "corriendo perros rapidamente"
        expected = "corr perr rapid"
        self.assertEqual(text_stemming_pystemmer(text), expected)

    def test_empty_string(self):
        """Test that an empty string returns an empty string safely."""
        self.assertEqual(text_stemming_pystemmer(""), "")

    def test_whitespace_only(self):
        """Test that a string of only spaces tokenizes and returns an empty string."""
        self.assertEqual(text_stemming_pystemmer("     "), "")

    def test_case_handling(self):
        """Test that capitalized words are handled by the stemmer."""
        text = "Gatos y Perros"
        # PyStemmer preserves the capitalization of the original word!
        expected = "Gat y Perr"
        self.assertEqual(text_stemming_pystemmer(text), expected)

    def test_with_punctuation(self):
        """Test behavior with punctuation. word_tokenize separates them, stemmer ignores them."""
        text = "perros, gatos."
        # word_tokenize makes: ['perros', ',', 'gatos', '.']
        expected = "perr , gat ."
        self.assertEqual(text_stemming_pystemmer(text), expected)

    def test_no_change_needed(self):
        """Test words that are already stems or don't get changed by the algorithm."""
        text = "sal mar sol"
        expected = "sal mar sol"
        self.assertEqual(text_stemming_pystemmer(text), expected)

# ---------------------------------------------------------
# Tests for the text_filtering() function
# ---------------------------------------------------------

class TestTextFiltering(unittest.TestCase):

    def test_full_pipeline(self):
        """Test the complete pipeline: noise removal -> stopwords -> stemming."""
        # Input has: punctuation, uppercase, mentions, URLs, stopwords, and stemmable words.
        text = "¡Hola! @JuanDoe, mira los perros corriendo rápidamente en http://test.com"
        # 1. clean_text: "hola mira los perros corriendo rápidamente en"
        # 2. stopword_remover: "hola mira perros corriendo rápidamente" ('los', 'en' removed)
        # 3. text_stemming: "hol mir perr corr rapid"
        expected = "hol mir perr corr rapid"
        self.assertEqual(text_filtering(text), expected)

    def test_empty_string(self):
        """Test that an empty string passes through the pipeline safely."""
        self.assertEqual(text_filtering(""), "")

    def test_only_noise_and_stopwords(self):
        """Test a string that should be completely reduced to an empty string."""
        text = "@admin #test y en el de la https://nada.com 100!"
        # clean_text -> "y en el de la"
        # stopword -> ""
        # stemming -> ""
        self.assertEqual(text_filtering(text), "")

    def test_numbers_and_punctuation(self):
        """Test that numbers and punctuation are stripped before stemming."""
        text = "100 gatos cantaron, 50 perros ladraron!!!"
        # clean -> "gatos cantaron perros ladraron"
        # stopword -> "gatos cantaron perros ladraron"
        # stem -> "gat cant perr ladr"
        expected = "gat cant perr ladr"
        self.assertEqual(text_filtering(text), expected)

    def test_already_clean_text(self):
        """Test words that don't need cleaning, aren't stopwords, and are already stems."""
        text = "sal sol mar"
        expected = "sal sol mar"
        self.assertEqual(text_filtering(text), expected)

# ---------------------------------------------------------
# Tests for the process_csv() function
# ---------------------------------------------------------

class TestProcessCSV(unittest.TestCase):

    def setUp(self):
        "This is executed BEFORE each test. It creates a clean temporary environment."
        self.test_dir = tempfile.TemporaryDirectory()
        self.input_csv = os.path.join(self.test_dir.name, "input.csv")
        self.output_csv = os.path.join(self.test_dir.name, "output.csv")

    def tearDown(self):
        "This is executed AFTER each test. It deletes the temporary files and folder."
        self.test_dir.cleanup()

    def test_successful_processing(self):
        """Tests that the file is processed, the clean column is created, and it is saved correctly."""
        # 1. Create a mock DataFrame and save it as a CSV in the temporary folder
        df_mock = pd.DataFrame({
            "id": [1, 2],
            "tweet_text": [
                "¡Hola! @JuanDoe, mira los perros en http://test.com",
                "100 gatos corriendo!!!"
            ]
        })
        df_mock.to_csv(self.input_csv, index=False, encoding="utf-8")

        # 2. Run the function to be tested
        process_csv(self.input_csv, self.output_csv, text_column="tweet_text")

        # 3. Verify that the output file exists
        self.assertTrue(os.path.exists(self.output_csv))

        # 4. Read the output file and verify its structure
        df_result = pd.read_csv(self.output_csv, encoding="utf-8")

        # Check that the new column exists and the rows are maintained
        self.assertIn("tweet_text_clean", df_result.columns)
        self.assertEqual(len(df_result), 2)

        # Check that the original column is still there
        self.assertIn("tweet_text", df_result.columns)

    def test_missing_column_raises_error(self):
        """Tests that the function raises a ValueError if the specified column does not exist in the CSV."""
        # 1. Create a CSV without the 'tweet_text' column
        df_mock = pd.DataFrame({
            "usuario": ["@user1", "@user2"],
            "texto_diferente": ["hola", "adiós"]
        })
        df_mock.to_csv(self.input_csv, index=False, encoding="utf-8")

        # 2. Verify that a ValueError is raised during processing
        with self.assertRaises(ValueError) as context:
            process_csv(self.input_csv, self.output_csv, text_column="tweet_text")

        # 3. Verify that the error message includes the correct information
        self.assertIn("Column 'tweet_text' not found", str(context.exception))

if __name__ == '__main__':
    unittest.main()