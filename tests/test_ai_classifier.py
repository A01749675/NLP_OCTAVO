import os
import tempfile
import unittest
import pandas as pd
from unittest.mock import patch

from ai_classifier import (
    read_tweets,
    classify_tweet,
    classify_tweet_few_shot,
    classify_chain_of_thought,
    classify_tweets,
    save_results,
)


class TestAiClassifier(unittest.TestCase):

    def test_read_tweets_reads_and_renames_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "tweets.csv")
            pd.DataFrame({
                "tweet_id": [1],
                "tweet_text": ["hola"],
                "class": ["control"],
            }).to_csv(csv_path, index=False)

            df = read_tweets(csv_path)
            self.assertEqual(list(df.columns), ["tweet_id", "tweet_text", "real_class"])
            self.assertEqual(df.loc[0, "real_class"], "control")

    @patch("ai_classifier.ollama.chat")
    def test_classify_tweet_returns_anorexia(self, mock_chat):
        mock_chat.return_value = {"message": {"content": "anorexia"}}
        self.assertEqual(classify_tweet("texto"), "anorexia")

    @patch("ai_classifier.ollama.chat")
    def test_classify_tweet_defaults_to_control_on_unknown(self, mock_chat):
        mock_chat.return_value = {"message": {"content": "I do not know"}}
        self.assertEqual(classify_tweet("texto"), "control")

    @patch("ai_classifier.ollama.chat")
    def test_classify_tweet_few_shot_uses_few_shot_prompt(self, mock_chat):
        mock_chat.return_value = {"message": {"content": "control"}}
        self.assertEqual(classify_tweet_few_shot("texto"), "control")

    @patch("ai_classifier.ollama.chat")
    def test_classify_chain_of_thought_returns_control(self, mock_chat):
        mock_chat.return_value = {"message": {"content": "control"}}
        self.assertEqual(classify_chain_of_thought("texto"), "control")

    def test_classify_tweets_adds_new_class_column(self):
        df = pd.DataFrame({
            "tweet_id": [1, 2],
            "tweet_text": ["uno", "dos"],
            "real_class": ["control", "anorexia"],
        })

        with patch("ai_classifier.classify_tweet", side_effect=["control", "anorexia"]):
            result_df = classify_tweets(df, few_shot=False)

        self.assertIn("new_class", result_df.columns)
        self.assertEqual(list(result_df["new_class"]), ["control", "anorexia"])

    def test_save_results_writes_expected_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.csv")
            df = pd.DataFrame({
                "tweet_id": [1],
                "tweet_text": ["hola"],
                "real_class": ["control"],
                "new_class": ["control"],
                "extra": ["ignore"],
            })

            save_results(df, output_path)
            persisted = pd.read_csv(output_path)
            self.assertEqual(list(persisted.columns), ["tweet_id", "tweet_text", "real_class", "new_class"])
