import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay

def get_data():
    df_tfidf = pd.read_csv("data_train_tfidf.csv", encoding="utf-8")
    df_cleaned = pd.read_csv("data_train_cleaned.csv", encoding="utf-8")

    if len(df_tfidf) != len(df_cleaned):
        raise ValueError("Files do not have the same number of rows.")

    # Optional safety check
    if "tweet_text_clean" in df_tfidf.columns and "tweet_text_clean" in df_cleaned.columns:
        if not df_tfidf["tweet_text_clean"].fillna("").reset_index(drop=True).equals(
            df_cleaned["tweet_text_clean"].fillna("").reset_index(drop=True)
        ):
            raise ValueError("Rows do not match between TF-IDF file and cleaned file.")

    X = df_tfidf.drop(columns=["tweet_text_clean", "class"], errors="ignore")
    y = df_cleaned["class"]

    return X, y

def train_and_plot():
    X, y = get_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rnd_clf = RandomForestClassifier(random_state=42, n_estimators=100)
    rnd_clf.fit(X_train, y_train)

    y_pred = rnd_clf.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred, average="macro"))
    print("Recall:", metrics.recall_score(y_test, y_pred, average="macro"))
    print("F1 Score:", metrics.f1_score(y_test, y_pred, average="macro"))

    # -----------------------------
    # 1. Class distribution graph
    # -----------------------------
    plt.figure(figsize=(8, 5))
    y.value_counts().plot(kind="bar")
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 2. Confusion matrix graph
    # -----------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 3. Top feature importances
    # -----------------------------
    importances = pd.Series(rnd_clf.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    top_features.sort_values().plot(kind="barh")
    plt.title("Top 15 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 4. Train vs test accuracy
    # -----------------------------
    n_trees_list = [1, 5, 10, 20, 50, 100, 150, 200]
    train_scores = []
    test_scores = []

    for n in n_trees_list:
        model = RandomForestClassifier(n_estimators=n, random_state=42)
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_scores.append(metrics.accuracy_score(y_train, train_pred))
        test_scores.append(metrics.accuracy_score(y_test, test_pred))

    plt.figure(figsize=(8, 5))
    plt.plot(n_trees_list, train_scores, marker="o", label="Train Accuracy")
    plt.plot(n_trees_list, test_scores, marker="o", label="Test Accuracy")
    plt.title("Train vs Test Accuracy")
    plt.xlabel("Number of Trees")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

train_and_plot()