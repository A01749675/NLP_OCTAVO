import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    results = {
        "accuracy": metrics.accuracy_score(y_test, y_pred),
        "precision_macro": metrics.precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": metrics.recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": metrics.f1_score(y_test, y_pred, average="macro", zero_division=0),
        "specificity": specificity
    }

    return y_pred, results


def print_metrics(results):
    print("Accuracy:", results["accuracy"])
    print("Precision:", results["precision_macro"])
    print("Recall:", results["recall_macro"])
    print("F1 Score:", results["f1_macro"])
    print("Specificity:", results["specificity"])


def plot_class_distribution(y):
    plt.figure(figsize=(8, 5))
    y.value_counts().plot(kind="bar")
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_random_forest_feature_importance(model, X, top_n=15):
    if not hasattr(model, "feature_importances_"):
        print("This model does not have feature_importances_.")
        return

    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    top_features.sort_values().plot(kind="barh")
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


def plot_logistic_regression_coefficients(model, X, top_n=15):
    classifier = model.named_steps["classifier"]

    if not hasattr(classifier, "coef_"):
        print("This model does not have coefficients.")
        return

    coefficients = pd.Series(classifier.coef_[0], index=X.columns)

    top_positive = coefficients.sort_values(ascending=False).head(top_n)
    top_negative = coefficients.sort_values(ascending=True).head(top_n)

    top_coefficients = pd.concat([top_negative, top_positive])

    plt.figure(figsize=(10, 8))
    top_coefficients.sort_values().plot(kind="barh")
    plt.title(f"Top Logistic Regression Coefficients")
    plt.xlabel("Coefficient Value")
    plt.tight_layout()
    plt.show()


def plot_train_vs_test_accuracy_rf(X_train, X_test, y_train, y_test, random_state=42):
    from sklearn.ensemble import RandomForestClassifier

    n_trees_list = [1, 5, 10, 20, 50, 100, 150, 200]

    train_scores = []
    test_scores = []

    for n in n_trees_list:
        model = RandomForestClassifier(
            n_estimators=n,
            random_state=random_state,
            class_weight="balanced"
        )

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_scores.append(metrics.accuracy_score(y_train, train_pred))
        test_scores.append(metrics.accuracy_score(y_test, test_pred))

    plt.figure(figsize=(8, 5))
    plt.plot(n_trees_list, train_scores, marker="o", label="Train Accuracy")
    plt.plot(n_trees_list, test_scores, marker="o", label="Test Accuracy")
    plt.title("Random Forest Train vs Test Accuracy")
    plt.xlabel("Number of Trees")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()